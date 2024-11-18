#!/usr/bin/env python
"""Training on a single process."""
import torch
import time

from mammoth.model_builder import build_model, validate_optimizer_coverage
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.misc import set_random_seed
from mammoth.trainer import build_trainer
from mammoth.utils.model_saver import build_model_saver, load_parameters_from_checkpoint
from mammoth.utils.logging import init_logger, logger
from mammoth.utils.parse import ArgumentParser

from mammoth.distributed import broadcast_tensors
from mammoth.inputters import DynamicDatasetIter
from mammoth.transforms import get_transforms_cls


def configure_process(opts, device_id):
    logger.info("logger set device {} ".format(device_id))
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opts.seed, device_id >= 0)


def _get_model_opts(opts, frame_checkpoint=None):
    """Get `model_opts` to build model, may load from `checkpoint` if any."""
    if frame_checkpoint is not None:
        model_opts = ArgumentParser.checkpoint_model_opts(frame_checkpoint["opts"])
        ArgumentParser.update_model_opts(model_opts)
        ArgumentParser.validate_model_opts(model_opts)
        if opts.tensorboard_log_dir == model_opts.tensorboard_log_dir and \
                hasattr(model_opts, 'tensorboard_log_dir_dated'):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opts.tensorboard_log_dir_dated = model_opts.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        # model_opts.update_vocab = opts.update_vocab
    else:
        model_opts = opts
    return model_opts


def _build_valid_iter(opts, vocabs_dict, transforms_cls, task_queue_manager):
    """Build iterator used for validation."""
    if not any(opts.tasks[corpus_id].get('path_valid_src', False) for corpus_id in opts.tasks.keys()):
        logger.info("Validation set missing for: {}".format(
            [
                corpus_id for corpus_id in opts.tasks.keys()
                if not opts.tasks[corpus_id].get('path_valid_src', False)
            ]
        ))
        return None
    logger.info("creating validation iterator")
    valid_iter = DynamicDatasetIter.from_opts(
        task_queue_manager=task_queue_manager,
        transforms_cls=transforms_cls,
        vocabs_dict=vocabs_dict,
        opts=opts,
        is_train=False,
        line_idx_restore=None,
    )
    return valid_iter


def init_distributed(model, task_queue_manager):
    # All components on device, in consistent order across devices
    my_components = task_queue_manager.get_my_distributed_components()
    # Omit components not found elsewhere, as these don't need to be communicated
    components_to_communicate = [
        component for component in my_components
        if component.needs_communication()
    ]

    for component in components_to_communicate:
        weights = [p.data for name, p in component.named_parameters(model)]
        broadcast_tensors(weights, src=component.min_rank, group=component.group)

    logger.debug('After init_distributed')
    for name, p in model.named_parameters():
        logger.debug(f'{task_queue_manager.node_rank}:{task_queue_manager.local_rank} {name}: {p.flatten()[:10]}')


def main(
    opts,
    vocabs_dict,
    device_context,
    error_queue=None,
    batch_queue=None,
    semaphore=None,
    task_queue_manager=None,
    frame_checkpoint=None,
    frame_checkpoint_path=None,
):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opts`` has been validated and updated
    # at this point.
    # N.B: task_queue_manager is already local

    init_logger(opts.log_file, gpu_id=device_context.id)
    if device_context.is_distributed():
        sleep_s = device_context.local_rank * 3
        logger.warning(f'sleeping {sleep_s}s to alleviate ROCm deadlock')
        time.sleep(sleep_s)
        configure_process(opts, device_context.local_rank)
        gpu_rank_t = torch.distributed.get_rank()
        logger.info("RANK GPU FROM TORCH %s", str(gpu_rank_t))

    transforms_cls = get_transforms_cls(opts._all_transform)
    model_opts = _get_model_opts(opts, frame_checkpoint=frame_checkpoint)

    task_queue_manager.create_all_distributed_components(
        use_attention_bridge=(model_opts.ab_layers is not None and len(model_opts.ab_layers) != 0),
    )

    # Build model.
    model = build_model(model_opts, opts, vocabs_dict, task_queue_manager)

    logger.info("{} - Init model".format(device_context.id))
    if device_context.is_distributed():
        init_distributed(model, task_queue_manager)
    enc, dec = model.count_parameters()
    logger.info("{} - total encoder parameters: {}".format(device_context.id, enc))
    logger.info("{} - total decoder parameters: {}".format(device_context.id, dec))

    # Build optimizer.
    logger.info("{} - Build optimizer".format(device_context.id))
    optim = MultipleOptimizer.from_opts(
        model,
        opts,
        task_queue_manager=task_queue_manager,
        frame_checkpoint=frame_checkpoint,
    )
    logger.info("{} - total optimized parameters: {}".format(
        device_context.id,
        optim.count_parameters()
    ))
    validate_optimizer_coverage(model, optim)

    # Load parameters from checkpoint
    if opts.train_from:
        load_parameters_from_checkpoint(
            frame_checkpoint_path=frame_checkpoint_path,
            model=model,
            optim=optim,
            task_queue_manager=task_queue_manager,
            reset_optim=opts.reset_optim in {'all', 'states'},
        )
        optim.global_training_step = frame_checkpoint['global_training_step']

    # Build model saver
    model_saver = build_model_saver(
        model_opts,
        opts,
        model,
        vocabs_dict,
        optim,
        task_queue_manager=task_queue_manager,
    )

    logger.info("{} - Build trainer".format(device_context.id))
    trainer = build_trainer(
        opts,
        device_context,
        model,
        vocabs_dict,
        optim,
        task_queue_manager=task_queue_manager,
        model_saver=model_saver,
    )
    logger.info("{} - Trainer built".format(device_context.id))

    # It is no longer possible to train without multiprocessing
    assert batch_queue is not None
    assert semaphore is not None

    def _train_iter():
        while True:
            batch, metadata, communication_batch_id = batch_queue.get()
            semaphore.release()
            # TODO: confirm that batch-providing corpus has already been to'd to the correct place
            yield batch, metadata, communication_batch_id

    train_iter = _train_iter()
    # train_iter = iter_on_device(train_iter, device_context)
    valid_iter = _build_valid_iter(opts, vocabs_dict, transforms_cls, task_queue_manager)

    if len(opts.gpu_ranks):
        if device_context.is_master():
            logger.info('Starting training on GPU: %s' % opts.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opts.train_steps
    logger.info("{} - Starting training".format(device_context.id))
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opts.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opts.valid_steps,
        device_context=device_context,
    )

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
