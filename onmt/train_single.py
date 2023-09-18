#!/usr/bin/env python
"""Training on a single process."""
import torch
import time

from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.utils.distributed import broadcast_tensors
from onmt.inputters import DynamicDatasetIter
from onmt.transforms import get_transforms_cls


def configure_process(opt, device_id):
    logger.info("logger set device {} ".format(device_id))
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and hasattr(model_opt, 'tensorboard_log_dir_dated'):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        model_opt.update_vocab = opt.update_vocab
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, vocabs_dict, transforms_cls, task_queue_manager):
    """Build iterator used for validation."""
    if not any(opt.data[corpus_id].get('path_valid_src', False) for corpus_id in opt.data.keys()):
        return None
    logger.info("creating validation iterator")
    valid_iter = DynamicDatasetIter.from_opts(
        task_queue_manager=task_queue_manager,
        transforms_cls=transforms_cls,
        vocabs_dict=vocabs_dict,
        opts=opt,
        is_train=False,
    )
    return valid_iter


def init_distributed(model, task_queue_manager):
    my_component_groups = task_queue_manager.get_distributed_groups()
    for (layer_stack_index, encoder_id), (min_rank, group) in my_component_groups['encoder'].items():
        weights = [
            p.data for name, p
            in model.encoder.get_submodule(layer_stack_index, encoder_id).named_parameters()
            if 'embeddings' not in name and 'adapter' not in name
        ]
        broadcast_tensors(weights, src=min_rank, group=group)

    for (layer_stack_index, decoder_id), (min_rank, group) in my_component_groups['decoder'].items():
        weights = [
            p.data for name, p
            in model.decoder.get_submodule(layer_stack_index, decoder_id).named_parameters()
            if 'embeddings' not in name and 'adapter' not in name
        ]
        broadcast_tensors(weights, src=min_rank, group=group)

    for (src_lang,), (min_rank, group) in my_component_groups['src_emb'].items():
        embs = model.encoder.embeddings[f'embeddings_{src_lang}']
        weights = [p.data for p in embs.parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

    for (tgt_lang,), (min_rank, group) in my_component_groups['tgt_emb'].items():
        embs = model.decoder.embeddings[f'embeddings_{tgt_lang}']
        weights = [p.data for p in embs.parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

        weights = [p.data for p in model.generator[f'generator_{tgt_lang}'].parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

    for adapter_id, (min_rank, group) in my_component_groups['encoder_adapters'].items():
        layer_stack_index, encoder_id, adapter_group, sub_id = adapter_id
        adapter = model.encoder.get_submodule(layer_stack_index, encoder_id).get_adapter(adapter_group, sub_id)
        weights = [p.data for name, p in adapter.named_parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

    for adapter_id, (min_rank, group) in my_component_groups['decoder_adapters'].items():
        layer_stack_index, decoder_id, adapter_group, sub_id = adapter_id
        adapter = model.decoder.get_submodule(layer_stack_index, decoder_id).get_adapter(adapter_group, sub_id)
        weights = [p.data for name, p in adapter.named_parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

    weights = [p.data for p in model.attention_bridge.parameters()]
    broadcast_tensors(weights, src=0)

    logger.debug('After init_distributed')
    for name, p in model.named_parameters():
        logger.debug(f'{task_queue_manager.node_rank}:{task_queue_manager.local_rank} {name}: {p.flatten()[:10]}')


# def iter_on_device(iterator, device_context):
#     if device_context.is_gpu():
#         device = torch.device(f'cuda:{device_context.local_rank}')
#     else:
#         device = torch.device('cpu')
#     for batch, meta, comm_batch_id in iterator:
#         yield batch.to(device), meta, comm_batch_id


def main(
    opt,
    vocabs_dict,
    device_context,
    error_queue=None,
    batch_queue=None,
    semaphore=None,
    task_queue_manager=None,
):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    # N.B: task_queue_manager is already local

    init_logger(opt.log_file, gpu_id=device_context.id)
    if device_context.is_distributed():
        sleep_s = device_context.local_rank * 3
        logger.warning(f'sleeping {sleep_s}s to alleviate ROCm deadlock')
        time.sleep(sleep_s)
        configure_process(opt, device_context.local_rank)
        gpu_rank_t = torch.distributed.get_rank()
        logger.info("RANK GPU FROM TORCH %s", str(gpu_rank_t))

    transforms_cls = get_transforms_cls(opt._all_transform)
    checkpoint = None
    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    # Build model.

    model, generators_md = build_model(model_opt, opt, vocabs_dict, task_queue_manager, checkpoint)

    logger.info("{} - Init model".format(device_context.id))
    if device_context.is_distributed():
        init_distributed(model, task_queue_manager)
    else:
        # Initialize some data structures
        _ = task_queue_manager.get_distributed_groups()
    enc, dec = model.count_parameters(log=logger.debug)
    logger.info("{} - total encoder parameters: {}".format(device_context.id, enc))
    logger.info("{} - total decoder parameters: {}".format(device_context.id, dec))

    # Build optimizer.
    logger.info("{} - Build optimizer".format(device_context.id))
    optim = Optimizer.from_opt(
        model,
        opt,
        task_queue_manager=task_queue_manager,
        checkpoint=checkpoint,
    )

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, vocabs_dict, optim, device_context)

    logger.info("{} - Build trainer".format(device_context.id))
    trainer = build_trainer(
        opt,
        device_context,
        model,
        vocabs_dict,
        optim,
        task_queue_manager=task_queue_manager,
        model_saver=model_saver,
        generators_md=generators_md,
    )
    logger.info("{} - Trainer built".format(device_context.id))

    if batch_queue is None:
        train_iter = DynamicDatasetIter.from_opts(
            task_queue_manager=task_queue_manager,
            transforms_cls=transforms_cls,
            vocabs_dict=vocabs_dict,
            opts=opt,
            is_train=True,
        )
        # TODO: check that IterOnDevice is unnecessary here; corpora should be already on device
        # if device_context.is_gpu():
        #     train_iter = IterOnDevice(_train_iter, device_context.local_rank)
        # else:
        #     train_iter = IterOnDevice(_train_iter, -1)
    else:
        assert semaphore is not None, "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch, metadata, communication_batch_id = batch_queue.get()
                semaphore.release()
                # TODO: confirm that batch-providing corpus has already been to'd to the correct place
                yield batch, metadata, communication_batch_id

        train_iter = _train_iter()
    # train_iter = iter_on_device(train_iter, device_context)
    logger.info("Device {} - Valid iter".format(device_context.id))
    valid_iter = _build_valid_iter(opt, vocabs_dict, transforms_cls, task_queue_manager)
    # if valid_iter is not None:
    #    valid_iter = iter_on_device(valid_iter, device_context)

    if len(opt.gpu_ranks):
        if device_context.is_master():
            logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        if device_context.is_master():
            logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    logger.info("{} - Starting training".format(device_context.id))
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps,
        device_context=device_context,
    )

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
