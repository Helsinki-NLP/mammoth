#!/usr/bin/env python
"""Training on a single process."""
import torch

from onmt.inputters.inputter import IterOnDevice
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.utils.distributed import broadcast_tensors, Scheduler, is_master
from onmt.inputters.dynamic_iterator import DynamicDatasetIter
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


def _build_valid_iter(opt, fields, transforms_cls):
    """Build iterator used for validation."""
    # valid_iter = DynamicDatasetIter(
    #     fields, transforms_cls, opt, is_train=False)
    valid_iter = iter([])  # FIXME: validation temporarily disabled
    return valid_iter


def init_distributed(model, scheduler):
    my_component_groups = scheduler.get_distributed_groups()
    for encoder_id, (min_rank, group) in my_component_groups['encoder']:
        weights = [
            p.data for name, p in model.encoder[f'encoder{encoder_id}'].named_parameters() if 'embeddings' not in name
        ]
        broadcast_tensors(weights, src=min_rank, group=group)

    for decoder_id, (min_rank, group) in my_component_groups['decoder']:
        weights = [
            p.data for name, p in model.decoder[f'decoder{decoder_id}'].named_parameters() if 'embeddings' not in name
        ]
        broadcast_tensors(weights, src=min_rank, group=group)

    for src_emb_id, (min_rank, group) in my_component_groups['src_emb']:
        src_lang, encoder_id = src_emb_id
        embs = model.encoder[f'encoder{encoder_id}'].embeddings[f'embeddings{src_lang}']
        weights = [p.data for p in embs.parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

    for tgt_emb_id, (min_rank, group) in my_component_groups['tgt_emb']:
        tgt_lang, decoder_id = tgt_emb_id
        embs = model.decoder[f'decoder{decoder_id}'].embeddings[f'embeddings{tgt_lang}']
        weights = [p.data for p in embs.parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

        weights = [p.data for p in model.generator[f'generator{tgt_lang}'].parameters()]
        broadcast_tensors(weights, src=min_rank, group=group)

    weights = [p.data for p in model.attention_bridge.parameters()]
    broadcast_tensors(weights, src=0)

    logger.debug('After init_distributed')
    for name, p in model.named_parameters():
        logger.debug(f'{scheduler.node_rank}:{scheduler.local_rank} {name}: {p.flatten()[:10]}')


def main(
    opt,
    fields_dict,
    global_rank,
    error_queue=None,
    batch_queue=None,
    semaphore=None,
    node_rank=None,
    local_rank=None,
):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    scheduler = Scheduler(opt, node_rank=node_rank, local_rank=local_rank)

    init_logger(opt.log_file)
    if node_rank is not None and local_rank is not None:
        configure_process(opt, local_rank)
        gpu_rank_t = torch.distributed.get_rank()
        logger.info("RANK GPU FROM TORCH %s", str(gpu_rank_t))

    transforms_cls = get_transforms_cls(opt._all_transform)
    checkpoint = None
    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    # Build model.
    model, generators_md = build_model(model_opt, opt, fields_dict, scheduler, checkpoint)

    logger.info("GPU {} - Init model".format(global_rank))
    if node_rank is not None and local_rank is not None:
        init_distributed(model, scheduler)
    enc, dec = model.count_parameters(log=logger.debug)
    logger.info("GPU {} - total encoder parameters: {}".format(global_rank, enc))
    logger.info("GPU {} - total decoder parameters: {}".format(global_rank, dec))

    # Build optimizer.
    logger.info("GPU {} - Build optimizer".format(global_rank))
    optim = Optimizer.from_opt(
        model,
        opt,
        scheduler=scheduler,
        checkpoint=checkpoint,
    )

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields_dict, optim, global_rank)

    logger.info("GPU {} - Build trainer".format(global_rank))
    trainer = build_trainer(
        opt,
        local_rank,
        model,
        fields_dict,
        optim,
        scheduler=scheduler,
        model_saver=model_saver,
        generators_md=generators_md,
    )
    logger.info("GPU {} - Trainer built".format(global_rank))

    if batch_queue is None:
        _train_iter = DynamicDatasetIter.from_opts(
            scheduler=scheduler,
            transforms_cls=transforms_cls,
            fields_dict=fields_dict,
            opts=opt,
            is_train=True,
            stride=1,
            offset=0,
        )
        train_iter = IterOnDevice(_train_iter, local_rank)
    else:
        assert semaphore is not None, "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch, metadata, communication_batch_id = batch_queue.get()
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, local_rank)
                yield batch, metadata, communication_batch_id

        train_iter = _train_iter()
    logger.info("GPU {} - Valid iter".format(global_rank))
    valid_iter = _build_valid_iter(opt, fields_dict, transforms_cls)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, local_rank)

    if len(opt.gpu_ranks):
        if is_master(global_rank):
            logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        if is_master(global_rank):
            logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0
    logger.info("GPU {} - Starting training".format(global_rank))
    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps,
        global_rank=global_rank,
    )

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
