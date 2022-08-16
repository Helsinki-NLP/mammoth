#!/usr/bin/env python
"""Train models with dynamic data."""
import sys
import torch
from functools import partial
import os

from onmt.utils.distributed import ErrorHandler, consumer, batch_producer, Scheduler
from onmt.utils.misc import set_random_seed
from onmt.modules.embeddings import prepare_pretrained_embeddings
from onmt.utils.logging import init_logger, logger

from onmt.models.model_saver import load_checkpoint
from onmt.train_single import main as single_main
from onmt.inputters.dynamic_iterator import DynamicDatasetIter

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts
from onmt.inputters.corpus import save_transformed_sample
from onmt.inputters.fields import build_dynamic_fields, save_fields, load_fields, build_dynamic_fields_langspec
from onmt.transforms import make_transforms, save_transforms, get_specials, get_transforms_cls
from collections import OrderedDict
from onmt.constants import ModelTask

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def prepare_fields_transforms(opt):
    """Prepare or dump fields & transforms before training."""
    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    fields = build_dynamic_fields(opt, src_specials=specials['src'], tgt_specials=specials['tgt'])

    # maybe prepare pretrained embeddings, if any
    prepare_pretrained_embeddings(opt, fields)

    if opt.dump_fields:
        save_fields(fields, opt.save_data, overwrite=opt.overwrite)
    if opt.dump_transforms or opt.n_sample != 0:
        transforms = make_transforms(opt, transforms_cls, fields)
    if opt.dump_transforms:
        save_transforms(transforms, opt.save_data, overwrite=opt.overwrite)
    if opt.n_sample != 0:
        logger.warning(
            f"`-n_sample` != 0: Training will not be started. Stop after saving {opt.n_sample} samples/corpus."
        )
        save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
        logger.info("Sample saved, please check it before restart training.")
        sys.exit()
    return fields, transforms_cls


def _init_train(opt):
    """Common initilization stuff for all training process."""
    ArgumentParser.validate_prepare_opts(opt)

    if opt.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(ckpt_path=opt.train_from)
        fields = load_fields(opt.save_data, checkpoint)
        transforms_cls = get_transforms_cls(opt._all_transform)
        if (
            hasattr(checkpoint["opt"], '_all_transform')
            and len(opt._all_transform.symmetric_difference(checkpoint["opt"]._all_transform)) != 0
        ):
            _msg = "configured transforms is different from checkpoint:"
            new_transf = opt._all_transform.difference(checkpoint["opt"]._all_transform)
            old_transf = checkpoint["opt"]._all_transform.difference(opt._all_transform)
            if len(new_transf) != 0:
                _msg += f" +{new_transf}"
            if len(old_transf) != 0:
                _msg += f" -{old_transf}."
            logger.warning(_msg)
        if opt.update_vocab:
            logger.info("Updating checkpoint vocabulary with new vocabulary")
            fields, transforms_cls = prepare_fields_transforms(opt)
    else:
        checkpoint = None
        fields, transforms_cls = prepare_fields_transforms(opt)

    # Report src and tgt vocab sizes
    for side in ['src', 'tgt']:
        f = fields[side]
        try:
            f_iter = iter(f)
        except TypeError:
            f_iter = [(side, f)]
        for sn, sf in f_iter:
            if sf.use_vocab:
                logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
    return checkpoint, fields, transforms_cls


def init_train_prepare_fields_transforms(opt, vocab_path, side):
    """Prepare or dump fields & transforms before training."""

    fields = build_dynamic_fields_langspec(opt, vocab_path, side)
    transforms_cls = get_transforms_cls(opt._all_transform)
    # TODO: maybe prepare pretrained embeddings, if any, with `prepare_pretrained_embeddings(opt, fields)`

    if opt.dump_fields:
        save_fields(fields, opt.save_data, overwrite=opt.overwrite)
    if opt.dump_transforms or opt.n_sample != 0:
        transforms = make_transforms(opt, transforms_cls, fields)
    if opt.dump_transforms:
        save_transforms(transforms, opt.save_data, overwrite=opt.overwrite)
    if opt.n_sample != 0:
        logger.warning(
            f"`-n_sample` != 0: Training will not be started. Stop after saving {opt.n_sample} samples/corpus."
        )
        save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
        logger.info("Sample saved, please check it before restart training.")
        sys.exit()

    for name, field in fields[side].fields:
        logger.debug(f'prepped: {name}  {len(field.vocab)}')

    return fields


def train(opt):
    init_logger(opt.log_file)
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)
    ArgumentParser.validate_prepare_opts(opt)
    set_random_seed(opt.seed, False)

    n_gpu = len(opt.gpu_ranks)
    # set PyTorch distributed related environment variables
    current_env = os.environ
    current_env["WORLD_SIZE"] = str(opt.world_size)
    # if `n_gpu` differs from `opt.world_size` we assume the training runs on multiple nodes
    if n_gpu != int(opt.world_size):
        current_env["MASTER_ADDR"] = opt.master_ip
        current_env["MASTER_PORT"] = str(opt.master_port)
        num_nodes = current_env.get("SLURM_NNODES", 1)
        node_rank = int(current_env.get("SLURM_NODEID", 0))
    else:
        num_nodes = 1
        node_rank = 0
    logger.info("Training on {} node(s)".format(num_nodes))

    opt.data_task = ModelTask.SEQ2SEQ
    transforms_cls = get_transforms_cls(opt._all_transform)

    fields_dict = OrderedDict()
    # For creating fields, we use a scheduler that doesn't filter by node and gpu
    global_scheduler = Scheduler(opt, node_rank=None, local_rank=None)

    for side in ('src', 'tgt'):
        for lang, vocab_path in global_scheduler.get_vocabularies(opt, side=side):
            fields_dict[(side, lang)] = init_train_prepare_fields_transforms(opt, vocab_path=vocab_path, side=side)
    # for key, val in fields_dict:
    #     print(f'{key}:\t{val}')

    train_process = partial(single_main, fields_dict=fields_dict)

    logger.debug(f"[{os.getpid()}] Initializing process group with: {current_env}")

    if opt.world_size > 1:

        queues = []
        mp = torch.multiprocessing.get_context('spawn')
        logger.info("world_size = {}, queue_size = {}".format(opt.world_size, opt.queue_size))
        semaphore = mp.Semaphore(opt.world_size * opt.queue_size)
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        producers = []

        for local_rank in range(n_gpu):
            # This scheduler will only yield the items that are active on this gpu
            scheduler = Scheduler(opt, node_rank=node_rank, local_rank=local_rank)

            # each process's rank
            global_rank = n_gpu * node_rank + local_rank
            current_env["RANK"] = str(global_rank)
            current_env["LOCAL_RANK"] = str(local_rank)

            q = mp.Queue(opt.queue_size)
            queues += [q]
            procs.append(
                mp.Process(
                    target=consumer,
                    args=(train_process, opt, global_rank, error_queue, q, semaphore, node_rank, local_rank),
                    daemon=True,
                )
            )
            procs[local_rank].start()
            logger.info(" Starting process pid: %d  " % procs[local_rank].pid)
            error_handler.add_child(procs[local_rank].pid)

            # Get the iterator to generate from
            # We can't stride here without losing data: each dataset only goes to one GPU
            train_iter = DynamicDatasetIter.from_opts(
                scheduler=scheduler,
                transforms_cls=transforms_cls,
                fields_dict=fields_dict,
                opts=opt,
                is_train=True,
                stride=1,
                offset=0,
            )

            producer = mp.Process(
                target=batch_producer, args=(train_iter, queues[local_rank], semaphore, opt, local_rank), daemon=True
            )
            producers.append(producer)
            producers[local_rank].start()
            logger.info(" Starting producer process pid: {}  ".format(producers[local_rank].pid))
            error_handler.add_child(producers[local_rank].pid)

        for p in procs:
            logger.info("DD logger")
            p.join()
        # Once training is done, we can terminate the producers
        for p in producers:
            p.terminate()

    elif n_gpu == 1:  # case 1 GPU only
        train_process(opt, global_rank=0, local_rank=0)
    else:  # case only CPU
        train_process(opt, global_rank=None, local_rank=None)


def _get_parser():
    parser = ArgumentParser(description='train.py')
    train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opt, unknown = parser.parse_known_args()
    train(opt)


if __name__ == "__main__":
    main()
