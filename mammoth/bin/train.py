#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial
import os

from mammoth.distributed import (
    DeviceContext,
    DeviceContextEnum,
    ErrorHandler,
    TaskQueueManager,
    WorldContext,
    batch_producer,
    consumer,
)
from mammoth.utils.misc import set_random_seed
# from mammoth.modules.embeddings import prepare_pretrained_embeddings
from mammoth.utils.logging import init_logger, logger

from mammoth.utils.model_saver import load_frame_checkpoint
from mammoth.train_single import main as single_main
from mammoth.inputters import DynamicDatasetIter

from mammoth.utils.parse import ArgumentParser
from mammoth.opts import train_opts
from mammoth.inputters import get_vocab, DEFAULT_SPECIALS
from mammoth.transforms import get_transforms_cls
from collections import OrderedDict
from mammoth.constants import ModelTask

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


def validate_slurm_node_opts(current_env, world_context, opts):
    """If you are using slurm, confirm that opts match slurm environment variables"""
    slurm_n_nodes = int(current_env['SLURM_NNODES'])
    if slurm_n_nodes != world_context.n_nodes:
        raise ValueError(
            f'Looks like you are running on {slurm_n_nodes} slurm nodes, '
            f'but set n_nodes to {world_context.n_nodes} in the conf'
        )
    slurm_node_id = int(current_env['SLURM_NODEID'])
    if slurm_node_id != opts.node_rank:
        raise ValueError(
            f'Looks like you are running on slurm node {slurm_node_id}, '
            f'but set node_rank to {opts.node_rank} on the command line'
        )


def train(opts):
    init_logger(
        opts.log_file,
        log_file_level=opts.log_file_level,
        structured_log_file=opts.structured_log_file
    )
    ArgumentParser.validate_train_opts(opts)
    ArgumentParser.update_model_opts(opts)
    ArgumentParser.validate_model_opts(opts)
    ArgumentParser.validate_prepare_opts(opts)
    set_random_seed(opts.seed, False)

    # set PyTorch distributed related environment variables
    current_env = os.environ
    current_env["WORLD_SIZE"] = str(opts.world_size)
    world_context = WorldContext.from_opts(opts)
    if 'SLURM_NNODES' in current_env:
        validate_slurm_node_opts(current_env, world_context, opts)
    logger.info(f'Training on {world_context}')

    opts.data_task = ModelTask.SEQ2SEQ

    transforms_cls = get_transforms_cls(opts._all_transform)
    if transforms_cls:
        logger.info(f'All transforms: {transforms_cls}')
        src_specials, tgt_specials = zip(*(cls.get_specials(opts) for cls in transforms_cls.values()))
        all_specials = set(DEFAULT_SPECIALS)
        for special_group in src_specials + tgt_specials:
            all_specials = all_specials | special_group
        # all_specials = set(src_specials + tgt_specials + DEFAULT_SPECIALS)
        all_specials = tuple(sorted(all_specials))  # get_vocab produces distinct lists
    else:
        logger.info('No transforms found')
        all_specials = tuple(sorted(DEFAULT_SPECIALS))  # get_vocab produces distinct lists
    logger.info(f'Final all specials: {all_specials}')

    vocabs_dict = OrderedDict()
    # For creating fields, we use a task_queue_manager that doesn't filter by node and gpu
    global_task_queue_manager = TaskQueueManager.from_opts(opts, world_context)

    frame_checkpoint = None
    checkpoint_path = None
    if opts.train_from:
        frame_checkpoint, checkpoint_path = load_frame_checkpoint(checkpoint_path=opts.train_from)
        vocabs_dict = frame_checkpoint.get('vocab')
    else:
        vocab_size = {'src': opts.src_vocab_size or None, 'tgt': opts.tgt_vocab_size or None}
        for side in ('src', 'tgt'):
            for lang in global_task_queue_manager.get_langs(side):
                vocab_path = opts.__getattribute__(f'{side}_vocab')[lang]
                # FIXME: for now, all specials are passed to all vocabs, this could be finer-grained
                vocabs_dict[(side, lang)] = get_vocab(vocab_path, lang, vocab_size[side], specials=all_specials)
    # for key, val in fields_dict:
    #     print(f'{key}:\t{val}')

    train_process = partial(single_main, vocabs_dict=vocabs_dict)

    logger.debug(f"[{os.getpid()}] Initializing process group with: {current_env}")

    if world_context.context == DeviceContextEnum.MULTI_GPU:
        current_env["MASTER_ADDR"] = opts.master_ip
        current_env["MASTER_PORT"] = str(opts.master_port)
        node_rank = opts.node_rank
        n_local_ranks = world_context.gpus_per_node
    else:
        n_local_ranks = 1

    queues = []
    semaphores = []
    mp = torch.multiprocessing.get_context('spawn')
    logger.info("world_size = {}, queue_size = {}".format(opts.world_size, opts.queue_size))
    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)
    # Train with multiprocessing.
    procs = []
    producers = []

    for local_rank in range(n_local_ranks):
        if world_context.context == DeviceContextEnum.MULTI_GPU:
            device_context: DeviceContext = world_context.global_to_local(
                node_rank=node_rank,
                local_rank=local_rank,
            )
            # store rank in env (FIXME: is this obsolete?)
            current_env["RANK"] = str(device_context.global_rank)
            current_env["LOCAL_RANK"] = str(device_context.local_rank)
        else:
            # Running in a non-distributed context: either single GPU or CPU
            node_rank = 0
            local_rank = 0
            device_context: DeviceContext = world_context.global_to_local(
                node_rank=0,
                local_rank=0,
            )

        # This task_queue_manager will only yield the items that are active on this gpu
        # for the consumer (trainer process)
        task_queue_manager = global_task_queue_manager.global_to_local(
            node_rank=node_rank,
            local_rank=local_rank,
            opts=opts
        )
        if device_context.is_master():
            # Enough to log this once
            logger.info(f'TaskQueueManager: {global_task_queue_manager}')

        q = mp.Queue(opts.queue_size)
        semaphore = mp.Semaphore(opts.queue_size)
        queues.append(q)
        semaphores.append(semaphore)
        procs.append(
            mp.Process(
                target=consumer,
                args=(
                    train_process,
                    opts,
                    device_context,
                    error_queue,
                    q,
                    semaphore,
                    task_queue_manager,
                    frame_checkpoint,
                    checkpoint_path,
                ),
                daemon=True,
            )
        )
        procs[local_rank].start()
        logger.info(" Starting process pid: %d  " % procs[local_rank].pid)
        error_handler.add_child(procs[local_rank].pid)

        # This task_queue_manager will only yield the items that are active on this gpu
        # for the producer (dataloader process)
        task_queue_manager = global_task_queue_manager.global_to_local(
            node_rank=node_rank,
            local_rank=local_rank,
            opts=opts
        )
        # Get the iterator to generate from
        line_idx_restore = None
        if frame_checkpoint is not None:
            line_idx_restore = frame_checkpoint['data_state']
        train_iter = DynamicDatasetIter.from_opts(
            task_queue_manager=task_queue_manager,
            transforms_cls=transforms_cls,
            vocabs_dict=vocabs_dict,
            opts=opts,
            is_train=True,
            line_idx_restore=line_idx_restore,
        )

        producer = mp.Process(
            target=batch_producer, args=(train_iter, q, semaphore, opts, local_rank), daemon=True
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


def _get_parser():
    parser = ArgumentParser(description='train.py')
    train_opts(parser)
    return parser


def main():
    parser = _get_parser()

    opts, unknown = parser.parse_known_args()
    train(opts)


if __name__ == "__main__":
    main()
