#!/usr/bin/env python
"""Train models with dynamic data."""
import torch
from functools import partial
import os

from onmt.utils.distributed import (
    DeviceContext,
    DeviceContextEnum,
    ErrorHandler,
    TaskQueueManager,
    WorldContext,
    batch_producer,
    consumer,
)
from onmt.utils.misc import set_random_seed
# from onmt.modules.embeddings import prepare_pretrained_embeddings
from onmt.utils.logging import init_logger, logger

from onmt.models.model_saver import load_checkpoint
from onmt.train_single import main as single_main
from onmt.inputters_mvp import DynamicDatasetIter

from onmt.utils.parse import ArgumentParser
from onmt.opts import train_opts
from onmt.inputters_mvp import get_vocab, DEFAULT_SPECIALS
from onmt.transforms import get_transforms_cls
from collections import OrderedDict
from onmt.constants import ModelTask

# Set sharing strategy manually instead of default based on the OS.
torch.multiprocessing.set_sharing_strategy('file_system')


# def prepare_fields_transforms(opt):
#     """Prepare or dump fields & transforms before training."""
#     transforms_cls = get_transforms_cls(opt._all_transform)
#     specials = get_specials(opt, transforms_cls)
#
#     fields = build_dynamic_fields(opt, src_specials=specials['src'], tgt_specials=specials['tgt'])
#
#     # maybe prepare pretrained embeddings, if any
#     prepare_pretrained_embeddings(opt, fields)
#
#     if opt.dump_fields:
#         save_fields(fields, opt.save_data, overwrite=opt.overwrite)
#     if opt.dump_transforms or opt.n_sample != 0:
#         transforms = make_transforms(opt, transforms_cls, fields)
#     if opt.dump_transforms:
#         save_transforms(transforms, opt.save_data, overwrite=opt.overwrite)
#     if opt.n_sample != 0:
#         logger.warning(
#             f"`-n_sample` != 0: Training will not be started. Stop after saving {opt.n_sample} samples/corpus."
#         )
#         save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
#         logger.info("Sample saved, please check it before restart training.")
#         sys.exit()
#     return fields, transforms_cls

# TODO: reimplement save_transformed_sample

def _init_train(opt):
    """Common initilization stuff for all training process."""
    ArgumentParser.validate_prepare_opts(opt)

    if opt.train_from:
        # Load checkpoint if we resume from a previous training.
        checkpoint = load_checkpoint(ckpt_path=opt.train_from)
        # fields = load_fields(opt.save_data, checkpoint)
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
            # fields, transforms_cls = prepare_fields_transforms(opt)
    else:
        checkpoint = None
        # fields, transforms_cls = prepare_fields_transforms(opt)

    # Report src and tgt vocab sizes
    # for side in ['src', 'tgt']:
    #     f = fields[side]
    #     try:
    #         f_iter = iter(f)
    #     except TypeError:
    #         f_iter = [(side, f)]
    #     for sn, sf in f_iter:
    #         if sf.use_vocab:
    #             logger.info(' * %s vocab size = %d' % (sn, len(sf.vocab)))
    return checkpoint, None, transforms_cls


# def init_train_prepare_fields_transforms(opt, vocab_path, side):
#     """Prepare or dump fields & transforms before training."""
#
#     fields = None # build_dynamic_fields_langspec(opt, vocab_path, side)
#     transforms_cls = get_transforms_cls(opt._all_transform)
#     # TODO: maybe prepare pretrained embeddings, if any, with `prepare_pretrained_embeddings(opt, fields)`
#
#     # if opt.dump_fields:
#     #     save_fields(fields, opt.save_data, overwrite=opt.overwrite)
#     if opt.dump_transforms or opt.n_sample != 0:
#         transforms = make_transforms(opt, transforms_cls, fields)
#     if opt.dump_transforms:
#         save_transforms(transforms, opt.save_data, overwrite=opt.overwrite)
#     if opt.n_sample != 0:
#         logger.warning(
#             f"`-n_sample` != 0: Training will not be started. Stop after saving {opt.n_sample} samples/corpus."
#         )
#         save_transformed_sample(opt, transforms, n_sample=opt.n_sample)
#         logger.info("Sample saved, please check it before restart training.")
#         sys.exit()
#
#     for name, field in fields[side].fields:
#         logger.debug(f'prepped: {name}  {len(field.vocab)}')
#
#     return fields


def validate_slurm_node_opts(current_env, world_context, opt):
    """If you are using slurm, confirm that opts match slurm environment variables"""
    slurm_n_nodes = int(current_env['SLURM_NNODES'])
    if slurm_n_nodes != world_context.n_nodes:
        raise ValueError(
            f'Looks like you are running on {slurm_n_nodes} slurm nodes, '
            f'but set n_nodes to {world_context.n_nodes} in the conf'
        )
    slurm_node_id = int(current_env['SLURM_NODEID'])
    if slurm_node_id != opt.node_rank:
        raise ValueError(
            f'Looks like you are running on slurm node {slurm_node_id}, '
            f'but set node_rank to {opt.node_rank} on the command line'
        )


def train(opt):
    init_logger(opt.log_file)
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)
    ArgumentParser.validate_prepare_opts(opt)
    set_random_seed(opt.seed, False)

    # set PyTorch distributed related environment variables
    current_env = os.environ
    current_env["WORLD_SIZE"] = str(opt.world_size)
    world_context = WorldContext.from_opt(opt)
    if 'SLURM_NNODES' in current_env:
        validate_slurm_node_opts(current_env, world_context, opt)
    logger.info(f'Training on {world_context}')

    opt.data_task = ModelTask.SEQ2SEQ

    transforms_cls = get_transforms_cls(opt._all_transform)
    if transforms_cls:
        logger.info(f'All transforms: {transforms_cls}')
        src_specials, tgt_specials = zip(*(cls.get_specials(opt) for cls in transforms_cls.values()))
        all_specials = set(DEFAULT_SPECIALS)
        for special_group in src_specials + tgt_specials:
            all_specials = all_specials | special_group
        # all_specials = set(src_specials + tgt_specials + DEFAULT_SPECIALS)
        all_specials = tuple(sorted(all_specials))  # get_vocab produces distinct lists
    else:
        logger.info('No transforms found')
        all_specials = tuple(sorted(DEFAULT_SPECIALS))  # get_vocab produces distinct lists

    vocabs_dict = OrderedDict()
    # For creating fields, we use a task_queue_manager that doesn't filter by node and gpu
    global_task_queue_manager = TaskQueueManager.from_opt(opt, world_context)

    vocab_size = {'src': opt.src_vocab_size or None, 'tgt': opt.tgt_vocab_size or None}
    for side in ('src', 'tgt'):
        for lang in global_task_queue_manager.get_langs(side):
            vocab_path = opt.__getattribute__(f'{side}_vocab')[lang]
            # FIXME: for now, all specials are passed to all vocabs, this could be finer-grained
            vocabs_dict[(side, lang)] = get_vocab(vocab_path, lang, vocab_size[side], specials=all_specials)
    # for key, val in fields_dict:
    #     print(f'{key}:\t{val}')

    train_process = partial(single_main, vocabs_dict=vocabs_dict)

    logger.debug(f"[{os.getpid()}] Initializing process group with: {current_env}")

    if world_context.context == DeviceContextEnum.MULTI_GPU:
        current_env["MASTER_ADDR"] = opt.master_ip
        current_env["MASTER_PORT"] = str(opt.master_port)
        node_rank = opt.node_rank

        queues = []
        semaphores = []
        mp = torch.multiprocessing.get_context('spawn')
        logger.info("world_size = {}, queue_size = {}".format(opt.world_size, opt.queue_size))
        # Create a thread to listen for errors in the child processes.
        error_queue = mp.SimpleQueue()
        error_handler = ErrorHandler(error_queue)
        # Train with multiprocessing.
        procs = []
        producers = []

        for local_rank in range(world_context.gpus_per_node):
            device_context: DeviceContext = world_context.global_to_local(
                node_rank=node_rank,
                local_rank=local_rank,
            )
            # This task_queue_manager will only yield the items that are active on this gpu
            task_queue_manager = global_task_queue_manager.global_to_local(
                node_rank=node_rank,
                local_rank=local_rank,
                opt=opt
            )

            # store rank in env (FIXME: is this obsolete?)
            current_env["RANK"] = str(device_context.global_rank)
            current_env["LOCAL_RANK"] = str(device_context.local_rank)

            q = mp.Queue(opt.queue_size)
            semaphore = mp.Semaphore(opt.queue_size)
            queues.append(q)
            semaphores.append(q)
            procs.append(
                mp.Process(
                    target=consumer,
                    args=(train_process, opt, device_context, error_queue, q, semaphore, task_queue_manager),
                    daemon=True,
                )
            )
            procs[local_rank].start()
            logger.info(" Starting process pid: %d  " % procs[local_rank].pid)
            error_handler.add_child(procs[local_rank].pid)

            # Get the iterator to generate from
            # We can't stride here without losing data: each dataset only goes to one GPU
            train_iter = DynamicDatasetIter.from_opts(
                task_queue_manager=task_queue_manager,
                transforms_cls=transforms_cls,
                vocabs_dict=vocabs_dict,
                opts=opt,
                is_train=True,
                stride=1,
                offset=0,
            )

            producer = mp.Process(
                target=batch_producer, args=(train_iter, q, semaphore, opt, local_rank), daemon=True
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

    else:
        # SINGLE_GPU or CPU
        device_context: DeviceContext = world_context.global_to_local(
            node_rank=0,
            local_rank=0,
        )
        task_queue_manager = global_task_queue_manager.global_to_local(
            node_rank=0,
            local_rank=0,
            opt=opt
        )
        train_process(opt, device_context=device_context, task_queue_manager=task_queue_manager)


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
