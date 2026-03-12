#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multi-task inference entry point.

Mirrors the training setup: tasks are assigned to GPUs via node_gpu in the
config, and each GPU runs its assigned tasks in a separate process (no NCCL
communication needed — each process is fully independent).

Usage:
    mammoth_translate -config inference.yaml

Each task in the config must define path_src and path_output.
path_tgt is optional (used for reference-based scoring).
"""
import codecs
import os

import torch
import torch.multiprocessing as mp

from mammoth.utils.logging import init_logger
from mammoth.utils.misc import split_corpus, set_random_seed
from mammoth.translate.translator import load_model_for_translation, GNMTGlobalScorer, Translator
from mammoth.transforms import get_transforms_cls, make_transforms, TransformPipe

import mammoth.opts as opts
from mammoth.distributed import TaskSpecs, TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum
from mammoth.distributed.tasks import get_adapter_ids
from mammoth.utils.parse import ArgumentParser
from mammoth.utils.misc import use_gpu


def _build_global_task_queue_manager(opts, world_context):
    """Build a TaskQueueManager with all tasks from opts, without requiring a
    task_distribution_strategy (not needed for inference)."""
    corpus_ids = sorted(opts.tasks.keys())
    n_tasks = len(corpus_ids)

    if world_context.is_distributed():
        node_gpu = [
            tuple(int(y) for y in opts.tasks[corpus_id]["node_gpu"].split(":", 1))
            for corpus_id in corpus_ids
        ]
    else:
        node_gpu = [(0, 0)] * n_tasks

    tasks = []
    uses_adapters = False
    for (node_rank, local_rank), corpus_id in zip(node_gpu, corpus_ids):
        corpus_opts = opts.tasks[corpus_id]
        src_lang, tgt_lang = corpus_opts["src_tgt"].split("-", 1)
        encoder_id = corpus_opts.get("enc_sharing_group", [src_lang])
        decoder_id = corpus_opts.get("dec_sharing_group", [tgt_lang])
        if "adapters" in corpus_opts:
            encoder_adapter_ids = get_adapter_ids(opts, corpus_opts, "encoder")
            decoder_adapter_ids = get_adapter_ids(opts, corpus_opts, "decoder")
            uses_adapters = True
        else:
            encoder_adapter_ids = None
            decoder_adapter_ids = None
        task = TaskSpecs(
            node_rank=node_rank,
            local_rank=local_rank,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            encoder_id=encoder_id,
            decoder_id=decoder_id,
            corpus_id=corpus_id,
            weight=corpus_opts.get("weight", 1.0),
            introduce_at_training_step=0,
            corpus_opts=corpus_opts,
            src_vocab=None,
            tgt_vocab=None,
            encoder_adapter_ids=encoder_adapter_ids,
            decoder_adapter_ids=decoder_adapter_ids,
        )
        tasks.append(task)

    return TaskQueueManager(
        tasks=tasks,
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=None,
        uses_adapters=uses_adapters,
    )


def _translate_on_device(local_rank, node_rank, opts, global_task_queue_manager):
    """Translate all tasks assigned to one GPU.

    Runs in its own process for multi-GPU setups; called directly for
    single-GPU setups.  No inter-GPU communication takes place.
    """
    logger = init_logger(opts.log_file)
    set_random_seed(opts.seed, use_gpu(opts))

    task_queue_manager = global_task_queue_manager.global_to_local(
        node_rank=node_rank,
        local_rank=local_rank,
        opts=opts,
    )
    # FIXME: fix the attention bridge in translation
    task_queue_manager.create_all_distributed_components(use_attention_bridge=False)

    my_tasks = task_queue_manager.get_my_tasks()
    if not my_tasks:
        logger.info(f"Rank {local_rank}: no tasks assigned — exiting.")
        return

    logger.info(f"Rank {local_rank}: tasks = {[t.corpus_id for t in my_tasks]}")

    # Load model ONCE for all tasks on this GPU
    vocabs_dict, model, model_opts = load_model_for_translation(
        opts=opts,
        task_queue_manager=task_queue_manager,
        tasks=my_tasks,
    )
    scorer = GNMTGlobalScorer.from_opts(opts)

    transforms_cls = get_transforms_cls(opts._all_transform)

    for task in my_tasks:
        corpus_opts = task.corpus_opts
        path_src = corpus_opts.get("path_src")
        if not path_src:
            logger.warning(f"Task '{task.corpus_id}' has no path_src — skipping.")
            continue
        path_tgt = corpus_opts.get("path_tgt", f"{task.corpus_id}_pred.txt")

        logger.info(f"Translating '{task.corpus_id}': {path_src} -> {path_tgt}")

        outdir = os.path.dirname(path_tgt)
        if outdir and not os.path.isdir(outdir):
            os.makedirs(outdir, exist_ok=True)
        out_file = codecs.open(path_tgt, "w+", "utf-8")

        # Build a vocab slice with flat 'src'/'tgt' keys expected by the Translator
        task_vocabs = dict(vocabs_dict)
        task_vocabs["src"] = vocabs_dict.get(("src", task.src_lang))
        task_vocabs["tgt"] = vocabs_dict.get(("tgt", task.tgt_lang))

        translator = Translator.from_opts(
            model,
            task_vocabs,
            opts,
            model_opts,
            global_scorer=scorer,
            out_file=out_file,
            report_align=opts.report_align,
            report_score=True,
            logger=logger,
            task=task,
            src_path=path_src,
        )

        transforms = make_transforms(opts, transforms_cls, task_vocabs, task=task)
        data_transform = [transforms[name] for name in opts.transforms if name in transforms]
        transform = TransformPipe.build_from(data_transform)

        for i, src_shard in enumerate(split_corpus(path_src, opts.shard_size)):
            logger.info(f"Task '{task.corpus_id}', shard {i}.")
            translator.translate_dynamic(
                src=src_shard,
                transform=transform,
                batch_size=opts.batch_size,
                batch_type=opts.batch_type,
                attn_debug=opts.attn_debug,
                align_debug=opts.align_debug,
            )

        out_file.close()


def translate(opts):
    ArgumentParser.validate_prepare_opts(opts)
    ArgumentParser.validate_translate_opts(opts)
    ArgumentParser.validate_translate_opts_dynamic(opts)

    logger = init_logger(opts.log_file)
    world_context = WorldContext.from_opts(opts)
    logger.info(f"Inference on {world_context}")

    global_task_queue_manager = _build_global_task_queue_manager(opts, world_context)
    logger.info(f"Tasks: {[t.corpus_id for t in global_task_queue_manager.tasks]}")

    node_rank = int(os.environ.get("SLURM_NODEID", 0))
    n_local_ranks = world_context.gpus_per_node if world_context.gpus_per_node > 0 else 1

    if world_context.is_distributed():
        # One process per GPU — each translates its assigned tasks independently
        mp_ctx = mp.get_context("spawn")
        procs = []
        for local_rank in range(n_local_ranks):
            p = mp_ctx.Process(
                target=_translate_on_device,
                args=(local_rank, node_rank, opts, global_task_queue_manager),
                daemon=True,
            )
            p.start()
            logger.info(f"Spawned inference process for local_rank={local_rank}, pid={p.pid}")
            procs.append(p)
        for p in procs:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Inference process pid={p.pid} exited with code {p.exitcode}")
    else:
        _translate_on_device(0, node_rank, opts, global_task_queue_manager)


def _get_parser():
    parser = ArgumentParser(description='translate.py')
    parser.translation = True

    opts.dynamic_prepare_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
