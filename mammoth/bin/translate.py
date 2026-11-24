#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import torch

from mammoth.utils.logging import init_logger
from mammoth.utils.misc import split_corpus
from mammoth.translate.translator import build_translator
# from mammoth.inputters.text_dataset import InferenceDataReader
from mammoth.transforms import get_transforms_cls, make_transforms, TransformPipe

import mammoth.opts as opts
from mammoth.distributed import TaskSpecs, TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum
from mammoth.distributed.tasks import get_adapter_ids
from mammoth.utils.parse import ArgumentParser
from mammoth.utils.misc import use_gpu


# Per-task input/output keys: tasks.<task_id>.{src,tgt,output} override the top-level options.
TASK_IO_KEYS = ('src', 'tgt', 'output')


def resolve_task_runs(opts):
    """One copy of opts per requested task id, each with a single task_id and that task's src/tgt/output."""
    task_ids = opts.task_id if isinstance(opts.task_id, list) else [opts.task_id]
    unknown = [task_id for task_id in task_ids if task_id not in opts.tasks]
    if unknown:
        raise ValueError(f'Unknown task_id {unknown}; the config defines {list(opts.tasks)}')
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f'task_id lists a task more than once: {task_ids}')

    runs = []
    for task_id in task_ids:
        task_opts = copy.copy(opts)
        task_opts.task_id = task_id
        for key in TASK_IO_KEYS:
            value = opts.tasks[task_id].get(key)
            if value is not None:
                setattr(task_opts, key, value)
        if not task_opts.src:
            raise ValueError(f"Task '{task_id}' has no source file: set tasks.{task_id}.src in the config or --src")
        runs.append(task_opts)

    outputs = [task_opts.output for task_opts in runs]
    shared = sorted({output for output in outputs if outputs.count(output) > 1})
    if shared:
        raise ValueError(f"Several tasks would write to {shared}: set 'output' under each task in the config")
    return runs


def translate(opts):
    ArgumentParser.validate_prepare_opts(opts)
    ArgumentParser.validate_translate_opts(opts)
    ArgumentParser.validate_translate_opts_dynamic(opts)
    logger = init_logger(opts.log_file)

    runs = resolve_task_runs(opts)
    if len(runs) == 1:
        translate_task(runs[0], logger)
        return

    # Several tasks: a failing task does not stop the others; report all failures at the end.
    failed = []
    for i, task_opts in enumerate(runs, 1):
        logger.info(f"Task {i}/{len(runs)}: '{task_opts.task_id}' ({task_opts.src} -> {task_opts.output})")
        try:
            translate_task(task_opts, logger)
        except Exception:
            logger.exception(f"Task '{task_opts.task_id}' failed")
            failed.append(task_opts.task_id)
    if failed:
        raise RuntimeError(f'{len(failed)} of {len(runs)} tasks failed: {failed}')


def translate_task(opts, logger):
    """Translate opts.src with the single task opts.task_id and write opts.output."""
    corpus_id = opts.task_id
    corpus_opts = opts.tasks[corpus_id]
    src_lang, tgt_lang = corpus_opts['src_tgt'].split('-', 1)
    encoder_id = corpus_opts.get('enc_sharing_group', [src_lang])
    decoder_id = corpus_opts.get('dec_sharing_group', [tgt_lang])
    if 'adapters' in corpus_opts:
        encoder_adapter_ids = get_adapter_ids(opts, corpus_opts, 'encoder')
        decoder_adapter_ids = get_adapter_ids(opts, corpus_opts, 'decoder')
        uses_adapters = True
    else:
        encoder_adapter_ids = None
        decoder_adapter_ids = None
        uses_adapters = False

    node_rank = 0
    local_rank = 0
    if use_gpu(opts):
        context_enum = DeviceContextEnum.SINGLE_GPU
        gpus_per_node = 1
    else:
        context_enum = DeviceContextEnum.CPU
        gpus_per_node = 0

    task = TaskSpecs(
        node_rank=node_rank,
        local_rank=local_rank,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        encoder_id=encoder_id,
        decoder_id=decoder_id,
        corpus_id=corpus_id,
        weight=1.0,
        introduce_at_training_step=0,
        corpus_opts=corpus_opts,
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=encoder_adapter_ids,
        decoder_adapter_ids=decoder_adapter_ids,
    )

    world_context = WorldContext(
        context=context_enum,
        n_nodes=1,
        gpus_per_node=gpus_per_node,
    )

    task_queue_manager = TaskQueueManager(
        tasks=[task],
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=None,
        uses_adapters=uses_adapters,
    ).global_to_local(
        node_rank=node_rank,
        local_rank=local_rank,
        opts=opts,
    )
    # FIXME: fix the attention bridge in translation
    task_queue_manager.create_all_distributed_components(
        use_attention_bridge=False,     # (opts.ab_layers is not None and len(opts.ab_layers) != 0),
    )
    
    translator = build_translator(opts, task_queue_manager, task, logger=logger, report_score=True)
    # data_reader = InferenceDataReader(opts.src, opts.tgt, opts.src_feats)
    src_shards = split_corpus(opts.src, opts.shard_size)
    tgt_shards = split_corpus(opts.tgt, opts.shard_size)
    features_shards = []
    features_names = []
    for feat_name, feat_path in opts.src_feats.items():
        features_shards.append(split_corpus(feat_path, opts.shard_size))
        features_names.append(feat_name)
    shard_pairs = zip(src_shards, tgt_shards, *features_shards)

    # Build transforms: use the task-level transforms list (not just global opts.transforms,
    # which may be empty when transforms are declared only under the task in the YAML).
    task_transforms = corpus_opts.get('transforms', [])
    all_transform_names = set(opts.transforms) | set(task_transforms)
    transforms_cls = get_transforms_cls(all_transform_names)
    transforms = make_transforms(opts, transforms_cls, translator.vocabs, task=task)
    data_transform = [
        transforms[name] for name in task_transforms if name in transforms
    ]
    transform = TransformPipe.build_from(data_transform)
    logger.info(f"Inference transforms for task '{corpus_id}': {transform}")

    for i, (src_shard, tgt_shard, *feats_shard) in enumerate(shard_pairs):
        logger.info("Translating shard %d." % i)
        translator.translate_dynamic(
            src=src_shard,
            transform=transform,
            # src_feats=feats_shard,  # TODO: put me back in
            tgt=tgt_shard,
            batch_size=opts.batch_size,
            batch_type=opts.batch_type,
            attn_debug=opts.attn_debug,
            align_debug=opts.align_debug,
        )
    translator.out_file.close()
    # Free this task's model before the next task loads its own.
    del translator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_parser():
    parser = ArgumentParser(description='translate.py')
    parser.translation = True

    opts.dynamic_prepare_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def main():
    parser = _get_parser()

    opts = parser.parse_args()
    translate(opts)


if __name__ == "__main__":
    main()
