#!/usr/bin/env python
# -*- coding: utf-8 -*-
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


def _build_task_from_task_id(opts):
    """Build a TaskSpecs from --task_id (the existing path)."""
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

    task = TaskSpecs(
        node_rank=0,
        local_rank=0,
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
    return task, [task], uses_adapters, corpus_opts


def _build_tasks_for_zero_shot(opts):
    """Build a synthetic TaskSpecs for zero-shot inference and collect all
    donor tasks from the config so all components are loaded."""
    encoder_id = opts.encoder_id
    decoder_id = opts.decoder_id
    src_lang = opts.src_lang
    tgt_lang = opts.tgt_lang

    # Synthetic task for the zero-shot combination
    zero_shot_task = TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        encoder_id=encoder_id,
        decoder_id=decoder_id,
        corpus_id='__zero_shot__',
        weight=1.0,
        introduce_at_training_step=0,
        corpus_opts={},
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )

    # Build donor tasks from all tasks in the config so all components get created
    donor_tasks = []
    for corpus_id, corpus_opts in opts.tasks.items():
        task_src_lang, task_tgt_lang = corpus_opts['src_tgt'].split('-', 1)
        task_encoder_id = corpus_opts.get('enc_sharing_group', [task_src_lang])
        task_decoder_id = corpus_opts.get('dec_sharing_group', [task_tgt_lang])
        donor_tasks.append(TaskSpecs(
            node_rank=0,
            local_rank=0,
            src_lang=task_src_lang,
            tgt_lang=task_tgt_lang,
            encoder_id=task_encoder_id,
            decoder_id=task_decoder_id,
            corpus_id=corpus_id,
            weight=1.0,
            introduce_at_training_step=0,
            corpus_opts=corpus_opts,
            src_vocab=None,
            tgt_vocab=None,
            encoder_adapter_ids=None,
            decoder_adapter_ids=None,
        ))

    return zero_shot_task, donor_tasks, False, {}


def translate(opts):
    ArgumentParser.validate_prepare_opts(opts)
    ArgumentParser.validate_translate_opts(opts)
    ArgumentParser.validate_translate_opts_dynamic(opts)
    logger = init_logger(opts.log_file)

    # Determine if we're using task-based or component-based (zero-shot) inference
    is_zero_shot = opts.task_id is None
    if is_zero_shot:
        if not all([opts.encoder_id, opts.decoder_id, opts.src_lang, opts.tgt_lang]):
            raise ValueError(
                "For zero-shot inference, must provide all of: "
                "--encoder_id, --decoder_id, --src_lang, --tgt_lang"
            )
        task, all_tasks, uses_adapters, corpus_opts = _build_tasks_for_zero_shot(opts)
        single_task = None
    else:
        task, all_tasks, uses_adapters, corpus_opts = _build_task_from_task_id(opts)
        single_task = task.corpus_id

    if use_gpu(opts):
        context_enum = DeviceContextEnum.SINGLE_GPU
        gpus_per_node = 1
    else:
        context_enum = DeviceContextEnum.CPU
        gpus_per_node = 0

    world_context = WorldContext(
        context=context_enum,
        n_nodes=1,
        gpus_per_node=gpus_per_node,
    )

    task_queue_manager = TaskQueueManager(
        tasks=all_tasks,
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=None,
        uses_adapters=uses_adapters,
    ).global_to_local(
        node_rank=0,
        local_rank=0,
        opts=opts,
    )
    # FIXME: fix the attention bridge in translation
    task_queue_manager.create_all_distributed_components(
        use_attention_bridge=False,     # (opts.ab_layers is not None and len(opts.ab_layers) != 0),
    )

    translator = build_translator(
        opts, task_queue_manager, task, logger=logger, report_score=True,
        single_task=single_task,
    )
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
    corpus_label = task.corpus_id if not is_zero_shot else f"zero-shot {task.src_lang}->{task.tgt_lang}"
    logger.info(f"Inference transforms for task '{corpus_label}': {transform}")

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
