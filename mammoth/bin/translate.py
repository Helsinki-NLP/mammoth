#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mammoth.utils.logging import init_logger
from mammoth.utils.misc import split_corpus
from mammoth.translate.translator import build_translator
# from mammoth.inputters.text_dataset import InferenceDataReader
from mammoth.transforms import get_transforms_cls, make_transforms, TransformPipe

import mammoth.opts as opts
from mammoth.distributed import TaskSpecs
from mammoth.utils.parse import ArgumentParser


def translate(opts):
    ArgumentParser.validate_translate_opts(opts)
    ArgumentParser._get_all_transform_translate(opts)
    ArgumentParser._validate_transforms_opts(opts)
    ArgumentParser.validate_translate_opts_dynamic(opts)
    logger = init_logger(opts.log_file)

    encoder_adapter_ids = set()
    for layer_stack_idx, stack in enumerate(opts.stack['encoder']):
        if 'adapters' in stack:
            for group_id, sub_id in stack['adapters']:
                encoder_adapter_ids.add((layer_stack_idx, group_id, sub_id))
    decoder_adapter_ids = set()
    for layer_stack_idx, stack in enumerate(opts.stack['decoder']):
        if 'adapters' in stack:
            for group_id, sub_id in stack['adapters']:
                decoder_adapter_ids.add((layer_stack_idx, group_id, sub_id))

    logger.info(
        'It is ok that src_vocab and tgt_vocab are None here. '
        'The vocabs are separately loaded in model_builder.'
    )
    task = TaskSpecs(
        node_rank=None,
        local_rank=None,
        src_lang=opts.src_lang,
        tgt_lang=opts.tgt_lang,
        encoder_id=[stack['id'] for stack in opts.stack['encoder']],
        decoder_id=[stack['id'] for stack in opts.stack['decoder']],
        corpus_id='trans',
        weight=1,
        corpus_opt=dict(),
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=encoder_adapter_ids,
        decoder_adapter_ids=decoder_adapter_ids,
    )

    translator = build_translator(opts, task, logger=logger, report_score=True)

    # data_reader = InferenceDataReader(opts.src, opts.tgt, opts.src_feats)
    src_shards = split_corpus(opts.src, opts.shard_size)
    tgt_shards = split_corpus(opts.tgt, opts.shard_size)
    features_shards = []
    features_names = []
    for feat_name, feat_path in opts.src_feats.items():
        features_shards.append(split_corpus(feat_path, opts.shard_size))
        features_names.append(feat_name)
    shard_pairs = zip(src_shards, tgt_shards, *features_shards)

    # Build transforms
    transforms_cls = get_transforms_cls(opts._all_transform)
    transforms = make_transforms(opts, transforms_cls, translator.vocabs, task=task)
    data_transform = [
        transforms[name] for name in opts.transforms if name in transforms
    ]
    transform = TransformPipe.build_from(data_transform)

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
            align_debug=opts.align_debug
        )


def _get_parser():
    parser = ArgumentParser(description='translate.py')

    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    opts.build_bilingual_model(parser)
    return parser


def main():
    parser = _get_parser()

    opts = parser.parse_args()
    translate(opts)


if __name__ == "__main__":
    main()
