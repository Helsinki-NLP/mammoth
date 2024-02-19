#!/usr/bin/env python
# -*- coding: utf-8 -*-
from mammoth.utils.logging import init_logger
from mammoth.utils.misc import split_corpus
from mammoth.translate.translator import build_translator
# from mammoth.inputters.text_dataset import InferenceDataReader
from mammoth.transforms import get_transforms_cls, make_transforms, TransformPipe

import mammoth.opts as opts
from mammoth.distributed import TaskSpecs
from mammoth.distributed.tasks import get_adapter_ids
from mammoth.utils.parse import ArgumentParser


def translate(opts):
    ArgumentParser.validate_prepare_opts(opts)
    ArgumentParser.validate_translate_opts(opts)
    ArgumentParser.validate_translate_opts_dynamic(opts)
    logger = init_logger(opts.log_file)

    corpus_id = opts.task_id
    corpus_opts = opts.tasks[corpus_id]
    src_lang, tgt_lang = corpus_opts['src_tgt'].split('-', 1)
    encoder_id = corpus_opts.get('enc_sharing_group', [src_lang])
    decoder_id = corpus_opts.get('dec_sharing_group', [tgt_lang])
    if 'adapters' in corpus_opts:
        encoder_adapter_ids = get_adapter_ids(opts, corpus_opts, 'encoder')
        decoder_adapter_ids = get_adapter_ids(opts, corpus_opts, 'decoder')
    else:
        encoder_adapter_ids = None
        decoder_adapter_ids = None
    task = TaskSpecs(
        node_rank=None,
        local_rank=None,
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
