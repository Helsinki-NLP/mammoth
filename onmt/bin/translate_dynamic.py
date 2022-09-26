#!/usr/bin/env python
# -*- coding: utf-8 -*-
from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
from onmt.inputters.text_dataset import InferenceDataReader
from onmt.transforms import get_transforms_cls, make_transforms, TransformPipe

import onmt.opts as opts
from onmt.utils.distributed import TaskSpecs
from onmt.utils.parse import ArgumentParser


def translate(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    logger = init_logger(opt.log_file)

    # only src_lang and tgt_lang are used at the moment
    task = TaskSpecs(
        node_rank=None,
        local_rank=None,
        src_lang=opt.src_lang,
        tgt_lang=opt.tgt_lang,
        encoder_id=opt.enc_id if opt.enc_id else opt.src_lang,
        decoder_id=opt.dec_id if opt.dec_id else opt.tgt_lang,
        corpus_id='trans',
        weight=1,
        corpus_opt=dict(),
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=opt.enc_adapters,
        decoder_adapter_ids=opt.dec_adapters,
    )

    translator = build_translator(opt, task, logger=logger, report_score=True)

    data_reader = InferenceDataReader(opt.src, opt.tgt, opt.src_feats)

    # Build transforms
    transforms_cls = get_transforms_cls(opt._all_transform)
    transforms = make_transforms(opt, transforms_cls, translator.fields, task=task)
    data_transform = [
        transforms[name] for name in opt.transforms if name in transforms
    ]
    transform = TransformPipe.build_from(data_transform)

    for i, (src_shard, tgt_shard, feats_shard) in enumerate(data_reader):
        logger.info("Translating shard %d." % i)
        translator.translate_dynamic(
            src=src_shard,
            transform=transform,
            src_feats=feats_shard,
            tgt=tgt_shard,
            batch_size=opt.batch_size,
            batch_type=opt.batch_type,
            attn_debug=opt.attn_debug,
            align_debug=opt.align_debug
            )


def _get_parser():
    parser = ArgumentParser(description='translate_dynamic.py')

    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    opts.build_bilingual_model(parser)
    return parser


def main():
    parser = _get_parser()

    opt = parser.parse_args()
    translate(opt)


if __name__ == "__main__":
    main()
