#!/usr/bin/env python
"""Get vocabulary coutings from transformed corpora samples."""
from collections import Counter, defaultdict
from mammoth.utils.logging import init_logger
from mammoth.utils.misc import set_random_seed, check_path
from mammoth.utils.parse import ArgumentParser
from mammoth.opts import dynamic_prepare_opts
from mammoth.inputters import build_vocab_counts
from mammoth.transforms import make_transforms, get_transforms_cls
from mammoth.distributed import TaskSpecs


def build_vocab_main(opts):
    """Apply transforms to samples of specified data and build vocab from it.

    Transforms that need vocab will be disabled in this.
    Built vocab is saved in plain text format as following and can be pass as
    `-src_vocab` (and `-tgt_vocab`) when training:
    ```
    <tok_0>\t<count_0>
    <tok_1>\t<count_1>
    ```
    """

    ArgumentParser.validate_prepare_opts(opts, build_vocab_only=True)
    assert opts.n_sample == -1 or opts.n_sample > 1, f"Illegal argument n_sample={opts.n_sample}."

    logger = init_logger()
    set_random_seed(opts.seed, False)
    transforms_cls = get_transforms_cls(opts._all_transform)

    src_counters_by_lang = defaultdict(Counter)
    tgt_counters_by_lang = defaultdict(Counter)

    for corpus_id in opts.tasks:
        lang_pair = opts.tasks[corpus_id]['src_tgt']
        src_lang, tgt_lang = lang_pair.split('-')
        task = TaskSpecs(
            node_rank=None,
            local_rank=None,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            encoder_id=src_lang,
            decoder_id=tgt_lang,
            corpus_id=f'{src_lang}-{tgt_lang}',
            weight=1,
            corpus_opts=dict(),
            src_vocab=None,
            tgt_vocab=None,
            encoder_adapter_ids=None,
            decoder_adapter_ids=None,
        )

        transforms = make_transforms(opts, transforms_cls, vocabs=None, task=task)

        logger.info(f"Counter vocab from {corpus_id} {opts.n_sample} samples.")
        src_counter, tgt_counter = build_vocab_counts(
            opts, corpus_id=corpus_id, transforms=transforms, n_sample=opts.n_sample
        )
        src_counters_by_lang[src_lang].update(src_counter)
        tgt_counters_by_lang[tgt_lang].update(tgt_counter)
        # TODO: implement src_feats

    def save_counter(counter, save_path):
        check_path(save_path, exist_ok=opts.overwrite, log=logger.warning)
        with open(save_path, "w", encoding="utf8") as fo:
            for tok, count in counter.most_common():
                fo.write(tok + "\t" + str(count) + "\n")

    if opts.share_vocab:
        raise Exception('--share_vocab not supported')
        # src_counter += tgt_counter
        # tgt_counter = src_counter
        # logger.info(f"Counters after share:{len(src_counter)}")
        # save_counter(src_counter, opts.src_vocab)

    # TODO: vocab configurability is somewhat limited at the moment.
    # Only language-specific vocabs are possible.
    # Any attempt at vocab sharing between languages will cause the following to fail.
    # Reimplementing --share_vocab may not be optimal
    # (it should mean combining all sources and all targets into one big vocab?).
    # Perhaps we should gracefully handle the setting where several languages point to the same vocab file?

    for src_lang, src_counter in src_counters_by_lang.items():
        logger.info(f"=== Source lang: {src_lang}")

        logger.info(f"Counters src:{len(src_counter)}")
        # for feat_name, feat_counter in src_feats_counter.items():
        #     logger.info(f"Counters {feat_name}:{len(feat_counter)}")

        logger.info(f"Saving to {opts.src_vocab[src_lang]}")
        save_counter(src_counter, opts.src_vocab[src_lang])

        # for k, v in src_feats_counter.items():
        #     save_counter(v, opts.src_feats_vocab[k])

    for tgt_lang, tgt_counter in tgt_counters_by_lang.items():
        logger.info(f"=== Target lang: {tgt_lang}")
        logger.info(f"Counters tgt:{len(tgt_counter)}")
        logger.info(f"Saving to {opts.tgt_vocab[tgt_lang]}")
        save_counter(tgt_counter, opts.tgt_vocab[tgt_lang])


def _get_parser():
    parser = ArgumentParser(description='build_vocab.py')
    dynamic_prepare_opts(parser, build_vocab_only=True)
    return parser


def main():
    parser = _get_parser()
    opts, unknown = parser.parse_known_args()
    build_vocab_main(opts)


if __name__ == '__main__':
    main()
