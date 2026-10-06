"""KV-cached decoding in the Translator path must match full-prefix recompute.

Covers the pieces that have to agree for the cache to be usable:
- NativeTransformerWrapper: rotary offset + per-stack cache_offset
- GreedySearch / BeamSearch: cache pruning/reordering as sentences finish
- encoder memory tiling (one copy per beam, along the batch dim only)
"""
import unittest

import torch
import torch.nn as nn

from mammoth.modules.transformer import (
    DecoderBlock, NativeTransformerWrapper, RotaryEmbedding, TransformerStack,
)
from mammoth.translate.beam_search import BeamSearch
from mammoth.translate.greedy_search import GreedySearch
from mammoth.translate.translator import run_decode_loop

PAD, BOS, EOS, UNK = 0, 1, 2, 3
VOCAB, DIM, HEADS = 24, 32, 4


class _ScorerStub:
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.0
        self.has_cov_pen = False
        self.has_len_pen = False


def _build_decoder(stack_depths=(2, 1), seed=0) -> NativeTransformerWrapper:
    """Random float64 decoder with several stacks and wrapper-level rotary."""
    torch.manual_seed(seed)
    stacks = []
    for i, depth in enumerate(stack_depths):
        blocks = nn.ModuleList([DecoderBlock(DIM, HEADS, ff_mult=2) for _ in range(depth)])
        stacks.append(TransformerStack(blocks, nn.RMSNorm(DIM) if i == len(stack_depths) - 1 else None,
                                       DIM, i, f"dec_{i}"))
    token_emb = nn.Embedding(VOCAB, DIM)
    to_logits = nn.Linear(DIM, VOCAB, bias=False)
    # Larger scale -> peaky logits, so EOS is emitted at varied steps.
    nn.init.normal_(token_emb.weight, std=2.0)
    nn.init.normal_(to_logits.weight, std=1.0)
    wrapper = NativeTransformerWrapper(
        token_emb=token_emb,
        post_emb_norm=nn.RMSNorm(DIM),
        stacks=stacks,
        rotary_emb=RotaryEmbedding(DIM // HEADS),
        to_logits=to_logits,
    )
    # The wrapper only holds plain refs to the shared modules; keep them
    # registered somewhere so .double()/.eval() reach them.
    holder = nn.ModuleList([token_emb, to_logits, wrapper._post_emb_norm, *stacks])
    object.__setattr__(wrapper, "_holder", holder)  # keep alive without registering
    holder.double().eval()
    wrapper.eval()
    return wrapper


class TestWrapperIncrementalDecoding(unittest.TestCase):
    def test_cached_incremental_logits_match_full_forward(self):
        dec = _build_decoder(stack_depths=(2, 1))
        batch, src_len, steps = 3, 7, 9
        ctx = torch.randn(batch, src_len, DIM, dtype=torch.float64)
        ctx_mask = torch.ones(batch, src_len, dtype=torch.bool)
        ids = torch.randint(4, VOCAB, (batch, steps))

        with torch.no_grad():
            full = dec(ids, context=ctx, context_mask=ctx_mask)
            cache = dec.new_kv_cache()
            self.assertEqual(len(cache.layers), 3)  # 2 + 1 blocks, flat
            for t in range(steps):
                step_logits, cache = dec(
                    ids[:, t:t + 1], context=ctx, context_mask=ctx_mask,
                    return_intermediates=True, cache=cache,
                )
                torch.testing.assert_close(step_logits[:, 0], full[:, t], atol=1e-9, rtol=1e-9)

        # Every block got its own cache slot with the right prefix length.
        for layer in cache.layers:
            self.assertEqual(layer.self_k.size(2), steps)


def _run_search(strategy_name, use_kv_cache, beam_size=1, batch_size=4, seed=0):
    dec = _build_decoder(seed=seed)
    torch.manual_seed(100 + seed)
    src_len = 6
    enc = torch.randn(batch_size, src_len, DIM, dtype=torch.float64)
    mask = torch.ones(batch_size, src_len, dtype=torch.bool)
    common = dict(
        pad=PAD, bos=BOS, eos=EOS, unk=UNK, batch_size=batch_size,
        global_scorer=_ScorerStub(), min_length=0, max_length=14,
        block_ngram_repeat=0, exclusion_tokens=set(), ban_unk_token=False,
        device=torch.device("cpu"),
    )
    if strategy_name == "greedy":
        strat = GreedySearch(sampling_temp=1.0, keep_topk=1, keep_topp=0, beam_size=1, **common)
    else:
        strat = BeamSearch(
            beam_size=beam_size, n_best=2, stepwise_penalty=False, ratio=0.0,
            dtype=torch.float64, **common,
        )
    strat.initialize(encoder_output=enc, src_mask=mask)
    with torch.no_grad():
        run_decode_loop(dec, strat, use_kv_cache=use_kv_cache)
    return strat


class TestCachedSearchMatchesUncached(unittest.TestCase):
    def _assert_same(self, a, b):
        self.assertEqual(len(a.predictions), len(b.predictions))
        for pa, pb, sa, sb in zip(a.predictions, b.predictions, a.scores, b.scores):
            self.assertEqual(len(pa), len(pb))
            for x, y in zip(pa, pb):
                self.assertTrue(torch.equal(x, y), f"{x.tolist()} != {y.tolist()}")
            for x, y in zip(sa, sb):
                torch.testing.assert_close(torch.as_tensor(x), torch.as_tensor(y), atol=1e-8, rtol=1e-8)

    def test_greedy(self):
        for seed in range(3):
            ref = _run_search("greedy", use_kv_cache=False, seed=seed)
            got = _run_search("greedy", use_kv_cache=True, seed=seed)
            self._assert_same(ref, got)

    def test_beam(self):
        for beam_size in (2, 4):
            for seed in range(3):
                ref = _run_search("beam", use_kv_cache=False, beam_size=beam_size, seed=seed)
                got = _run_search("beam", use_kv_cache=True, beam_size=beam_size, seed=seed)
                self._assert_same(ref, got)

    def test_sentences_finish_at_different_steps(self):
        # Guard the guard: the equivalence tests above only exercise cache
        # pruning if sentences really finish at different times.
        lengths = set()
        for seed in range(3):
            strat = _run_search("beam", use_kv_cache=True, beam_size=3, seed=seed)
            lengths |= {len(p[0]) for p in strat.predictions}
        self.assertGreater(len(lengths), 1)


class TestEncoderMemoryTiling(unittest.TestCase):
    def test_encoder_output_tiled_only_along_batch(self):
        batch, beam, src_len, d = 2, 5, 7, 4
        enc = torch.randn(batch, src_len, d)
        mask = torch.ones(batch, src_len, dtype=torch.bool)
        common = dict(
            pad=PAD, bos=BOS, eos=EOS, unk=UNK, batch_size=batch,
            global_scorer=_ScorerStub(), min_length=0, max_length=5,
            block_ngram_repeat=0, exclusion_tokens=set(), ban_unk_token=False,
            device=torch.device("cpu"),
        )
        beam_search = BeamSearch(beam_size=beam, n_best=1, stepwise_penalty=False, ratio=0.0, **common)
        greedy = GreedySearch(sampling_temp=1.0, keep_topk=1, keep_topp=0, beam_size=beam, **common)
        for strat in (beam_search, greedy):
            strat.initialize(encoder_output=enc, src_mask=mask)
            self.assertEqual(tuple(strat.encoder_output_tiled.shape), (batch * beam, src_len, d))
            self.assertEqual(tuple(strat.src_mask_tiled.shape), (batch * beam, src_len))
            # rows are grouped per sentence: [s0 x beam, s1 x beam, ...]
            for row in range(batch * beam):
                self.assertTrue(torch.equal(strat.encoder_output_tiled[row], enc[row // beam]))


if __name__ == "__main__":
    unittest.main()


class TestKvCacheOption(unittest.TestCase):
    def _parser(self):
        import configargparse
        from mammoth.opts import _add_decoding_opts

        parser = configargparse.ArgumentParser()
        _add_decoding_opts(parser, include_reproducibility=False)
        return parser

    def test_kv_cache_defaults_to_on(self):
        self.assertTrue(self._parser().parse_args([]).kv_cache)

    def test_kv_cache_accepts_true_false_values(self):
        parser = self._parser()
        for text in ("false", "False", "no", "0", "off"):
            self.assertFalse(parser.parse_args(["--kv_cache", text]).kv_cache, text)
        for text in ("true", "True", "yes", "1", "on"):
            self.assertTrue(parser.parse_args(["--kv_cache", text]).kv_cache, text)
        self.assertTrue(parser.parse_args(["--kv_cache"]).kv_cache)  # bare flag == true

    def test_kv_cache_rejects_garbage(self):
        with self.assertRaises(SystemExit):
            self._parser().parse_args(["--kv_cache", "maybe"])

    def test_kv_cache_from_yaml_config(self):
        import tempfile, os
        parser = self._parser()
        parser.add_argument("-c", "--config", is_config_file=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.yaml")
            with open(path, "w") as f:
                f.write("kv_cache: false\n")
            self.assertFalse(parser.parse_args(["-c", path]).kv_cache)

    def test_translator_forwards_flag_to_decode_loop(self):
        from unittest import mock
        from mammoth.translate import translator as translator_mod

        for use_cache in (True, False):
            # Bypass __init__: only the attributes the decode step touches are set.
            tr = object.__new__(translator_mod.Translator)
            tr.use_kv_cache = use_cache
            tr.logger = None
            tr._device = torch.device("cpu")
            tr._tgt_bos_idx = BOS
            tr.task = mock.Mock()
            tr.task.corpus_opts = {}
            tr.model = mock.Mock()
            tr._run_encoder = mock.Mock(return_value=("enc_out", "src_mask"))
            tr._gold_score = mock.Mock(return_value=[0])
            tr.report_results = mock.Mock()
            strategy = mock.Mock()

            with mock.patch.object(translator_mod, "run_decode_loop") as loop:
                tr._translate_batch_with_strategy(mock.Mock(batch_size=1), None, strategy)

            self.assertEqual(loop.call_count, 1)
            self.assertEqual(loop.call_args.kwargs["use_kv_cache"], use_cache)

    def test_no_cache_path_does_not_create_cache(self):
        strat = _run_search("greedy", use_kv_cache=False)
        self.assertIsNone(strat.cache)
        strat = _run_search("greedy", use_kv_cache=True)
        self.assertIsNotNone(strat.cache)
