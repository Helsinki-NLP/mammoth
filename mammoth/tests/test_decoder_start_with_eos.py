"""Tests for BART-style decoder_start_with_eos plumbing at inference time.

BART was trained so the decoder sequence is [</s>, <s>, content..., </s>].
At inference, the decoder must therefore be primed with </s> (EOS),
not <s> (BOS) — otherwise the model emits </s> immediately and stops.
"""
import types
import unittest

import torch

from mammoth.translate.beam_search import BeamSearch
from mammoth.translate.greedy_search import GreedySearch
from mammoth.translate.translator import resolve_decoder_start_with_eos


class GlobalScorerStub:
    alpha = 0
    beta = 0

    def __init__(self):
        self.length_penalty = lambda x, alpha: 1.0
        self.cov_penalty = lambda cov, beta: torch.zeros(
            (1, cov.shape[-2]), device=cov.device, dtype=torch.float
        )
        self.has_cov_pen = False
        self.has_len_pen = False


PAD, BOS, EOS, UNK = 0, 1, 2, 3


def _make_beam(decoder_start_with_eos):
    beam = BeamSearch(
        beam_size=2,
        batch_size=1,
        pad=PAD, bos=BOS, eos=EOS, unk=UNK,
        n_best=1,
        global_scorer=GlobalScorerStub(),
        min_length=0, max_length=10,
        block_ngram_repeat=0, exclusion_tokens=set(),
        stepwise_penalty=False, ratio=0.0, ban_unk_token=False,
        device=torch.device("cpu"),
        decoder_start_with_eos=decoder_start_with_eos,
    )
    beam.initialize(
        encoder_output=torch.randn(1, 5, 8),
        src_mask=torch.zeros(1, 5, dtype=torch.bool),
    )
    return beam


def _make_greedy(decoder_start_with_eos):
    samp = GreedySearch(
        pad=PAD, bos=BOS, eos=EOS, unk=UNK,
        batch_size=1,
        global_scorer=GlobalScorerStub(),
        min_length=0,
        block_ngram_repeat=0,
        exclusion_tokens=set(),
        max_length=10,
        sampling_temp=1.0, keep_topk=1, keep_topp=0,
        beam_size=1, ban_unk_token=False,
        device=torch.device("cpu"),
        decoder_start_with_eos=decoder_start_with_eos,
    )
    samp.initialize(
        encoder_output=torch.randn(1, 5, 8),
        src_mask=torch.zeros(1, 5, dtype=torch.bool),
    )
    return samp


class TestDecodeStrategyStartToken(unittest.TestCase):
    def test_beam_search_default_starts_with_bos(self):
        beam = _make_beam(decoder_start_with_eos=False)
        self.assertTrue((beam.alive_seq == BOS).all())
        self.assertEqual(beam.alive_seq.shape, (1 * 2, 1))

    def test_beam_search_bart_mode_starts_with_eos(self):
        beam = _make_beam(decoder_start_with_eos=True)
        self.assertTrue((beam.alive_seq == EOS).all())
        self.assertEqual(beam.alive_seq.shape, (1 * 2, 1))

    def test_greedy_default_starts_with_bos(self):
        samp = _make_greedy(decoder_start_with_eos=False)
        self.assertTrue((samp.alive_seq == BOS).all())

    def test_greedy_bart_mode_starts_with_eos(self):
        samp = _make_greedy(decoder_start_with_eos=True)
        self.assertTrue((samp.alive_seq == EOS).all())


class TestResolveDecoderStartWithEos(unittest.TestCase):
    """The translator must figure out whether to prime decoding with </s>.
    Priority: explicit opts.decoder_start_with_eos > tgt_vocab attribute > False.
    """

    def test_returns_false_when_neither_set(self):
        opts = types.SimpleNamespace()
        vocab = types.SimpleNamespace()
        self.assertFalse(resolve_decoder_start_with_eos(opts, vocab))

    def test_picks_up_opts_flag_true(self):
        opts = types.SimpleNamespace(decoder_start_with_eos=True)
        vocab = types.SimpleNamespace()
        self.assertTrue(resolve_decoder_start_with_eos(opts, vocab))

    def test_picks_up_vocab_attribute_true(self):
        opts = types.SimpleNamespace()
        vocab = types.SimpleNamespace(decoder_start_with_eos=True)
        self.assertTrue(resolve_decoder_start_with_eos(opts, vocab))

    def test_opts_overrides_vocab_when_explicit_false(self):
        # Explicit opts=False should win over vocab=True (lets users disable per-run).
        opts = types.SimpleNamespace(decoder_start_with_eos=False)
        vocab = types.SimpleNamespace(decoder_start_with_eos=True)
        self.assertFalse(resolve_decoder_start_with_eos(opts, vocab))


if __name__ == "__main__":
    unittest.main()
