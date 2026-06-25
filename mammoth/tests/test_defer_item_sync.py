"""Tests for deferring GPU->CPU `.item()` syncs in Trainer._gradient_accumulation.

These exercise the REAL `Trainer._gradient_accumulation` method (not a hand-written
simulation), so they catch every synchronisation point that actually fires in the
hot loop - including the per-batch NaN check.

Goals verified here:
  1. Behaviour parity: the aggregated statistics (loss, n_words, n_src_words,
     n_sents, per-task loss) are numerically identical to the old per-batch logic.
  2. No per-batch sync: the number of GPU->CPU synchronisations (`.item()` /
     `.tolist()`) is CONSTANT, independent of the gradient-accumulation count.
  3. NaN detection still raises NanLossException.
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from mammoth.trainer import Trainer, NanLossException
from mammoth.utils.statistics import Statistics


class _FakeMetadata:
    """Stand-in for task metadata; compares equal to itself by identity."""

    def __init__(self, corpus_id="corpus0", src_lang="src", tgt_lang="tgt"):
        self.corpus_id = corpus_id
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang


def _make_batch(metadata, batch_size, src_len, tgt_len, src_tokens, tgt_tokens):
    """Build a fake batch with controllable real-token counts.

    `src_tokens` / `tgt_tokens` set how many mask entries are True (non-padding),
    so we know exactly what `mask.sum()` should produce.
    Tensor shapes follow the trainer's expectation: (time, batch, 1).
    """
    src_tensor = torch.zeros(src_len, batch_size, 1, dtype=torch.long)
    tgt_tensor = torch.zeros(tgt_len, batch_size, 1, dtype=torch.long)

    src_mask = torch.zeros(src_len, batch_size, dtype=torch.bool)
    tgt_mask = torch.zeros(tgt_len, batch_size, dtype=torch.bool)
    # Flip exactly `src_tokens` / `tgt_tokens` entries to True.
    src_mask.view(-1)[:src_tokens] = True
    tgt_mask.view(-1)[:tgt_tokens] = True

    src = SimpleNamespace(tensor=src_tensor, mask=src_mask)
    tgt = SimpleNamespace(tensor=tgt_tensor, mask=tgt_mask)
    return SimpleNamespace(
        src=src, tgt=tgt, batch_size=batch_size, line_idx=0,
    )


def _make_fake_trainer(loss_values, metadata, report_training_accuracy=False):
    """Construct a minimal object carrying just the attributes the method reads.

    `loss_values` is consumed one per batch by the fake loss function so each
    batch has a known, distinct loss.
    """
    loss_iter = iter(loss_values)

    def fake_model(src, decoder_input, src_mask, metadata):
        b, t = decoder_input.shape
        logits = torch.zeros(b, t, 4)            # (b, t, vocab)
        decoder_output = torch.zeros(b, t, 2)    # (b, t, dim)
        return logits, decoder_output

    def fake_loss_fn(logits_flat, target_flat):
        return torch.tensor(next(loss_iter), dtype=torch.float32)

    @contextmanager
    def fake_profiler_range(_name):
        yield

    return SimpleNamespace(
        profiler_range=fake_profiler_range,
        device_context=SimpleNamespace(is_gpu=lambda: False),
        model_dtype="fp32",
        norm_method="tokens",
        _data_state={},
        model=fake_model,
        loss_functions={metadata.tgt_lang: fake_loss_fn},
        optim=SimpleNamespace(amp=False, backward=lambda loss: None),
        report_training_accuracy=report_training_accuracy,
        report_tflops=False,
        flops_config={},
    )


def _run(fake_self, batches, metadata):
    total_stats = Statistics()
    report_stats = Statistics()
    my_task = SimpleNamespace(get_serializable_metadata=lambda: metadata)
    batches_with_meta = [(b, metadata, i) for i, b in enumerate(batches)]
    # All batches in one accumulation window share one comm_batch id.
    batches_with_meta = [(b, metadata, 0) for b in batches]
    Trainer._gradient_accumulation(
        fake_self, batches_with_meta, total_stats, report_stats, my_task, gradient_syncs=[],
    )
    return total_stats, report_stats


class TestBehaviourParity:
    def test_aggregated_statistics_match_hand_computation(self):
        metadata = _FakeMetadata()
        loss_values = [1.5, 2.5, 0.5]
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
            _make_batch(metadata, batch_size=3, src_len=3, tgt_len=4, src_tokens=4, tgt_tokens=7),
            _make_batch(metadata, batch_size=1, src_len=3, tgt_len=4, src_tokens=2, tgt_tokens=3),
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)
        total_stats, report_stats = _run(fake_self, batches, metadata)

        assert total_stats.loss == pytest.approx(sum(loss_values))
        assert total_stats.n_words == 6 + 7 + 3
        assert report_stats.n_src_words == 5 + 4 + 2
        assert report_stats.n_sents == 2 + 3 + 1
        key = f"{metadata.src_lang}_{metadata.tgt_lang}"
        assert report_stats.loss_per_task[key] == pytest.approx(sum(loss_values))


class TestSyncCountIsConstant:
    def _count_syncs(self, n_batches):
        metadata = _FakeMetadata()
        loss_values = [1.0] * n_batches
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6)
            for _ in range(n_batches)
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)

        counter = {"n": 0}
        real_item = torch.Tensor.item
        real_tolist = torch.Tensor.tolist
        # `if some_tensor:` (e.g. `if torch.isnan(loss):`) also forces a
        # GPU->CPU sync via __bool__, so count it too.
        real_bool = torch.Tensor.__bool__

        def counting_item(self):
            counter["n"] += 1
            return real_item(self)

        def counting_tolist(self):
            counter["n"] += 1
            return real_tolist(self)

        def counting_bool(self):
            counter["n"] += 1
            return real_bool(self)

        torch.Tensor.item = counting_item
        torch.Tensor.tolist = counting_tolist
        torch.Tensor.__bool__ = counting_bool
        try:
            _run(fake_self, batches, metadata)
        finally:
            torch.Tensor.item = real_item
            torch.Tensor.tolist = real_tolist
            torch.Tensor.__bool__ = real_bool
        return counter["n"]

    def test_sync_count_independent_of_accumulation_count(self):
        syncs_2 = self._count_syncs(2)
        syncs_8 = self._count_syncs(8)
        assert syncs_2 == syncs_8, (
            f"GPU->CPU syncs scale with accumulation count: "
            f"{syncs_2} (n=2) vs {syncs_8} (n=8); a per-batch sync remains."
        )


class TestNanStillDetected:
    def test_nan_loss_raises(self):
        metadata = _FakeMetadata()
        loss_values = [1.0, float("nan"), 2.0]
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6)
            for _ in range(3)
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)
        with pytest.raises(NanLossException):
            _run(fake_self, batches, metadata)