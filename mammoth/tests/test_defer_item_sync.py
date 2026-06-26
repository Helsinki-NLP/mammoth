"""Tests for deferring GPU->CPU `.item()` syncs from the training hot loop.

Previously the fast (non-accuracy) path read its running totals back with one
`.tolist()` per `Trainer._gradient_accumulation` call - i.e. one GPU->CPU sync
per training step. We now keep loss, token counts and the NaN flag on the GPU
across steps inside `Statistics` device accumulators, and read them back with a
single `.tolist()` only when the figures are actually needed (at the report
step, via `Statistics.materialize`).

These exercise the REAL `Trainer._gradient_accumulation`, so they catch every
synchronisation point that actually fires in the hot loop.

Goals verified here:
  1. Hot loop is sync-free: `_gradient_accumulation` performs ZERO GPU->CPU
     syncs in the fast path, no matter how many steps accumulate.
  2. Single sync per report: N accumulation windows followed by one
     `materialize()` cost exactly ONE sync, independent of N.
  3. Behaviour parity: after `materialize()` the aggregated statistics (loss,
     n_words, n_src_words, n_sents, per-task loss) match a hand computation.
  4. NaN detection is deferred, not lost: the hot loop no longer raises; the NaN
     surfaces as `report_stats.had_nan` once `materialize()` runs (this is what
     the trainer checks at the report step before raising NanLossException).
"""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from mammoth.trainer import Trainer
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


def _accumulate_window(fake_self, batches, metadata, total_stats, report_stats):
    """Run one `_gradient_accumulation` window (one training step's worth)."""
    my_task = SimpleNamespace(get_serializable_metadata=lambda: metadata)
    # All batches in one accumulation window share one comm_batch id.
    batches_with_meta = [(b, metadata, 0) for b in batches]
    Trainer._gradient_accumulation(
        fake_self, batches_with_meta, total_stats, report_stats, my_task, gradient_syncs=[],
    )


def _run(fake_self, batches, metadata):
    """Run a single accumulation window and return fresh stats objects."""
    total_stats = Statistics()
    report_stats = Statistics()
    _accumulate_window(fake_self, batches, metadata, total_stats, report_stats)
    return total_stats, report_stats


@contextmanager
def _count_syncs():
    """Count GPU->CPU synchronisations (`.item()` / `.tolist()` / `if tensor:`).

    Yields a one-element list whose value is the running count; read it after the
    `with` block.
    """
    counter = [0]
    real_item = torch.Tensor.item
    real_tolist = torch.Tensor.tolist
    real_bool = torch.Tensor.__bool__

    def counting_item(self):
        counter[0] += 1
        return real_item(self)

    def counting_tolist(self):
        counter[0] += 1
        return real_tolist(self)

    def counting_bool(self):
        counter[0] += 1
        return real_bool(self)

    torch.Tensor.item = counting_item
    torch.Tensor.tolist = counting_tolist
    torch.Tensor.__bool__ = counting_bool
    try:
        yield counter
    finally:
        torch.Tensor.item = real_item
        torch.Tensor.tolist = real_tolist
        torch.Tensor.__bool__ = real_bool


class TestHotLoopIsSyncFree:
    """The fast path must never touch the CPU inside `_gradient_accumulation`."""

    def _build(self, n_steps):
        metadata = _FakeMetadata()
        # n_steps windows, 2 batches each -> distinct, finite losses.
        loss_values = [1.0] * (2 * n_steps)
        fake_self = _make_fake_trainer(loss_values, metadata)
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
        ]
        return metadata, fake_self, batches

    @pytest.mark.parametrize("n_steps", [1, 5])
    def test_no_sync_during_accumulation(self, n_steps):
        metadata, fake_self, batches = self._build(n_steps)
        total_stats = Statistics()
        report_stats = Statistics()
        with _count_syncs() as counter:
            for _ in range(n_steps):
                _accumulate_window(fake_self, batches, metadata, total_stats, report_stats)
        assert counter[0] == 0, (
            f"expected zero GPU->CPU syncs in the hot loop, got {counter[0]} "
            f"over {n_steps} accumulation window(s)"
        )


class TestSingleSyncPerReport:
    """N steps + one materialize() == exactly one sync, independent of N."""

    def _total_syncs(self, n_windows):
        metadata = _FakeMetadata()
        loss_values = [1.0] * (2 * n_windows)
        fake_self = _make_fake_trainer(loss_values, metadata)
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
        ]
        total_stats = Statistics()
        report_stats = Statistics()
        with _count_syncs() as counter:
            for _ in range(n_windows):
                _accumulate_window(fake_self, batches, metadata, total_stats, report_stats)
            report_stats.materialize()  # the report step: the one and only sync
        return counter[0]

    def test_sync_count_is_one_regardless_of_window_count(self):
        assert self._total_syncs(2) == 1
        assert self._total_syncs(8) == 1


class TestBehaviourParity:
    def test_stats_zero_until_materialize_then_match_hand_computation(self):
        metadata = _FakeMetadata()
        loss_values = [1.5, 2.5, 0.5]
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
            _make_batch(metadata, batch_size=3, src_len=3, tgt_len=4, src_tokens=4, tgt_tokens=7),
            _make_batch(metadata, batch_size=1, src_len=3, tgt_len=4, src_tokens=2, tgt_tokens=3),
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)
        total_stats, report_stats = _run(fake_self, batches, metadata)

        # Deferred quantities have not synced yet -> still zero.
        assert report_stats.loss == 0
        assert report_stats.n_words == 0
        assert report_stats.n_src_words == 0
        # Sentence counts are plain CPU ints, NOT deferred -> already correct.
        assert report_stats.n_sents == 2 + 3 + 1
        assert report_stats.cumulative_sents == 2 + 3 + 1

        report_stats.materialize()
        total_stats.materialize()

        assert total_stats.loss == pytest.approx(sum(loss_values))
        assert total_stats.n_words == 6 + 7 + 3
        assert report_stats.loss == pytest.approx(sum(loss_values))
        assert report_stats.n_words == 6 + 7 + 3
        assert report_stats.n_src_words == 5 + 4 + 2
        key = f"{metadata.src_lang}_{metadata.tgt_lang}"
        assert report_stats.loss_per_task[key] == pytest.approx(sum(loss_values))

    def test_materialize_is_idempotent(self):
        metadata = _FakeMetadata()
        loss_values = [1.5, 2.5]
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6),
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)
        _, report_stats = _run(fake_self, batches, metadata)

        report_stats.materialize()
        loss_after_first = report_stats.loss
        n_words_after_first = report_stats.n_words
        # A second materialize with nothing pending must not double-count.
        report_stats.materialize()
        assert report_stats.loss == loss_after_first
        assert report_stats.n_words == n_words_after_first


class TestNanDeferred:
    def test_hot_loop_does_not_raise_and_flags_nan_on_materialize(self):
        metadata = _FakeMetadata()
        loss_values = [1.0, float("nan"), 2.0]
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6)
            for _ in range(3)
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)

        # The hot loop must NOT raise anymore (detection is deferred).
        total_stats, report_stats = _run(fake_self, batches, metadata)
        assert report_stats.had_nan is False  # not yet synced

        # The report step reads it back; this is the flag the trainer checks
        # before raising NanLossException.
        report_stats.materialize()
        assert report_stats.had_nan is True

    def test_no_nan_keeps_flag_false(self):
        metadata = _FakeMetadata()
        loss_values = [1.0, 2.0, 3.0]
        batches = [
            _make_batch(metadata, batch_size=2, src_len=3, tgt_len=4, src_tokens=5, tgt_tokens=6)
            for _ in range(3)
        ]
        fake_self = _make_fake_trainer(loss_values, metadata)
        _, report_stats = _run(fake_self, batches, metadata)
        report_stats.materialize()
        assert report_stats.had_nan is False
