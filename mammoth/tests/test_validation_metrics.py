"""Tests for additional validation metrics (BLEU, chrF) in Trainer.

These tests exercise Trainer._compute_validation_metrics in isolation.
The method does not read any instance attributes, so we invoke it as an
unbound method with `None` for self.

Predictions/references are grouped by corpus_id (a plain string), not by
(src_lang, tgt_lang) direction: two different tasks can declare the same
`src_tgt` (e.g. a weight=0 eval-only task copy-pasted from a training task)
and must still get independent scores instead of being silently merged.
"""

import pytest
from unittest.mock import patch

from mammoth.trainer import Trainer
import mammoth.trainer as trainer_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def preds_by_direction():
    # Two tasks, with one near-perfect match and one poor match,
    # so both BLEU and chrF should return non-trivial, distinct scores.
    return {
        'en-fi': ['the cat sat on the mat', 'hello world'],
        'en-de': ['completely wrong output here'],
    }


@pytest.fixture
def refs_by_direction():
    return {
        'en-fi': ['the cat sat on the mat', 'hello world'],
        'en-de': ['a totally different reference sentence'],
    }


def _call(valid_metrics, preds, refs):
    """Invoke the method unbound — it doesn't touch self."""
    return Trainer._compute_validation_metrics(None, preds, refs, valid_metrics)


# ---------------------------------------------------------------------------
# chrF happy path
# ---------------------------------------------------------------------------

def test_chrf_returns_per_direction_keys(preds_by_direction, refs_by_direction):
    """chrF should produce a `chrf/{src}-{tgt}` key per direction."""
    metrics = _call(['chrf'], preds_by_direction, refs_by_direction)

    assert 'chrf/en-fi' in metrics
    assert 'chrf/en-de' in metrics
    # Perfect match direction should score much higher than the mismatched one.
    assert metrics['chrf/en-fi'] > metrics['chrf/en-de']
    # sacrebleu chrF scores are in [0, 100].
    for value in metrics.values():
        assert 0.0 <= value <= 100.0


def test_chrf_and_bleu_coexist(preds_by_direction, refs_by_direction):
    """Requesting both metrics should yield both families of keys."""
    metrics = _call(['bleu', 'chrf'], preds_by_direction, refs_by_direction)

    bleu_keys = [k for k in metrics if k.startswith('bleu/')]
    chrf_keys = [k for k in metrics if k.startswith('chrf/')]
    assert len(bleu_keys) == 2
    assert len(chrf_keys) == 2


# ---------------------------------------------------------------------------
# Error / edge cases
# ---------------------------------------------------------------------------

def test_chrf_skips_direction_with_empty_preds(refs_by_direction):
    """Empty predictions for one direction should be skipped, others still scored."""
    preds = {
        'en-fi': ['the cat sat on the mat'],
        'en-de': [],  # empty — should be skipped
    }
    refs = {
        'en-fi': ['the cat sat on the mat'],
        'en-de': ['some reference'],
    }
    metrics = _call(['chrf'], preds, refs)

    assert 'chrf/en-fi' in metrics
    assert 'chrf/en-de' not in metrics


def test_sacrebleu_unavailable_skips_chrf(preds_by_direction, refs_by_direction):
    """When sacrebleu is not importable, chrF should log a warning and return {}."""
    with patch.object(trainer_mod, 'SACREBLEU_AVAILABLE', False):
        metrics = _call(['chrf'], preds_by_direction, refs_by_direction)
    assert metrics == {}


def test_unknown_metric_name_is_ignored(preds_by_direction, refs_by_direction):
    """An unknown metric name should neither crash nor pollute results."""
    metrics = _call(['not_a_real_metric'], preds_by_direction, refs_by_direction)
    # No bleu/chrf keys should appear — only whatever the unknown branch emits (nothing).
    assert not any(k.startswith('chrf/') for k in metrics)
    assert not any(k.startswith('bleu/') for k in metrics)


# ---------------------------------------------------------------------------
# ModelSaver: metric_for_best_model='chrf' auto-inference
# ---------------------------------------------------------------------------

def test_greater_is_better_autoinfer_for_chrf():
    """metric_for_best_model='chrf' with greater_is_better=None should infer True.

    This test pokes the auto-infer block in model_saver.py directly to avoid
    constructing a full ModelSaver (which needs a real model, vocabs, etc.).
    It mirrors the logic; once implemented, the real assertion is that 'chrf'
    is added to the `greater_is_better = True` branch at model_saver.py:431.
    """
    from mammoth.utils import model_saver as ms
    import inspect
    src = inspect.getsource(ms)
    # After implementation, the `['accuracy', 'bleu']` list in the
    # greater_is_better auto-infer branch should include 'chrf'.
    assert "'chrf'" in src or '"chrf"' in src, (
        "Expected 'chrf' to be added to the greater_is_better auto-infer branch "
        "in model_saver.py (around line 431)."
    )


# ---------------------------------------------------------------------------
# Sanity: BLEU still works (regression guard while editing the method)
# ---------------------------------------------------------------------------

def test_bleu_still_works(preds_by_direction, refs_by_direction):
    metrics = _call(['bleu'], preds_by_direction, refs_by_direction)
    assert 'bleu/en-fi' in metrics
    assert 'bleu/en-de' in metrics
    assert metrics['bleu/en-fi'] > metrics['bleu/en-de']


# ---------------------------------------------------------------------------
# Two tasks sharing a declared direction must NOT be merged into one score.
#
# Regression test for a bug where a weight=0 eval-only task ("eng-fra_eval")
# copy-pasted its `src_tgt: eng-spa` from the training task it was cloned
# from. Because scores used to be grouped by (src_lang, tgt_lang) instead of
# by corpus_id, both tasks' predictions/references landed in the same
# bucket and produced a single corrupted BLEU/chrF number. Grouping by
# corpus_id keeps every validated task's score independent even when two
# tasks happen to declare the same direction.
# ---------------------------------------------------------------------------

def test_same_direction_tasks_get_independent_scores():
    """Two corpus_ids sharing a direction must each keep their own score."""
    preds = {
        'eng-spa': ['el gato se sento en la alfombra'],
        'eng-fra_eval': ['completely unrelated garbage'],
    }
    refs = {
        'eng-spa': ['el gato se sento en la alfombra'],
        'eng-fra_eval': ['le chat etait assis sur le tapis'],
    }
    metrics = _call(['bleu', 'chrf'], preds, refs)

    assert 'bleu/eng-spa' in metrics
    assert 'bleu/eng-fra_eval' in metrics
    assert 'chrf/eng-spa' in metrics
    assert 'chrf/eng-fra_eval' in metrics
    # The near-perfect match must score much higher than the mismatched one —
    # this would be impossible if the two tasks' texts had been merged
    # into a single sacrebleu corpus call.
    assert metrics['bleu/eng-spa'] > metrics['bleu/eng-fra_eval']
    assert metrics['chrf/eng-spa'] > metrics['chrf/eng-fra_eval']
