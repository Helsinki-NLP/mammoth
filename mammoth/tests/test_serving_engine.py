"""Mammoth server, step 2: TranslationEngine (text in, text out) on top of ComponentPool.

Uses the random-weight three-task checkpoint of test_hf_modular, so translations are
nonsense; the tests check the plumbing: tokenizers per side, batching that does not change
results, input order, length limits, and settings handling.
"""

import math
import threading

import pytest
import torch

from mammoth.serving import ComponentPool, GenerationSettings, TranslationEngine, make_batches, open_source
from mammoth.tests.test_hf_modular import VOCAB_SIZES, checkpoint  # noqa: F401  (fixture)

# Different lengths, so the engine has to sort, batch and pad.
SEGMENTS = ["deu5 deu6 deu7", "deu8", "deu9 deu10 deu11 deu12 deu13 deu14", "deu15 deu16",
            "deu17 deu18 deu19 deu20", "deu21"]


@pytest.fixture(scope="module")
def pool(checkpoint):
    ckpt, _, _ = checkpoint
    pool = ComponentPool(open_source(str(ckpt)))
    pool.add_task("deu-fra")  # zero-shot
    pool.add_task("eng-spa")
    return pool


def _reference(loaded, segment, num_beams, max_new_tokens):
    """One segment through model.generate directly, without the engine."""
    inputs = loaded.src_tokenizer([segment], return_tensors="pt")
    out = loaded.model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                                num_beams=num_beams, max_new_tokens=max_new_tokens,
                                early_stopping=num_beams > 1)
    return loaded.tgt_tokenizer.decode(out[0], skip_special_tokens=True).strip()


@pytest.mark.parametrize("num_beams", [1, 3])
def test_matches_direct_generate(pool, num_beams):
    engine = TranslationEngine(pool, GenerationSettings(num_beams=num_beams, max_new_tokens=8))
    loaded = pool.tasks["deu-fra"]
    hyps = engine.translate("deu-fra", SEGMENTS)
    assert hyps == [_reference(loaded, s, num_beams, 8) for s in SEGMENTS]
    # Output is in the target vocabulary, decoded with the target tokenizer.
    assert all(w.startswith("fra") for h in hyps for w in h.split())


@pytest.mark.parametrize("num_beams", [1, 3])
def test_batching_does_not_change_results(pool, num_beams):
    engine = TranslationEngine(pool, GenerationSettings(num_beams=num_beams, max_new_tokens=8))
    one_by_one = engine.translate("deu-fra", SEGMENTS, max_batch_size=1)
    assert engine.translate("deu-fra", SEGMENTS) == one_by_one
    assert engine.translate("deu-fra", SEGMENTS, max_batch_tokens=10) == one_by_one
    assert engine.translate("deu-fra", SEGMENTS[::-1]) == one_by_one[::-1]  # input order kept


def test_str_and_list_inputs(pool):
    engine = TranslationEngine(pool, GenerationSettings(max_new_tokens=6))
    hyp = engine.translate("eng-spa", "eng5 eng6")
    assert isinstance(hyp, str) and hyp.split()[0].startswith("spa")
    assert engine.translate("eng-spa", ["eng5 eng6"]) == [hyp]
    assert engine.translate("eng-spa", ["", "eng5 eng6", "  "]) == ["", hyp, ""]
    assert engine.translate("eng-spa", []) == []
    assert engine.translate("eng-spa", "") == ""


def test_length_limit(pool):
    """A segment may not grow beyond max_length_factor × its length + slack, nor max_new_tokens."""
    engine = TranslationEngine(pool, GenerationSettings(num_beams=1, max_new_tokens=100))
    for factor, slack, cap in ((0.5, 1, 100), (2.0, 10, 3)):
        # One segment per batch: the limit is the segment's own (a batch uses its longest).
        hyps = engine.translate("deu-fra", SEGMENTS, max_length_factor=factor, max_length_slack=slack,
                                max_new_tokens=cap, max_batch_size=1)
        for seg, hyp in zip(SEGMENTS, hyps):
            src_len = len(seg.split()) + 2  # + BOS, EOS
            assert len(hyp.split()) <= min(cap, math.ceil(factor * src_len) + slack)


def test_errors(pool):
    engine = TranslationEngine(pool)
    with pytest.raises(KeyError, match="not loaded"):
        engine.translate("deu-spa", "deu5")
    with pytest.raises(ValueError, match="num_beams"):
        engine.translate("deu-fra", "deu5", num_beams=0)
    with pytest.raises(ValueError, match="max_length_factor"):
        GenerationSettings(max_length_factor=0).validate()


def test_concurrent_calls(pool):
    """Calls from several threads (as a web server makes them) give the sequential results."""
    engine = TranslationEngine(pool, GenerationSettings(num_beams=2, max_new_tokens=6))
    expected = {t: engine.translate(t, SEGMENTS if t == "deu-fra" else ["eng5 eng6", "eng7"])
                for t in ("deu-fra", "eng-spa")}
    results, errors = [], []

    def worker(task):
        try:
            inputs = SEGMENTS if task == "deu-fra" else ["eng5 eng6", "eng7"]
            results.append((task, engine.translate(task, inputs)))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in ["deu-fra", "eng-spa"] * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(hyps == expected[task] for task, hyps in results) and len(results) == 8


def test_unload_while_translating(checkpoint):
    """A call that already looked up its task finishes even if the task is unloaded meanwhile."""
    ckpt, _, _ = checkpoint
    pool = ComponentPool(open_source(str(ckpt)))
    pool.add_task("eng-spa")
    engine = TranslationEngine(pool, GenerationSettings(num_beams=1, max_new_tokens=4))
    loaded = pool.tasks["eng-spa"]
    pool.remove_task("eng-spa")
    assert pool.loaded_components() == {}
    # The task's model still holds its modules, so generation works.
    assert engine._translate(loaded, ["eng5 eng6"], engine.settings)[0].startswith("spa")


def test_make_batches():
    lengths = [3, 10, 1, 7, 7, 2]
    batches = make_batches(lengths, max_batch_size=2, max_batch_tokens=100)
    assert batches == [[1, 3], [4, 0], [5, 2]]
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    # Token budget: (segments × longest) <= 14, except a lone segment longer than the budget.
    batches = make_batches(lengths, max_batch_size=32, max_batch_tokens=14)
    assert batches == [[1], [3, 4], [0, 5, 2]]
    assert make_batches([20], 32, 14) == [[0]]
    assert make_batches([], 32, 14) == []
