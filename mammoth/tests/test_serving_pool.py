"""Mammoth server, step 1: model sources + ComponentPool.

Uses the three-task checkpoint of test_hf_modular (eng-spa, eng-fra, deu-spa sharing
stacks) served two ways — straight from the training checkpoint and from its modular HF
export — and checks that pooled models reproduce the reference model exactly in fp32,
that shared components are loaded once and shared in memory, and that zero-shot
directions are served like trained ones.
"""

import pytest
import torch

from mammoth.serving import CheckpointSource, ComponentPool, HFSource, default_path, open_source
from mammoth.tests.test_hf_modular import (  # noqa: F401  (fixtures)
    TASKS, VOCAB_SIZES, _logits, _reference, checkpoint, modular_dir,
)


class _CountingSource:
    """Wraps a source and counts component reads."""

    def __init__(self, source):
        self.source, self.manifest, self.reads = source, source.manifest, []

    def load_component(self, name):
        self.reads.append(name)
        return self.source.load_component(name)

    def load_tokenizer(self, side, lang):
        return self.source.load_tokenizer(side, lang)


@pytest.fixture(params=["checkpoint", "hf"])
def source(request, checkpoint, modular_dir):
    ckpt, _, _ = checkpoint
    return open_source(str(ckpt) if request.param == "checkpoint" else str(modular_dir))


def test_open_source_detects_format(checkpoint, modular_dir):
    ckpt, _, _ = checkpoint
    assert isinstance(open_source(str(ckpt)), CheckpointSource)
    assert isinstance(open_source(str(modular_dir)), HFSource)


def test_sources_agree(checkpoint, modular_dir):
    """A checkpoint describes itself exactly like its modular export (minus file entries)."""
    ckpt, _, _ = checkpoint
    from_ckpt, from_hf = CheckpointSource(str(ckpt)), HFSource(str(modular_dir))
    for key in ("architecture", "tokenizers", "tasks"):
        assert from_ckpt.manifest[key] == from_hf.manifest[key]
    assert from_ckpt.manifest["components"].keys() == from_hf.manifest["components"].keys()
    for name in from_hf.manifest["components"]:
        a, b = from_ckpt.load_component(name), from_hf.load_component(name)
        assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    for side, lang in (("src", "eng"), ("tgt", "fra")):
        text = "eng5 eng6" if side == "src" else "fra5 fra6"
        assert (from_ckpt.load_tokenizer(side, lang)(text)["input_ids"]
                == from_hf.load_tokenizer(side, lang)(text)["input_ids"])


def test_available_tasks(source):
    pool = ComponentPool(source)
    tasks = pool.available_tasks()
    assert set(tasks) == {"eng-spa", "eng-fra", "deu-spa", "deu-fra"}
    assert tasks["deu-fra"] == dict(src="deu", tgt="fra", encoder=["deu", "all"],
                                    decoder=["fra", "all"], trained=False)
    assert all(tasks[t]["trained"] for t in TASKS)


@pytest.mark.parametrize("task", sorted(TASKS) + ["deu-fra"])
def test_exact_match(checkpoint, source, task):
    """Pooled models — trained and zero-shot — equal the model plugged from original modules."""
    _, opts, modules = checkpoint
    pool = ComponentPool(source)
    for t in sorted(TASKS) + ["deu-fra"]:  # load everything, so the task is built from a busy pool
        pool.add_task(t)
    loaded = pool.tasks[task]
    src, tgt = task.split("-")
    ref = _reference(opts, modules, src, tgt, loaded.spec["encoder"], loaded.spec["decoder"])
    assert torch.equal(_logits(loaded.model, src, tgt), _logits(ref, src, tgt))
    assert loaded.trained == (task in TASKS)
    assert len(loaded.src_tokenizer) == VOCAB_SIZES[("src", src)]
    assert len(loaded.tgt_tokenizer) == VOCAB_SIZES[("tgt", tgt)]


def test_components_loaded_once_and_shared(source):
    counting = _CountingSource(source)
    pool = ComponentPool(counting)
    a, b = pool.add_task("eng-spa"), pool.add_task("eng-fra")
    pool.add_task("deu-spa")
    assert pool.add_task("eng-spa") is a  # already loaded: no-op
    assert len(counting.reads) == len(set(counting.reads)) == 14  # 24 without sharing

    assert a.model.encoder.stacks[0] is b.model.encoder.stacks[0]  # enc__0__eng
    assert a.model.encoder.token_emb is b.model.encoder.token_emb  # src_emb__eng
    assert a.model.decoder.stacks[1] is pool.tasks["deu-spa"].model.decoder.stacks[1]  # dec__1__all
    assert a.model.decoder.stacks[0] is not b.model.decoder.stacks[0]  # spa vs fra
    assert pool.loaded_components()["enc__1__all"] == ["deu-spa", "eng-fra", "eng-spa"]

    # Zero-shot deu-fra needs nothing new: every component is already in the pool.
    pool.add_task("deu-fra")
    assert len(counting.reads) == 14


def test_remove_frees_unused_components(source):
    pool = ComponentPool(source)
    for t in ("eng-spa", "eng-fra", "deu-spa"):
        pool.add_task(t)
    pool.remove_task("eng-spa")
    comps = pool.loaded_components()
    assert comps["enc__0__eng"] == ["eng-fra"]  # still used
    assert comps["dec__0__spa"] == ["deu-spa"]
    pool.remove_task("eng-fra")
    comps = pool.loaded_components()
    assert not any("eng" in name or "fra" in name for name in comps)
    assert set(pool._tokenizers) == {("src", "deu"), ("tgt", "spa")}
    with pytest.raises(KeyError):
        pool.remove_task("eng-fra")


def test_unservable_directions(source):
    pool = ComponentPool(source)
    with pytest.raises(ValueError, match="never a source language"):
        pool.add_task("spa-eng")
    with pytest.raises(ValueError, match="src-tgt"):
        pool.add_task("engspa")
    assert pool.tasks == {} and pool.loaded_components() == {}


def test_default_path_prefers_most_used():
    manifest = {"tasks": {
        "a": dict(src="eng", tgt="x", encoder=["eng", "b"], decoder=[]),
        "b": dict(src="eng", tgt="y", encoder=["eng", "a"], decoder=[]),
        "c": dict(src="eng", tgt="z", encoder=["eng", "b"], decoder=[]),
        "d": dict(src="deu", tgt="x", encoder=["deu", "b"], decoder=[]),
        "e": dict(src="deu", tgt="y", encoder=["deu", "a"], decoder=[]),
    }}
    assert default_path(manifest, "src", "eng") == ["eng", "b"]  # used by 2 tasks
    assert default_path(manifest, "src", "deu") == ["deu", "a"]  # tie → alphabetical


def test_generate(source):
    loaded = ComponentPool(source).add_task("deu-fra")
    inputs = loaded.src_tokenizer(["deu5 deu6 deu7"], return_tensors="pt")
    inputs.pop("token_type_ids", None)
    out = loaded.model.generate(**inputs, max_new_tokens=5, num_beams=2)
    assert out.shape[0] == 1 and out.max().item() < VOCAB_SIZES[("tgt", "fra")]
