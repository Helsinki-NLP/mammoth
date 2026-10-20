"""Modular HF export (convert_mammoth_to_hf --modular) + MammothHub component loading.

Builds a tiny sharded Mammoth checkpoint with three tasks that share layer stacks:

    eng-spa  encoder [eng, all]  decoder [spa, all]
    eng-fra  encoder [eng, all]  decoder [fra, all]
    deu-spa  encoder [deu, all]  decoder [spa, all]

and checks that models assembled from the exported components reproduce, exactly
in fp32, the reference model built by plugging the original component modules
together — for trained tasks and for the zero-shot deu-fra combination.
"""

import argparse
import json
import os

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
    assemble_state_dict, config_from_opts, convert, convert_modular,
)
from mammoth.hf_integration.to_hf.mammoth_hub import MammothHub
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration

PREFIX = "model_best"
TASKS = {
    "eng-spa": (["eng", "all"], ["spa", "all"]),
    "eng-fra": (["eng", "all"], ["fra", "all"]),
    "deu-spa": (["deu", "all"], ["spa", "all"]),
}
# Distinct vocab sizes so a language mix-up shows up as a shape error.
VOCAB_SIZES = {("src", "eng"): 40, ("src", "deu"): 45, ("tgt", "spa"): 50, ("tgt", "fra"): 55}
SPECIALS = ["</s>", "<pad>", "<s>", "<unk>"]  # ids 0..3: EOS=0, PAD=1, BOS=2, UNK=3


class _Vocab:
    """Minimal stand-in for mammoth.inputters.vocab.HFTokenizerVocab (picklable)."""

    def __init__(self, size, tag):
        words = SPECIALS + [f"{tag}{i}" for i in range(size - len(SPECIALS))]
        tok = Tokenizer(WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
        tok.pre_tokenizer = Whitespace()
        self.tokenizer = tok
        self.specials = {w: i for i, w in enumerate(SPECIALS)}

    def __len__(self):
        return self.tokenizer.get_vocab_size()


def _opts():
    return argparse.Namespace(
        model_dim=32, heads=4, ff_mult=2.0, ff_swiglu=True, enc_layers=[1, 1], dec_layers=[2, 1],
        attn_dropout=0.0, ff_dropout=0.0, dropout=[0.0], rotary_pos_emb=True, post_emb_norm=True,
        tasks={
            f"train_{name}": {"src_tgt": name, "enc_sharing_group": enc, "dec_sharing_group": dec}
            for name, (enc, dec) in TASKS.items()
        },
    )


def _randomize(module):
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0, 0.2)  # RMSNorm weights start at 1; randomize so swaps are detectable


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """Sharded checkpoint dir + the original component modules (the ground truth)."""
    torch.manual_seed(0)
    ckpt = tmp_path_factory.mktemp("ckpt")
    opts = _opts()
    vocab = {key: _Vocab(size, key[1]) for key, size in VOCAB_SIZES.items()}
    torch.save({"opts": opts, "vocab": vocab}, ckpt / f"{PREFIX}_frame.pt")

    modules = {}  # shard name → module

    def save(name, module):
        if name not in modules:
            _randomize(module)
            modules[name] = module
            torch.save(module.state_dict(), ckpt / f"{PREFIX}_{name}.pt")

    for name, (enc, dec) in TASKS.items():
        src, tgt = name.split("-")
        cfg = config_from_opts(opts, VOCAB_SIZES[("src", src)], VOCAB_SIZES[("tgt", tgt)])
        m = MammothForConditionalGeneration(cfg)
        save(f"src_embeddings_{src}", m.encoder.token_emb)
        save(f"tgt_embeddings_{tgt}", m.decoder.token_emb)
        for i, x in enumerate(enc):
            save(f"encoder_{i}_{x}", m.encoder.stacks[i])
        for i, x in enumerate(dec):
            save(f"decoder_{i}_{x}", m.decoder.stacks[i])
        save(f"encoder_wrapper_{'_'.join(enc)}", torch.nn.ModuleDict(
            {"post_emb_norm": m.encoder.post_emb_norm}))
        save(f"decoder_wrapper_{'_'.join(dec)}", torch.nn.ModuleDict(
            {"post_emb_norm": m.decoder.post_emb_norm, "to_logits": m.decoder.to_logits}))
    # Optimizer shards sit next to the weights in real checkpoints; the exporter must skip them.
    torch.save({"state": {}}, ckpt / f"{PREFIX}_encoder_1_all_optim.pt")
    return ckpt, opts, modules


@pytest.fixture(scope="module")
def modular_dir(checkpoint, tmp_path_factory):
    ckpt, _, _ = checkpoint
    out = tmp_path_factory.mktemp("modular")
    convert_modular(str(ckpt), str(out))
    return out


def _reference(opts, modules, src, tgt, enc, dec):
    """Build the model by plugging the original component modules together."""
    cfg = config_from_opts(opts, VOCAB_SIZES[("src", src)], VOCAB_SIZES[("tgt", tgt)])
    m = MammothForConditionalGeneration(cfg)
    m.encoder.token_emb.load_state_dict(modules[f"src_embeddings_{src}"].state_dict())
    m.decoder.token_emb.load_state_dict(modules[f"tgt_embeddings_{tgt}"].state_dict())
    for i, x in enumerate(enc):
        m.encoder.stacks[i].load_state_dict(modules[f"encoder_{i}_{x}"].state_dict())
    for i, x in enumerate(dec):
        m.decoder.stacks[i].load_state_dict(modules[f"decoder_{i}_{x}"].state_dict())
    enc_w = modules[f"encoder_wrapper_{'_'.join(enc)}"]
    dec_w = modules[f"decoder_wrapper_{'_'.join(dec)}"]
    m.encoder.post_emb_norm.load_state_dict(enc_w["post_emb_norm"].state_dict())
    m.decoder.post_emb_norm.load_state_dict(dec_w["post_emb_norm"].state_dict())
    m.decoder.to_logits.load_state_dict(dec_w["to_logits"].state_dict())
    return m.eval()


@torch.no_grad()
def _logits(model, src, tgt):
    g = torch.Generator().manual_seed(1)
    src_ids = torch.randint(4, VOCAB_SIZES[("src", src)], (2, 7), generator=g)
    dec_ids = torch.randint(4, VOCAB_SIZES[("tgt", tgt)], (2, 5), generator=g)
    mask = torch.ones_like(src_ids)
    mask[1, -2:] = 0
    return model(input_ids=src_ids, attention_mask=mask, decoder_input_ids=dec_ids,
                 use_cache=False).logits


def test_components_stored_once(modular_dir):
    manifest = json.loads((modular_dir / "config.json").read_text())
    assert manifest["model_type"] == "mammoth_modular"
    assert set(manifest["components"]) == {
        "src_emb__eng", "src_emb__deu", "tgt_emb__spa", "tgt_emb__fra",
        "enc__0__eng", "enc__0__deu", "enc__1__all",
        "dec__0__spa", "dec__0__fra", "dec__1__all",
        "enc_wrapper__eng__all", "enc_wrapper__deu__all",
        "dec_wrapper__spa__all", "dec_wrapper__fra__all",
    }
    files = sorted(os.listdir(modular_dir / "components"))
    assert len(files) == 14 and not any("optim" in f for f in files)
    assert manifest["tasks"]["eng-fra"] == {
        "src": "eng", "tgt": "fra", "encoder": ["eng", "all"], "decoder": ["fra", "all"]}
    assert set(manifest["tokenizers"]["src"]) == {"eng", "deu"}
    assert set(manifest["tokenizers"]["tgt"]) == {"spa", "fra"}
    for f in ("mammoth_hub.py", "modeling_mammoth.py", "configuration_mammoth.py",
              "native_transformer.py", "inference.py", "model_downloader.py"):
        assert (modular_dir / f).exists()


@pytest.mark.parametrize("task", sorted(TASKS))
def test_trained_task_exact_match(checkpoint, modular_dir, task):
    _, opts, modules = checkpoint
    src, tgt = task.split("-")
    enc, dec = TASKS[task]
    model = MammothHub.from_pretrained(str(modular_dir), task=task)
    ref = _reference(opts, modules, src, tgt, enc, dec)
    assert torch.equal(_logits(model, src, tgt), _logits(ref, src, tgt))
    assert model.mammoth_spec == {"src": src, "tgt": tgt, "encoder": enc, "decoder": dec}
    assert len(model.src_tokenizer) == VOCAB_SIZES[("src", src)]
    assert len(model.tgt_tokenizer) == VOCAB_SIZES[("tgt", tgt)]


def test_matches_per_task_conversion(checkpoint, modular_dir, tmp_path):
    """Modular loading reproduces the existing one-model-per-task conversion."""
    ckpt, _, _ = checkpoint
    convert(str(ckpt), PREFIX, "eng", "spa", str(tmp_path))
    from transformers import AutoModelForSeq2SeqLM
    per_task = AutoModelForSeq2SeqLM.from_pretrained(str(tmp_path), trust_remote_code=True).eval()
    modular = MammothHub.from_pretrained(str(modular_dir), task="eng-spa")
    assert torch.equal(_logits(modular, "eng", "spa"), _logits(per_task, "eng", "spa"))


def test_zero_shot_combination(checkpoint, modular_dir):
    """deu-fra was never trained, but both of its paths were: paths are inferred from src/tgt."""
    _, opts, modules = checkpoint
    model = MammothHub.from_pretrained(str(modular_dir), src="deu", tgt="fra")
    assert model.mammoth_spec["encoder"] == ["deu", "all"]
    assert model.mammoth_spec["decoder"] == ["fra", "all"]
    ref = _reference(opts, modules, "deu", "fra", ["deu", "all"], ["fra", "all"])
    assert torch.equal(_logits(model, "deu", "fra"), _logits(ref, "deu", "fra"))


def test_untrained_path_is_rejected(modular_dir):
    """A path no task trained has no wrapper modules, so it cannot be assembled."""
    with pytest.raises(ValueError, match="dec_wrapper__fra__spa"):
        MammothHub.from_pretrained(str(modular_dir), src="eng", tgt="fra",
                                   encoder=["eng", "all"], decoder=["fra", "spa"])


def test_bad_path_length_and_unknown_task(modular_dir):
    with pytest.raises(ValueError, match="one per encoder layer stack"):
        MammothHub.from_pretrained(str(modular_dir), src="eng", tgt="spa", encoder=["eng"])
    with pytest.raises(ValueError, match="Available tasks"):
        MammothHub.from_pretrained(str(modular_dir), task="fin-spa")


def test_list_helpers(modular_dir):
    assert sorted(MammothHub.list_tasks(str(modular_dir))) == sorted(TASKS)
    assert MammothHub.list_components(str(modular_dir))["enc__1__all"]["xcoder_id"] == "all"


def test_generate_with_tokenizers(modular_dir):
    model = MammothHub.from_pretrained(str(modular_dir), task="eng-spa")
    inputs = model.src_tokenizer(["eng5 eng6 eng7"], return_tensors="pt")
    inputs.pop("token_type_ids", None)
    assert inputs["input_ids"][0, 0].item() == 2 and inputs["input_ids"][0, -1].item() == 0
    out = model.generate(**inputs, max_new_tokens=5, num_beams=2)
    assert out.shape[0] == 1 and out.max().item() < VOCAB_SIZES[("tgt", "spa")]


def test_shipped_inference_load_model(checkpoint, modular_dir, tmp_path):
    """The inference.py copied into an artifact is the single user entry point for every repo type."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("shipped_inference", modular_dir / "inference.py")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

    _, _, model = inference.load_model(str(modular_dir), task="eng-spa")
    ref = MammothHub.from_pretrained(str(modular_dir), task="eng-spa")
    assert torch.equal(_logits(model, "eng", "spa"), _logits(ref, "eng", "spa"))

    _, _, model = inference.load_model(str(modular_dir), src="deu", tgt="fra")
    assert model.mammoth_spec["decoder"] == ["fra", "all"]

    with pytest.raises(ValueError, match="multi-task repo"):
        inference.load_model(str(modular_dir))

    ckpt, _, _ = checkpoint
    convert(str(ckpt), PREFIX, "eng", "spa", str(tmp_path))
    src_tok, tgt_tok, model = inference.load_model(str(tmp_path))
    hyps = inference.translate(src_tok, tgt_tok, model, ["eng5 eng6 eng7"], num_beams=2, max_new_tokens=5)
    assert len(hyps) == 1 and isinstance(hyps[0], str)


def test_assemble_state_dict_unchanged(checkpoint):
    """The refactored per-task assembly still fills a per-task model exactly."""
    ckpt, opts, _ = checkpoint
    sd = assemble_state_dict(str(ckpt), PREFIX, "deu", "spa", ["deu", "all"], ["spa", "all"])
    model = MammothForConditionalGeneration(
        config_from_opts(opts, VOCAB_SIZES[("src", "deu")], VOCAB_SIZES[("tgt", "spa")]))
    assert set(sd) == set(model.state_dict())


def test_partial_download_patterns(modular_dir, tmp_path):
    """The files modular_file_patterns selects are enough to load the task (simulated Hub download)."""
    import fnmatch
    import shutil
    from mammoth.hf_integration.to_hf.mammoth_hub import modular_file_patterns

    manifest = json.loads((modular_dir / "config.json").read_text())
    patterns = modular_file_patterns(manifest, task="eng-fra")
    copied = []
    for path in modular_dir.rglob("*"):
        rel = path.relative_to(modular_dir).as_posix()
        if path.is_file() and any(fnmatch.fnmatch(rel, p) for p in patterns):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(path, tmp_path / rel)
            copied.append(rel)
    components = {c for c in copied if c.startswith("components/")}
    assert components == {f"components/{n}.safetensors" for n in (
        "src_emb__eng", "enc__0__eng", "enc__1__all", "enc_wrapper__eng__all",
        "tgt_emb__fra", "dec__0__fra", "dec__1__all", "dec_wrapper__fra__all")}
    assert not any(c.startswith(("tokenizers/src/deu", "tokenizers/tgt/spa")) for c in copied)
    MammothHub.from_pretrained(str(tmp_path), task="eng-fra")


def test_downloader_collects_combinations(modular_dir, tmp_path, monkeypatch):
    """model_downloader.py fetches a src/tgt(/encoder/decoder) combination; downloads into one
    folder add up, and each collected direction loads offline from it (simulated Hub)."""
    import fnmatch
    import shutil
    from mammoth.hf_integration.to_hf import model_downloader

    def fake_snapshot(repo_id, local_dir, allow_patterns):
        for path in modular_dir.rglob("*"):
            rel = path.relative_to(modular_dir).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(rel, p) for p in allow_patterns):
                os.makedirs(os.path.dirname(os.path.join(local_dir, rel)), exist_ok=True)
                shutil.copy(path, os.path.join(local_dir, rel))

    monkeypatch.setattr(model_downloader, "hf_hub_download", lambda repo, f: str(modular_dir / f))
    monkeypatch.setattr(model_downloader, "snapshot_download", fake_snapshot)
    out = str(tmp_path / "model")
    model_downloader.download("org/x", out, src="deu", tgt="fra")
    model_downloader.download("org/x", out, src="eng", tgt="spa",
                              encoder=["eng", "all"], decoder=["spa", "all"])

    MammothHub.from_pretrained(out, src="deu", tgt="fra")
    MammothHub.from_pretrained(out, src="eng", tgt="spa", encoder=["eng", "all"],
                               decoder=["spa", "all"])
    with pytest.raises(ValueError, match="both --src and --tgt"):
        model_downloader.download("org/x", out, src="deu")


def _card_commands(readme: str) -> list:
    """The `python inference.py ...` lines of a model card, split into argument lists."""
    import shlex
    return [shlex.split(line)[2:] for line in readme.splitlines()
            if line.startswith("python inference.py ")]


def _load_from_card_args(inference, model_dir, args):
    """Run load_model with the direction options a card command uses."""
    kw = {}
    for i, a in enumerate(args):
        if a in ("--task", "--src", "--tgt"):
            kw[a[2:]] = args[i + 1]
        elif a in ("--encoder", "--decoder"):
            ids = args[i + 1:]
            kw[a[2:]] = ids[:next((j for j, x in enumerate(ids) if x.startswith("--")), len(ids))]
    return inference.load_model(model_dir, **kw)


@pytest.mark.parametrize("fmt", ["standalone", "single_artifact", "modular"])
def test_model_card_commands_load(checkpoint, modular_dir, tmp_path, fmt):
    """Each format gets a README whose example commands load with the shipped inference.py."""
    import importlib.util
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import convert_multi_task_artifact
    from mammoth.hf_integration.to_hf.push_to_hub import fill_repo_id

    ckpt, _, _ = checkpoint
    if fmt == "modular":
        out = modular_dir
    elif fmt == "standalone":
        out = tmp_path
        convert(str(ckpt), PREFIX, "eng", "spa", str(out))
    else:
        out = tmp_path
        convert_multi_task_artifact(str(ckpt), str(out))

    readme = (out / "README.md").read_text()
    assert readme.startswith("---\n") and "pipeline_tag: translation" in readme
    commands = _card_commands(readme)
    assert commands and all(c[:2] == ["--model-dir", "<repo_id>"] for c in commands)

    spec = importlib.util.spec_from_file_location("card_inference", out / "inference.py")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    for args in commands:
        _load_from_card_args(inference, str(out), args)

    if fmt == "modular":
        assert "--src deu --tgt fra" in readme   # the untrained pair is offered as zero-shot
        assert "| `eng` | all 2 target languages | eng all | — |" in readme

    if fmt == "standalone":  # fill_repo_id edits in place; keep the shared modular fixture intact
        fill_repo_id(str(out), "org/my-model")
        assert "<repo_id>" not in (out / "README.md").read_text()
