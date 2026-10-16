"""MammothHub: load a translation model from a Mammoth HF repo.

Two repo formats are supported (detected from config.json "model_type"):

* ``mammoth_modular`` (convert_mammoth_to_hf.py --modular): every Mammoth
  component is stored once — one file per language embedding, per
  (layer_stack, xcoder_id) encoder/decoder stack, and per wrapper path. A model
  is assembled at load time from the components a task needs, exactly like
  Mammoth inference does. Components shared by several tasks are stored and
  downloaded once.
* ``mammoth_hub`` (convert_mammoth_to_hf.py --single-artifact, legacy): one
  fused weight file per task.

Internal loader: users should go through inference.py (shipped in every
artifact), whose load_model() routes multi-task repos here. Entry points:
MammothHub.from_pretrained(model_id, task=... | src=, tgt=, encoder=, decoder=),
MammothHub.list_tasks(model_id), MammothHub.list_components(model_id).

A local directory path works anywhere a Hub repo id does.
"""

import json
import os
import shutil
import tempfile
from typing import Optional, Sequence

from transformers import PreTrainedTokenizerFast

CODE_FILES = ("configuration_mammoth.py", "modeling_mammoth.py", "native_transformer.py")

MODULAR_MODEL_TYPE = "mammoth_modular"
LEGACY_MODEL_TYPE = "mammoth_hub"


# ---------------------------------------------------------------------------
# Component naming and component → HF key mapping.
# Shared with convert_mammoth_to_hf.py so export and load cannot drift apart.
# ---------------------------------------------------------------------------

# Component kinds, and which keys of a Mammoth shard each one contributes.
#   src_emb / tgt_emb          nn.Embedding                → "weight"
#   enc_stack / dec_stack      TransformerStack            → every key
#   enc_wrapper / dec_wrapper  per-path wrapper modules    → post_emb_norm.*, to_logits.*
_KEPT_PREFIXES = {
    "src_emb": ("weight",),
    "tgt_emb": ("weight",),
    "enc_stack": ("",),
    "dec_stack": ("",),
    "enc_wrapper": ("post_emb_norm.",),
    "dec_wrapper": ("post_emb_norm.", "to_logits."),
}

_HF_PREFIX = {
    "src_emb": "encoder.token_emb.",
    "tgt_emb": "decoder.token_emb.",
    "enc_stack": "encoder.stacks.{stack}.",
    "dec_stack": "decoder.stacks.{stack}.",
    "enc_wrapper": "encoder.",
    "dec_wrapper": "decoder.",
}


def component_name(kind: str, *, lang: str = None, stack: int = None,
                   xcoder_id: str = None, path: Sequence[str] = None) -> str:
    """Canonical component name, also used as its file stem, e.g. 'enc__1__all'."""
    if kind in ("src_emb", "tgt_emb"):
        return f"{kind}__{lang}"
    if kind in ("enc_stack", "dec_stack"):
        return f"{kind[:3]}__{stack}__{xcoder_id}"
    if kind in ("enc_wrapper", "dec_wrapper"):
        return f"{kind}__{'__'.join(path)}"
    raise ValueError(f"Unknown component kind {kind!r}")


def filter_component_state(kind: str, state_dict: dict) -> dict:
    """Keep only the shard keys the HF model consumes for this component kind."""
    kept = _KEPT_PREFIXES[kind]
    return {k: v for k, v in state_dict.items() if any(k.startswith(p) for p in kept)}


def component_to_hf(kind: str, state_dict: dict, stack: int = None) -> dict:
    """Map a component's (Mammoth shard) state dict to MammothForConditionalGeneration keys."""
    prefix = _HF_PREFIX[kind].format(stack=stack)
    return {prefix + k: v for k, v in filter_component_state(kind, state_dict).items()}


# ---------------------------------------------------------------------------
# Repo access
# ---------------------------------------------------------------------------

def _read_manifest(model_id: str) -> dict:
    if os.path.isdir(model_id):
        path = os.path.join(model_id, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(model_id, "config.json")
    with open(path) as f:
        return json.load(f)


def _fetch(model_id: str, patterns: list, local_dir: Optional[str] = None) -> str:
    """Return a local directory containing at least the files matching `patterns`.

    Hub files go to `local_dir` when given (a plain, browsable directory that can be
    passed back as `model_id` later), otherwise to the HF cache.
    """
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_id, allow_patterns=patterns, local_dir=local_dir)


def _load_tensors(path: str) -> dict:
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def _instantiate(config_dict: dict, code_dir: str):
    """Build an (untrained) MammothForConditionalGeneration from the repo's vendored code."""
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    tmp_dir = tempfile.mkdtemp(prefix="mammoth_hub_")
    try:
        with open(os.path.join(tmp_dir, "config.json"), "w") as f:
            json.dump(config_dict, f)
        for fname in CODE_FILES:
            src = os.path.join(code_dir, fname)
            if os.path.exists(src):
                os.symlink(src, os.path.join(tmp_dir, fname))
        config = AutoConfig.from_pretrained(tmp_dir, trust_remote_code=True)
        return AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


_AUTO_MAP = {
    "AutoConfig": "configuration_mammoth.MammothConfig",
    "AutoModelForSeq2SeqLM": "modeling_mammoth.MammothForConditionalGeneration",
}


# ---------------------------------------------------------------------------
# Modular repos: resolving which components a model needs
# ---------------------------------------------------------------------------

def _infer_path(manifest: dict, side: str, lang: str) -> list:
    """Pick the encoder (side='src') or decoder (side='tgt') path trained tasks use for `lang`."""
    key = "encoder" if side == "src" else "decoder"
    paths = {tuple(t[key]) for t in manifest["tasks"].values() if t[side] == lang}
    if len(paths) == 1:
        return list(paths.pop())
    if not paths:
        raise ValueError(
            f"No trained task has {side}={lang!r}, so its {key} path cannot be inferred. "
            f"Pass {key}=[...] explicitly.")
    raise ValueError(
        f"Trained tasks use several {key} paths for {side}={lang!r}: "
        f"{sorted(list(p) for p in paths)}. Pass {key}=[...] explicitly.")


def resolve_components(manifest: dict, task: Optional[str] = None, src: Optional[str] = None,
                       tgt: Optional[str] = None, encoder: Optional[Sequence[str]] = None,
                       decoder: Optional[Sequence[str]] = None) -> dict:
    """Return the spec {src, tgt, encoder, decoder, components} for one model.

    `components` maps component name → (kind, stack index or None), in load order.
    Raises ValueError with a readable reason if any required component is absent.
    """
    if task is not None:
        if task not in manifest["tasks"]:
            raise ValueError(f"Task {task!r} not found. Available tasks: {sorted(manifest['tasks'])}")
        t = manifest["tasks"][task]
        src, tgt = t["src"], t["tgt"]
        encoder = encoder or t["encoder"]
        decoder = decoder or t["decoder"]
    elif src is None or tgt is None:
        raise ValueError("Pass either task='src-tgt' or both src= and tgt=.")
    encoder = list(encoder) if encoder else _infer_path(manifest, "src", src)
    decoder = list(decoder) if decoder else _infer_path(manifest, "tgt", tgt)

    arch = manifest["architecture"]
    if len(encoder) != len(arch["enc_layers"]):
        raise ValueError(f"encoder path {encoder} needs {len(arch['enc_layers'])} entries "
                         f"(one per encoder layer stack)")
    if len(decoder) != len(arch["dec_layers"]):
        raise ValueError(f"decoder path {decoder} needs {len(arch['dec_layers'])} entries "
                         f"(one per decoder layer stack)")

    wanted = [(component_name("src_emb", lang=src), "src_emb", None),
              (component_name("enc_wrapper", path=encoder), "enc_wrapper", None)]
    wanted += [(component_name("enc_stack", stack=i, xcoder_id=x), "enc_stack", i)
               for i, x in enumerate(encoder)]
    wanted += [(component_name("tgt_emb", lang=tgt), "tgt_emb", None),
               (component_name("dec_wrapper", path=decoder), "dec_wrapper", None)]
    wanted += [(component_name("dec_stack", stack=i, xcoder_id=x), "dec_stack", i)
               for i, x in enumerate(decoder)]

    available = manifest["components"]
    missing = [name for name, _, _ in wanted if name not in available]
    if missing:
        hint = ""
        if any("wrapper__" in m for m in missing):
            hint = ("\nWrapper modules (post_emb_norm, to_logits) are trained per complete "
                    "encoder/decoder path, so only paths used by some trained task can be "
                    "loaded. Trained paths: encoder "
                    f"{sorted({tuple(t['encoder']) for t in manifest['tasks'].values()})}, decoder "
                    f"{sorted({tuple(t['decoder']) for t in manifest['tasks'].values()})}.")
        raise ValueError(f"Components not in this repo: {missing}{hint}")

    for side, lang in (("src", src), ("tgt", tgt)):
        if lang not in manifest["tokenizers"][side]:
            raise ValueError(f"No {side} tokenizer for language {lang!r} in this repo")

    tgt_vocab = available[component_name("dec_wrapper", path=decoder)].get("vocab_size")
    tgt_emb_vocab = available[component_name("tgt_emb", lang=tgt)]["vocab_size"]
    if tgt_vocab is not None and tgt_vocab != tgt_emb_vocab:
        raise ValueError(f"decoder path {decoder} projects to {tgt_vocab} tokens, but the "
                         f"{tgt!r} target vocabulary has {tgt_emb_vocab}")

    return {
        "src": src, "tgt": tgt, "encoder": encoder, "decoder": decoder,
        "components": {name: (kind, stack) for name, kind, stack in wanted},
    }


def modular_file_patterns(manifest: dict, **selection) -> list:
    """Repo files (allow_patterns) needed for one model of a modular repo.

    `selection` is passed to resolve_components: task=..., or src/tgt/encoder/decoder.
    """
    spec = resolve_components(manifest, **selection)
    tokenizers = manifest["tokenizers"]
    patterns = ["config.json", "*.py"]
    patterns += [manifest["components"][name]["file"] for name in spec["components"]]
    patterns += [f"{tokenizers['src'][spec['src']]['dir']}/*",
                 f"{tokenizers['tgt'][spec['tgt']]['dir']}/*"]
    return patterns


class MammothHub:
    """Loads a translation model from a Mammoth HF repo (modular or legacy bundle)."""

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        task: Optional[str] = None,
        device: Optional[str] = None,
        *,
        src: Optional[str] = None,
        tgt: Optional[str] = None,
        encoder: Optional[Sequence[str]] = None,
        decoder: Optional[Sequence[str]] = None,
        local_dir: Optional[str] = None,
    ):
        """Load one translation model.

        Args:
            model_id: Local directory path or HF Hub repo id (e.g. "org/repo").
            task: Trained task name, e.g. "eng-spa".
            device: Device to load the model on (default: cpu).
            src, tgt: Source/target language — alternative to `task` (modular repos only).
            encoder, decoder: xcoder_id per layer stack, e.g. ["eng", "all"]. Override the
                task's paths, or define new combinations (modular repos only).
            local_dir: Download the needed Hub files into this directory instead of the
                HF cache. Ignored when `model_id` is already a local directory.

        Returns:
            A MammothForConditionalGeneration in eval mode, with .src_tokenizer and
            .tgt_tokenizer attached (and .mammoth_spec for modular repos).
        """
        manifest = _read_manifest(model_id)
        model_type = manifest.get("model_type")
        if model_type == MODULAR_MODEL_TYPE:
            model = cls._load_modular(model_id, manifest, task, src, tgt, encoder, decoder,
                                      local_dir)
        elif model_type == LEGACY_MODEL_TYPE:
            if task is None or any(v is not None for v in (src, tgt, encoder, decoder)):
                raise ValueError("Per-task bundles only support task='src-tgt'; convert with "
                                 "--modular to combine components freely.")
            model = cls._load_legacy(model_id, manifest, task, local_dir)
        else:
            raise ValueError(f"{model_id} is not a Mammoth multi-task repo "
                             f"(config.json model_type={model_type!r})")
        if device:
            model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def list_tasks(model_id: str) -> dict:
        """Trained tasks: name → {src, tgt, encoder, decoder} (or task config for legacy bundles)."""
        return _read_manifest(model_id).get("tasks", {})

    @staticmethod
    def list_components(model_id: str) -> dict:
        """Components of a modular repo: name → {kind, file, ...}."""
        manifest = _read_manifest(model_id)
        if manifest.get("model_type") != MODULAR_MODEL_TYPE:
            raise ValueError(f"{model_id} is not a modular Mammoth repo")
        return manifest["components"]

    # -- modular ------------------------------------------------------------

    @staticmethod
    def _load_modular(model_id, manifest, task, src, tgt, encoder, decoder, local_dir=None):
        spec = resolve_components(manifest, task=task, src=src, tgt=tgt,
                                  encoder=encoder, decoder=decoder)
        components = manifest["components"]
        src_tok = manifest["tokenizers"]["src"][spec["src"]]
        tgt_tok = manifest["tokenizers"]["tgt"][spec["tgt"]]

        repo_path = _fetch(model_id, modular_file_patterns(
            manifest, src=spec["src"], tgt=spec["tgt"],
            encoder=spec["encoder"], decoder=spec["decoder"]), local_dir=local_dir)

        config_dict = dict(manifest["architecture"])
        config_dict.update(
            model_type="mammoth",
            auto_map=_AUTO_MAP,
            src_vocab_size=components[component_name("src_emb", lang=spec["src"])]["vocab_size"],
            tgt_vocab_size=components[component_name("tgt_emb", lang=spec["tgt"])]["vocab_size"],
            # Decoding produces target-side tokens, so the special ids come from the target vocab.
            bos_token_id=tgt_tok["bos_token_id"],
            eos_token_id=tgt_tok["eos_token_id"],
            pad_token_id=tgt_tok["pad_token_id"],
            decoder_start_token_id=tgt_tok["bos_token_id"],
            encoder_sharing_groups=spec["encoder"],
            decoder_sharing_groups=spec["decoder"],
        )
        model = _instantiate(config_dict, repo_path)

        state_dict = {}
        for name, (kind, stack) in spec["components"].items():
            tensors = _load_tensors(os.path.join(repo_path, components[name]["file"]))
            state_dict.update(component_to_hf(kind, tensors, stack=stack))
        result = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Components {list(spec['components'])} do not fill the model exactly.\n"
                f"  missing: {result.missing_keys[:10]}\n  unexpected: {result.unexpected_keys[:10]}")

        model.src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(repo_path, src_tok["dir"]))
        model.tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(repo_path, tgt_tok["dir"]))
        model.mammoth_spec = {k: spec[k] for k in ("src", "tgt", "encoder", "decoder")}
        return model

    # -- legacy per-task bundle ---------------------------------------------

    @staticmethod
    def _load_legacy(model_id, manifest, task, local_dir=None):
        tasks = manifest.get("tasks", {})
        if task not in tasks:
            raise ValueError(f"Task {task!r} not found. Available tasks: {list(tasks.keys())}")

        # Download only the files needed for this task, not every task's weights.
        repo_path = _fetch(model_id, [
            "config.json", "*.py",
            f"{task}.safetensors", f"{task}.bin",
            f"{task}_src_tokenizer/**", f"{task}_tgt_tokenizer/**",
        ], local_dir=local_dir)

        weight_file = os.path.join(repo_path, f"{task}.safetensors")
        if not os.path.exists(weight_file):
            weight_file = os.path.join(repo_path, f"{task}.bin")
        if not os.path.exists(weight_file):
            raise FileNotFoundError(f"No weight file for task {task!r} in {repo_path}")

        # Create a temp dir with symlinks so from_pretrained can load it
        # as a standard single-task HF model.
        tmp_dir = tempfile.mkdtemp(prefix="mammoth_hub_")
        try:
            config_dict = dict(tasks[task])
            config_dict["model_type"] = "mammoth"
            config_dict["auto_map"] = _AUTO_MAP
            with open(os.path.join(tmp_dir, "config.json"), "w") as f:
                json.dump(config_dict, f)

            std_name = "model.safetensors" if weight_file.endswith(".safetensors") \
                else "pytorch_model.bin"
            os.symlink(weight_file, os.path.join(tmp_dir, std_name))
            for fname in CODE_FILES:
                src = os.path.join(repo_path, fname)
                if os.path.exists(src):
                    os.symlink(src, os.path.join(tmp_dir, fname))

            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(tmp_dir, trust_remote_code=True)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        model.src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(repo_path, f"{task}_src_tokenizer"))
        model.tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(repo_path, f"{task}_tgt_tokenizer"))
        return model
