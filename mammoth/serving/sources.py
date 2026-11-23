"""Where the Mammoth server reads a model from.

A ModelSource exposes a model as a modular manifest (the config.json layout written by
convert_mammoth_to_hf.py --modular: architecture, components, tokenizers, tasks) plus
on-demand access to single components and tokenizers. Two implementations:

* HFSource          a modular HF repo: local directory or Hub repo id
* ArtifactSource    per-task HF artifacts: a standalone single-task model, or a
                    --single-artifact bundle (one fused weight file per task)
* CheckpointSource  a Mammoth training checkpoint directory (sharded *.pt files)

Everything above this layer (ComponentPool, the server) sees only the manifest and
component names, so both formats share one code path.
"""

import glob
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import PreTrainedTokenizerFast

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.mammoth_hub import (
    ARCH_KEYS, LEGACY_MODEL_TYPE, MODULAR_MODEL_TYPE, _fetch, _load_tensors, _read_manifest,
    component_name, filter_component_state,
)

STANDALONE_MODEL_TYPE = "mammoth"


class ModelSource(ABC):
    manifest: dict

    @abstractmethod
    def load_component(self, name: str) -> dict:
        """State dict of one component, keyed as in the Mammoth shard (see component_to_hf)."""

    @abstractmethod
    def load_tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        """Tokenizer of language `lang` on side 'src' or 'tgt'."""


def _repo_tokenizer(model_id: str, tok_dir: str) -> PreTrainedTokenizerFast:
    repo_path = _fetch(model_id, [f"{tok_dir}/*"])
    return PreTrainedTokenizerFast.from_pretrained(os.path.join(repo_path, tok_dir))


class HFSource(ModelSource):
    """A modular HF repo (convert_mammoth_to_hf.py --modular); files are fetched on demand."""

    def __init__(self, model_id: str, manifest: Optional[dict] = None):
        self.model_id = model_id
        self.manifest = manifest or _read_manifest(model_id)
        model_type = self.manifest.get("model_type")
        if model_type != MODULAR_MODEL_TYPE:
            raise ValueError(f"{model_id} is not a modular Mammoth repo "
                             f"(config.json model_type={model_type!r}); use open_source()")

    def load_component(self, name: str) -> dict:
        file = self.manifest["components"][name]["file"]
        return _load_tensors(os.path.join(_fetch(self.model_id, [file]), file))

    def load_tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        return _repo_tokenizer(self.model_id, self.manifest["tokenizers"][side][lang]["dir"])


@dataclass
class _ArtifactTask:
    name: str
    src: str
    tgt: str
    config: MammothConfig
    weight_files: tuple  # candidate file names, first existing one is used
    src_tokenizer_dir: str
    tgt_tokenizer_dir: str

    @property
    def encoder(self) -> list:
        # Without recorded sharing groups, assume the task shares nothing.
        return list(self.config.encoder_sharing_groups or [self.name] * len(self.config.enc_layers))

    @property
    def decoder(self) -> list:
        return list(self.config.decoder_sharing_groups or [self.name] * len(self.config.dec_layers))


def post_emb_norm_owner(norms: list) -> str:
    """Which component a side's post-embedding norm belongs to, judged from its weights.

    norms: one (language, path, {key: tensor}) per task. Training keeps one norm either per
    language ("emb") or per encoder/decoder path ("wrapper"); fused per-task files do not say
    which, so pick the grouping under which the tasks agree.
    """
    def consistent(group: int) -> bool:
        seen = {}
        for item in norms:
            ref = seen.setdefault(item[group], item[2])
            if ref.keys() != item[2].keys() or not all(torch.equal(ref[k], item[2][k]) for k in ref):
                return False
        return True

    if consistent(0):
        return "emb"
    if consistent(1):
        return "wrapper"
    raise ValueError("Post-embedding norm weights differ between tasks that share both the "
                     "language and the encoder/decoder path; the artifact cannot be split "
                     "into components")


class ArtifactSource(ModelSource):
    """Per-task HF artifacts, split back into components.

    * standalone (config.json model_type "mammoth"): one task, model.safetensors
    * bundle (model_type "mammoth_hub", --single-artifact): {task}.safetensors per task

    Each component is named from the task's languages and sharing groups, exactly as in a
    modular repo, and the post-embedding norm goes to the language embedding or to the
    wrapper as config.json's post_emb_norm_scope says (judged from the weights for exports
    that predate it), so tasks of a bundle share components in memory and zero-shot directions
    work. A component found in several task files was exported from the same checkpoint
    shard, so the first task's copy is used.
    """

    def __init__(self, model_id: str, config: Optional[dict] = None, task: Optional[str] = None):
        self.model_id = model_id
        config = config or _read_manifest(model_id)
        self._weight_paths, self._bin_cache = {}, (None, None)
        model_type = config.get("model_type")
        if model_type == LEGACY_MODEL_TYPE:
            if task is not None:
                raise ValueError("task only names the direction of a standalone model")
            self._tasks = [
                _ArtifactTask(name, t["_src"], t["_tgt"], _config(t),
                              (f"{name}.safetensors", f"{name}.bin"),
                              f"{name}_src_tokenizer", f"{name}_tgt_tokenizer")
                for name, t in config["tasks"].items()]
        elif model_type == STANDALONE_MODEL_TYPE:
            src, tgt = config.get("src_lang"), config.get("tgt_lang")
            if task is not None:
                src, _, tgt = task.partition("-")
            if not (src and tgt):
                raise ValueError(
                    f"{model_id} does not record its languages (converted before src_lang/"
                    f"tgt_lang were saved); name the direction, e.g. task='eng-spa'")
            cfg = _config(config)
            self._tasks = [_ArtifactTask(f"{src}-{tgt}", src, tgt, cfg,
                                         ("model.safetensors", "pytorch_model.bin"),
                                         cfg.src_tokenizer_dir, cfg.tgt_tokenizer_dir)]
        else:
            raise ValueError(f"{model_id}: unsupported config.json model_type {model_type!r}")
        self.manifest = self._build_manifest()

    def _build_manifest(self) -> dict:
        arch = {k: self._tasks[0].config.to_dict()[k] for k in ARCH_KEYS}
        tasks, components, tokenizers = {}, {}, {"src": {}, "tgt": {}}
        self._where = {}  # component name → (task index, kind, stack)
        for i, t in enumerate(self._tasks):
            cfg = t.config.to_dict()
            if {k: cfg[k] for k in ARCH_KEYS} != arch:
                raise ValueError(f"Task {t.name} has a different architecture than {self._tasks[0].name}")
            tasks[t.name] = dict(src=t.src, tgt=t.tgt, encoder=t.encoder, decoder=t.decoder)
            wanted = [("src_emb", component_name("src_emb", lang=t.src), None,
                       dict(lang=t.src, vocab_size=t.config.src_vocab_size)),
                      ("enc_wrapper", component_name("enc_wrapper", path=t.encoder), None,
                       dict(path=t.encoder)),
                      ("tgt_emb", component_name("tgt_emb", lang=t.tgt), None,
                       dict(lang=t.tgt, vocab_size=t.config.tgt_vocab_size)),
                      ("dec_wrapper", component_name("dec_wrapper", path=t.decoder), None,
                       dict(path=t.decoder, vocab_size=t.config.tgt_vocab_size))]
            for kind, path in (("enc_stack", t.encoder), ("dec_stack", t.decoder)):
                wanted += [(kind, component_name(kind, stack=j, xcoder_id=x), j,
                            dict(stack=j, xcoder_id=x)) for j, x in enumerate(path)]
            for kind, name, stack, meta in wanted:
                if name not in components:
                    components[name] = dict(kind=kind, **meta)
                    self._where[name] = (i, kind, stack)
            for side, lang, tok_dir in (("src", t.src, t.src_tokenizer_dir),
                                        ("tgt", t.tgt, t.tgt_tokenizer_dir)):
                tokenizers[side].setdefault(lang, {
                    "dir": tok_dir, "bos_token_id": t.config.bos_token_id,
                    "eos_token_id": t.config.eos_token_id, "pad_token_id": t.config.pad_token_id,
                    "unk_token_id": 3,
                })

        self._norm_owner = {}
        if arch["post_emb_norm"]:
            scopes = {getattr(t.config, "post_emb_norm_scope", None) for t in self._tasks}
            if len(scopes) > 1:
                raise ValueError(f"Tasks disagree on post_emb_norm_scope: {sorted(map(str, scopes))}")
            scope = scopes.pop()
            for side, key in (("src", "encoder"), ("tgt", "decoder")):
                if scope is not None:
                    self._norm_owner[side] = {"language": "emb", "component": "wrapper"}[scope]
                else:  # exported before the scope was recorded: judge from the weights
                    self._norm_owner[side] = post_emb_norm_owner([
                        (getattr(t, side), tuple(getattr(t, key)), self._read(i, f"{key}.post_emb_norm."))
                        for i, t in enumerate(self._tasks)])
        return {"model_type": MODULAR_MODEL_TYPE, "format_version": 1, "architecture": arch,
                "components": components, "tokenizers": tokenizers, "tasks": tasks}

    def load_component(self, name: str) -> dict:
        i, kind, stack = self._where[name]
        side, xcoder = ("src", "encoder") if kind.startswith(("src", "enc")) else ("tgt", "decoder")
        # (prefix in the fused HF state dict, prefix in the Mammoth shard)
        prefixes = {
            "src_emb": [("encoder.token_emb.", "")], "tgt_emb": [("decoder.token_emb.", "")],
            "enc_stack": [(f"encoder.stacks.{stack}.", "")], "dec_stack": [(f"decoder.stacks.{stack}.", "")],
            "enc_wrapper": [], "dec_wrapper": [("decoder.to_logits.", "to_logits.")],
        }[kind]
        owner = "emb" if kind.endswith("emb") else "wrapper" if kind.endswith("wrapper") else None
        if owner and self._norm_owner.get(side) == owner:
            prefixes.append((f"{xcoder}.post_emb_norm.", "post_emb_norm."))
        state = {}
        for hf_prefix, shard_prefix in prefixes:
            state.update({shard_prefix + k[len(hf_prefix):]: v
                          for k, v in self._read(i, hf_prefix).items()})
        return state

    def load_tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        return _repo_tokenizer(self.model_id, self.manifest["tokenizers"][side][lang]["dir"])

    def _read(self, i: int, prefix: str) -> dict:
        """Tensors of task i's weight file whose names start with prefix."""
        path = self._weight_path(i)
        if path.endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(path, framework="pt") as f:
                return {k: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
        if self._bin_cache[0] != path:
            self._bin_cache = (path, torch.load(path, map_location="cpu", weights_only=True))
        return {k: v for k, v in self._bin_cache[1].items() if k.startswith(prefix)}

    def _weight_path(self, i: int) -> str:
        if i not in self._weight_paths:
            files = self._tasks[i].weight_files
            repo_path = _fetch(self.model_id, list(files))
            found = [os.path.join(repo_path, f) for f in files if os.path.exists(os.path.join(repo_path, f))]
            if not found:
                raise FileNotFoundError(f"No weight file {files} for task {self._tasks[i].name} in {repo_path}")
            self._weight_paths[i] = found[0]
        return self._weight_paths[i]


def _config(config_dict: dict) -> MammothConfig:
    return MammothConfig.from_dict({k: v for k, v in config_dict.items() if not k.startswith("_")})


class CheckpointSource(ModelSource):
    """A Mammoth training checkpoint directory, read shard by shard without conversion."""

    def __init__(self, ckpt_dir: str, step: Optional[int] = None):
        # Imported here: the converter pulls in the full mammoth training stack.
        from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
            _load, modular_manifest, resolve_prefix,
        )
        self._load = _load
        self.ckpt_dir = ckpt_dir
        prefix = resolve_prefix(ckpt_dir, step)
        frame = _load(os.path.join(ckpt_dir, f"{prefix}_frame.pt"))
        self._vocab = frame["vocab"]
        self.manifest, self._shard_paths = modular_manifest(
            ckpt_dir, prefix, frame["opts"], self._vocab)

    def load_component(self, name: str) -> dict:
        kind = self.manifest["components"][name]["kind"]
        return filter_component_state(kind, self._load(self._shard_paths[name]))

    def load_tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import build_tokenizer
        tok = self.manifest["tokenizers"][side][lang]
        ids = dict(bos=tok["bos_token_id"], eos=tok["eos_token_id"],
                   pad=tok["pad_token_id"], unk=tok["unk_token_id"])
        return build_tokenizer(self._vocab[(side, lang)], ids, add_bos_eos=(side == "src"))


def open_source(model: str, step: Optional[int] = None, task: Optional[str] = None) -> ModelSource:
    """Open a training checkpoint directory (has *_frame.pt), or an HF repo of any Mammoth format.

    step picks a checkpoint step; task names the direction of a standalone HF model whose
    config.json does not record its languages.
    """
    if os.path.isdir(model) and glob.glob(os.path.join(model, "*_frame.pt")):
        if task is not None:
            raise ValueError("task only names the direction of a standalone HF model")
        return CheckpointSource(model, step=step)
    if step is not None:
        raise ValueError("step only applies to training checkpoint directories")
    config = _read_manifest(model)
    if config.get("model_type") == MODULAR_MODEL_TYPE:
        if task is not None:
            raise ValueError("task only names the direction of a standalone HF model")
        return HFSource(model, config)
    return ArtifactSource(model, config, task=task)
