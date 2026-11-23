"""Where the Mammoth server reads a model from.

A ModelSource exposes a model as a modular manifest (the config.json layout written by
convert_mammoth_to_hf.py --modular: architecture, components, tokenizers, tasks) plus
on-demand access to single components and tokenizers. Two implementations:

* HFSource          a modular HF repo: local directory or Hub repo id
* CheckpointSource  a Mammoth training checkpoint directory (sharded *.pt files)

Everything above this layer (ComponentPool, the server) sees only the manifest and
component names, so both formats share one code path.
"""

import glob
import os
from abc import ABC, abstractmethod
from typing import Optional

from transformers import PreTrainedTokenizerFast

from mammoth.hf_integration.to_hf.mammoth_hub import (
    MODULAR_MODEL_TYPE, _fetch, _load_tensors, _read_manifest, filter_component_state,
)


class ModelSource(ABC):
    manifest: dict

    @abstractmethod
    def load_component(self, name: str) -> dict:
        """State dict of one component, keyed as in the Mammoth shard (see component_to_hf)."""

    @abstractmethod
    def load_tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        """Tokenizer of language `lang` on side 'src' or 'tgt'."""


class HFSource(ModelSource):
    """A modular HF repo (convert_mammoth_to_hf.py --modular); files are fetched on demand."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.manifest = _read_manifest(model_id)
        model_type = self.manifest.get("model_type")
        if model_type != MODULAR_MODEL_TYPE:
            raise ValueError(
                f"{model_id} is not a modular Mammoth repo (config.json model_type={model_type!r}). "
                f"Serve the training checkpoint directory instead, or re-export it with "
                f"convert_mammoth_to_hf.py --modular.")

    def load_component(self, name: str) -> dict:
        file = self.manifest["components"][name]["file"]
        return _load_tensors(os.path.join(_fetch(self.model_id, [file]), file))

    def load_tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        tok_dir = self.manifest["tokenizers"][side][lang]["dir"]
        repo_path = _fetch(self.model_id, [f"{tok_dir}/*"])
        return PreTrainedTokenizerFast.from_pretrained(os.path.join(repo_path, tok_dir))


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


def open_source(model: str, step: Optional[int] = None) -> ModelSource:
    """A checkpoint directory (has *_frame.pt) → CheckpointSource; anything else → HFSource."""
    if os.path.isdir(model) and glob.glob(os.path.join(model, "*_frame.pt")):
        return CheckpointSource(model, step=step)
    if step is not None:
        raise ValueError("step only applies to training checkpoint directories")
    return HFSource(model)
