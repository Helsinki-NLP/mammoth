"""ComponentPool: serve several translation directions while loading each component once.

A direction (task) "src-tgt" needs a set of components (embeddings, layer stacks, wrapper
modules; see mammoth_hub.resolve_components). The pool keeps every loaded component as a
module, and builds each task's MammothForConditionalGeneration out of those very modules:
tasks that share a component share the same parameters in memory. Adding a task loads only
the components no loaded task has yet; removing one frees the components nobody uses.

Directions never trained together (zero-shot) are served exactly like trained tasks: the
encoder path is the one the source language was trained with, the decoder path the one
the target language was trained with.
"""

import re
import threading
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.mammoth_hub import (
    component_to_hf, model_config, resolve_components,
)
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration
from mammoth.serving.sources import ModelSource

# Submodules of MammothForConditionalGeneration that each come whole from one component.
# Together they hold every parameter of the model.
_GRAFT_POINT = re.compile(r"^(?:encoder|decoder)\.(?:token_emb|post_emb_norm|to_logits|stacks\.\d+)\.")


def default_path(manifest: dict, side: str, lang: str) -> list:
    """The encoder (side='src') or decoder (side='tgt') path to use for `lang`.

    The path most trained tasks use for that language; ties go to the alphabetically first.
    """
    key = "encoder" if side == "src" else "decoder"
    counts = Counter(tuple(t[key]) for t in manifest["tasks"].values() if t[side] == lang)
    if not counts:
        role = "source" if side == "src" else "target"
        raise ValueError(f"{lang!r} is never a {role} language in this model, so it has no {key}")
    return list(min(counts, key=lambda path: (-counts[path], path)))


def resolve_task(manifest: dict, src: str, tgt: str) -> tuple[dict, bool]:
    """(spec, trained) for direction src→tgt; spec as returned by resolve_components."""
    trained = sorted(name for name, t in manifest["tasks"].items()
                     if t["src"] == src and t["tgt"] == tgt)
    if trained:
        t = manifest["tasks"][trained[0]]
        encoder, decoder = t["encoder"], t["decoder"]
    else:
        encoder, decoder = default_path(manifest, "src", src), default_path(manifest, "tgt", tgt)
    spec = resolve_components(manifest, src=src, tgt=tgt, encoder=encoder, decoder=decoder)
    return spec, bool(trained)


def split_task(task: str) -> tuple[str, str]:
    src, sep, tgt = task.partition("-")
    if not (src and sep and tgt):
        raise ValueError(f"Task must look like 'src-tgt', got {task!r}")
    return src, tgt


@dataclass
class LoadedTask:
    name: str
    spec: dict  # src, tgt, encoder, decoder, components
    trained: bool
    model: MammothForConditionalGeneration
    src_tokenizer: PreTrainedTokenizerFast
    tgt_tokenizer: PreTrainedTokenizerFast


@dataclass
class _Component:
    modules: dict  # graft point (e.g. "encoder.stacks.1") → module
    users: set = field(default_factory=set)


class ComponentPool:
    def __init__(self, source: ModelSource, device: str = "cpu", dtype: torch.dtype = torch.float32):
        self.source = source
        self.manifest = source.manifest
        self.device = torch.device(device)
        self.dtype = dtype
        self.tasks: dict[str, LoadedTask] = {}
        self._components: dict[str, _Component] = {}
        self._tokenizers: dict[tuple, PreTrainedTokenizerFast] = {}
        self._lock = threading.RLock()

    # -- what can be served --------------------------------------------------

    def available_tasks(self) -> dict:
        """Every direction this model can serve: name → {src, tgt, encoder, decoder, trained}.

        Trained tasks plus every zero-shot src→tgt combination whose components exist.
        """
        srcs = sorted({t["src"] for t in self.manifest["tasks"].values()})
        tgts = sorted({t["tgt"] for t in self.manifest["tasks"].values()})
        out = {}
        for src in srcs:
            for tgt in tgts:
                try:
                    spec, trained = resolve_task(self.manifest, src, tgt)
                except ValueError:
                    continue
                if src == tgt and not trained:
                    continue
                out[f"{src}-{tgt}"] = dict(src=src, tgt=tgt, encoder=spec["encoder"],
                                           decoder=spec["decoder"], trained=trained)
        return out

    def loaded_components(self) -> dict:
        """Loaded component name → sorted names of the tasks using it."""
        return {name: sorted(c.users) for name, c in self._components.items()}

    # -- loading and unloading -------------------------------------------------

    def add_task(self, task: str) -> LoadedTask:
        """Load direction "src-tgt" (no-op if loaded); only missing components are read."""
        with self._lock:
            if task in self.tasks:
                return self.tasks[task]
            src, tgt = split_task(task)
            spec, trained = resolve_task(self.manifest, src, tgt)

            with torch.device("meta"):
                model = MammothForConditionalGeneration(MammothConfig(**model_config(self.manifest, spec)))
            # Newly read components join the pool only once the task is complete, so a
            # failure part-way leaves the pool as it was.
            new = {}
            for name, (kind, stack) in spec["components"].items():
                comp = self._components.get(name)
                if comp is None:
                    comp = new[name] = self._load_component(model, name, kind, stack)
                for point, module in comp.modules.items():
                    model.set_submodule(point, module)
            leftover = [n for n, t in model.state_dict().items() if t.is_meta]
            if leftover:
                raise RuntimeError(f"Components {list(spec['components'])} leave model weights "
                                   f"unset: {leftover[:10]}")
            model.eval()
            loaded = LoadedTask(
                name=task, spec=spec, trained=trained, model=model,
                src_tokenizer=self._tokenizer("src", src), tgt_tokenizer=self._tokenizer("tgt", tgt))

            self._components.update(new)
            for name in spec["components"]:
                self._components[name].users.add(task)
            self.tasks[task] = loaded
            return loaded

    def remove_task(self, task: str) -> None:
        """Unload a task; components and tokenizers no other loaded task uses are freed."""
        with self._lock:
            loaded = self.tasks.pop(task, None)
            if loaded is None:
                raise KeyError(f"Task {task!r} is not loaded")
            for name in loaded.spec["components"]:
                comp = self._components[name]
                comp.users.discard(task)
                if not comp.users:
                    del self._components[name]
            in_use = {key for t in self.tasks.values()
                      for key in (("src", t.spec["src"]), ("tgt", t.spec["tgt"]))}
            for key in list(self._tokenizers):
                if key not in in_use:
                    del self._tokenizers[key]

    # -- internals ---------------------------------------------------------------

    def _load_component(self, model: nn.Module, name: str, kind: str, stack) -> _Component:
        """Read one component and materialize the submodules of `model` (on meta) it fills."""
        by_point = {}
        for key, tensor in component_to_hf(kind, self.source.load_component(name), stack=stack).items():
            m = _GRAFT_POINT.match(key)
            if m is None:
                raise RuntimeError(f"Component {name}: unexpected weight {key}")
            point = m.group(0)[:-1]
            by_point.setdefault(point, {})[key[len(point) + 1:]] = tensor
        modules = {}
        for point, state in by_point.items():
            module = model.get_submodule(point).to_empty(device=self.device).to(self.dtype)
            module.load_state_dict(state, strict=True)
            module.requires_grad_(False)
            modules[point] = module
        return _Component(modules)

    def _tokenizer(self, side: str, lang: str) -> PreTrainedTokenizerFast:
        if (side, lang) not in self._tokenizers:
            self._tokenizers[(side, lang)] = self.source.load_tokenizer(side, lang)
        return self._tokenizers[(side, lang)]
