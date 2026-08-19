#!/usr/bin/env python3
"""
inspect-model-files.py

Inspect modular PyTorch checkpoint files produced by Mammoth / x-transformers
models and infer compact architecture summaries from the saved weights.

Main capabilities
-----------------
This script supports several complementary inspection modes:

1. General checkpoint inspection
   - summarize top-level object type and keys
   - count tensors and parameters
   - estimate tensor memory size
   - list largest tensors
   - group tensor keys by module prefix

2. Per-file config-like summary
   - infer model_dim, FF dimension, projection sizes, rotary dimension,
     vocab size, and logical layer counts from tensor shapes
   - distinguish between:
       * configured_num_heads      (from saved options if present)
       * configured_head_dim       (explicit saved head_dim if present)
       * calculated_head_dim       (computed as model_dim / configured_num_heads)
       * trained_head_dim          (computed from q_proj_out_dim / configured_num_heads)

3. Whole-model summary
   - merge representative component files into one YAML-like summary
   - preserve per-role fields for encoder / decoder / wrappers
   - hoist only genuinely shared architectural fields
   - warn when calculated_head_dim and trained_head_dim disagree

4. Saved-layer pattern inspection
   - inspect `_base_layers.<i>` key structure
   - infer logical layer counts from repeated saved block patterns
   - explain differences between raw saved block counts and config-level
     encoder/decoder layer counts

Why the head-dim distinction matters
------------------------------------
Older Mammoth / x-transformers checkpoints may have been trained with a
hardcoded head dimension (commonly 64), while newer code may infer head_dim
from `model_dim / heads`. Therefore this script does not use a single generic
`head_dim` field in summaries. Instead it separates:

- configured_num_heads
- configured_head_dim
- calculated_head_dim
- trained_head_dim

This makes checkpoint/config mismatches visible.

Representative-file sampling
----------------------------
For model-summary mode, the script can reduce very large multilingual model
directories to a small set of representative component files:
- one encoder shard
- one decoder shard
- one encoder wrapper
- all decoder wrappers (to preserve vocab-size ambiguity)
- optional embedding files
- the frame file for saved options

Safety note
-----------
By default, torch.load is used in safe weights-only mode when available.
Use `--unsafe` only for trusted checkpoint files, because it may unpickle
arbitrary Python objects.

The script is intended for inspection/debugging only. It does not modify
checkpoint files.
"""

import argparse
import os
from collections import defaultdict, Counter
from typing import Any

import torch


# Optional safe-loading allowlist for Mammoth / HF tokenizer objects.
# If these imports fail, frame files may still need --unsafe.
try:
    import argparse as _argparse
    from torch.serialization import add_safe_globals
    from mammoth.inputters.vocab import HFTokenizerVocab
    from tokenizers import Tokenizer
    from tokenizers.models import Model
    from tokenizers.normalizers import Normalizer
    from tokenizers.pre_tokenizers import PreTokenizer
    from tokenizers.decoders import Decoder
    from tokenizers.processors import PostProcessor
    from tokenizers.trainers import Trainer

    add_safe_globals([
        _argparse.Namespace,
        HFTokenizerVocab,
        Tokenizer,
        Model,
        Normalizer,
        PreTokenizer,
        Decoder,
        PostProcessor,
        Trainer,
    ])
except Exception:
    pass


def bytes_per_elem(dtype: torch.dtype) -> int:
    if dtype in (torch.float32, torch.int32):
        return 4
    if dtype in (torch.float16, torch.bfloat16, torch.int16):
        return 2
    if dtype in (torch.float64, torch.int64):
        return 8
    if dtype in (torch.int8, torch.uint8, torch.bool):
        return 1
    return 0


def mb(n_bytes: int) -> float:
    return n_bytes / (1024 * 1024)


def label_key(key: str) -> str:
    k = key.lower()

    if "to_logits" in k or "generator" in k or "lm_head" in k:
        return "OutputHead"

    if "embedding" in k or "emb.weight" in k or "embed" in k:
        return "Embedding"

    if "post_emb_norm" in k or "layer_norm" in k or "norm" in k:
        return "Norm"

    if "encoder_attn" in k or "cross_attn" in k or "crossattn" in k:
        return "CrossAttn"

    if "self_attn" in k or "selfattn" in k:
        return "SelfAttn"

    if ".to_q." in k or ".to_k." in k or ".to_v." in k or ".to_out." in k:
        return "AttnProj"

    if ".ff." in k or "ffn" in k or "feed_forward" in k or "mlp" in k:
        return "FFN"

    if "rotary" in k or "inv_freq" in k:
        return "RoPE"

    return "Other"


def group_key(key: str, depth: int) -> str:
    parts = key.split(".")
    if len(parts) <= depth:
        return key
    return ".".join(parts[:depth])


def print_table(headers, rows):
    for r in rows:
        if len(r) != len(headers):
            raise ValueError(
                f"Row has {len(r)} cells but headers have {len(headers)}: {r}"
            )

    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(row):
        return " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))


def get_state_dict_like(obj: Any):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
        return sd if any(torch.is_tensor(v) for v in sd.values()) else None

    if isinstance(obj, dict):
        return obj if any(torch.is_tensor(v) for v in obj.values()) else None

    return None

def layer_index_from_key(key: str):
    parts = key.split(".")
    if len(parts) >= 2 and parts[0] == "_base_layers":
        try:
            return int(parts[1])
        except ValueError:
            return None
    return None


def layer_signature(key: str) -> str:
    """
    Coarse structural label for a tensor key inside _base_layers.<i>.
    This is only for pattern inspection, not exact semantics.
    """
    lk = key.lower()

    if ".to_q." in lk:
        return "to_q"
    if ".to_k." in lk:
        return "to_k"
    if ".to_v." in lk:
        return "to_v"
    if ".to_out." in lk:
        return "to_out"

    if "cross_attn" in lk or "encoder_attn" in lk:
        return "cross_attn"

    if "self_attn" in lk:
        return "self_attn"

    if ".ff." in lk or "ffn" in lk or "feed_forward" in lk or "mlp" in lk:
        return "ff"

    if "norm" in lk or "gamma" in lk or "beta" in lk or "layer_norm" in lk:
        return "norm"

    if "rotary" in lk or "inv_freq" in lk:
        return "rope"

    return "other"


def inspect_layer_patterns(sd: dict[str, torch.Tensor], top_keys_per_layer: int = 12):
    """
    Print per-layer key patterns for _base_layers.<i>.
    Useful for checking whether saved layer count matches config-layer count.
    """
    by_layer = defaultdict(list)

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        i = layer_index_from_key(k)
        if i is not None:
            by_layer[i].append(k)

    if not by_layer:
        print("(No _base_layers.<i> keys found.)")
        return

    print("\n## Layer pattern summary")
    rows = []
    for i in sorted(by_layer):
        keys = by_layer[i]
        sig_counts = Counter(layer_signature(k) for k in keys)
        sig_summary = ", ".join(f"{name}:{sig_counts[name]}" for name in sorted(sig_counts))
        rows.append([i, len(keys), sig_summary])

    print_table(["layer", "#tensor_keys", "pattern_counts"], rows)

    print("\n## Example keys per layer")
    for i in sorted(by_layer):
        print(f"\n### _base_layers.{i}")
        for k in by_layer[i][:top_keys_per_layer]:
            print(k)
        if len(by_layer[i]) > top_keys_per_layer:
            print("...")

def load_checkpoint(path: str, unsafe: bool):
    if unsafe:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu")


def print_config_summary(path: str, cfg: dict[str, Any]):
    print(f"\n=== {path} ===")
    wanted = {
        "num_base_layers",
        "num_layers",
        "num_layers_source",
        "model_dim",
        "q_proj_out_dim",
        "k_proj_out_dim",
        "v_proj_out_dim",
        "ff_hidden_dim",
        "ff_mult",
        "rotary_dim",
        "configured_num_heads",
        "configured_num_heads_source",
        "configured_head_dim",
        "configured_head_dim_source",
        "calculated_head_dim",
        "calculated_head_dim_source",
        "trained_head_dim",
        "trained_head_dim_source",
        "vocab_size",
        "has_cross_attn",
        "dtype",
    }
    rows = [[k, cfg[k]] for k in sorted(cfg) if k in wanted]
    if rows:
        print("\n## Inferred config summary")
        print_table(["field", "value"], rows)
    else:
        print("\n## Inferred config summary")
        print("(Could not infer config-like fields from this file.)")



def infer_logical_layer_count(sd: dict[str, torch.Tensor], role: str) -> dict[str, Any]:
    by_layer = defaultdict(list)

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        i = layer_index_from_key(k)
        if i is not None:
            by_layer[i].append(k)

    if not by_layer:
        return {}

    layer_ids = sorted(by_layer)
    n_base = max(layer_ids) + 1

    signatures = []
    for i in layer_ids:
        sig_counts = Counter(layer_signature(k) for k in by_layer[i])
        has_attn = any(x in sig_counts for x in ["to_q", "to_k", "to_v", "to_out", "self_attn", "cross_attn"])
        has_ff = "ff" in sig_counts

        if has_attn and has_ff:
            signatures.append("mixed")
        elif has_attn:
            signatures.append("attn")
        elif has_ff:
            signatures.append("ff")
        else:
            signatures.append("other")

    out = {
        "num_base_layers": n_base,
        "base_layer_pattern": signatures,
    }

    if role == "encoder":
        ok = True
        for j, s in enumerate(signatures):
            expected = "attn" if j % 2 == 0 else "ff"
            if s != expected:
                ok = False
                break
        if ok and n_base % 2 == 0:
            out["num_layers"] = n_base // 2
            out["num_layers_source"] = "num_base_layers / 2 from encoder attn/ff alternation"
        else:
            out["num_layers"] = n_base
            out["num_layers_source"] = "raw num_base_layers"

    elif role == "decoder":
        ok = True
        for j, s in enumerate(signatures):
            expected = "ff" if j % 3 == 2 else "attn"
            if s != expected:
                ok = False
                break
        if ok and n_base % 3 == 0:
            out["num_layers"] = n_base // 3
            out["num_layers_source"] = "num_base_layers / 3 from decoder attn/attn/ff repetition"
        else:
            out["num_layers"] = n_base
            out["num_layers_source"] = "raw num_base_layers"

    else:
        out["num_layers"] = n_base
        out["num_layers_source"] = "raw num_base_layers"

    return out

def collect_scalar_candidates(obj, prefix=""):
    """
    Recursively collect small scalar / dict-like config fields from nested
    Python objects, dicts, lists, tuples, and argparse-style namespaces.
    """
    out = {}

    def visit(x, path):
        # Basic scalars
        if isinstance(x, (str, int, float, bool)) or x is None:
            out[path] = x
            return

        # Dict
        if isinstance(x, dict):
            for k, v in x.items():
                sk = str(k)
                visit(v, f"{path}.{sk}" if path else sk)
            return

        # list / tuple
        if isinstance(x, (list, tuple)):
            # only recurse into short sequences
            if len(x) <= 16:
                for i, v in enumerate(x):
                    visit(v, f"{path}[{i}]")
            return

        # generic object / argparse.Namespace-like
        if hasattr(x, "__dict__"):
            for k, v in vars(x).items():
                if k.startswith("_"):
                    continue
                visit(v, f"{path}.{k}" if path else k)
            return

    visit(obj, prefix)
    return out


def extract_head_hints_from_object(obj):
    """
    Search a loaded checkpoint/frame object for explicit attention-head settings.
    Ignore vocabulary tables and other unrelated mappings.
    """
    flat = collect_scalar_candidates(obj)

    wanted_suffixes = [
        "heads",
        "attn_heads",
        "encoder_heads",
        "decoder_heads",
        "head_dim",
        "dim_head",
    ]

    banned_fragments = [
        ".stoi.",
        ".itos.",
        "vocab.",
        ".freqs.",
        ".counter.",
    ]

    hits = {}
    for k, v in flat.items():
        lk = k.lower()

        if any(bad in lk for bad in banned_fragments):
            continue

        for suf in wanted_suffixes:
            if lk.endswith(suf):
                if isinstance(v, int) and v > 0 and v < 1024:
                    hits[k] = v
                break

    return hits

def resolve_head_info(cfg: dict[str, Any], head_hints: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    model_dim = out.get("model_dim")
    q_proj_out_dim = out.get("q_proj_out_dim")

    explicit_head_dim = None
    explicit_heads = None
    explicit_head_dim_source = None
    explicit_heads_source = None

    preferred_head_dim_keys = [
        "opts.x_transformers_opts.head_dim",
        "opts.x_transformers_opts.dim_head",
        "x_transformers_opts.head_dim",
        "x_transformers_opts.dim_head",
    ]
    preferred_heads_keys = [
        "opts.x_transformers_opts.heads",
        "x_transformers_opts.heads",
        "opts.decoder_heads",
        "opts.encoder_heads",
        "opts.attn_heads",
        "opts.ab_heads",
    ]

    for k in preferred_head_dim_keys:
        if k in head_hints:
            v = head_hints[k]
            if isinstance(v, int) and v > 0:
                explicit_head_dim = v
                explicit_head_dim_source = k
                break

    for k in preferred_heads_keys:
        if k in head_hints:
            v = head_hints[k]
            if isinstance(v, int) and v > 0:
                explicit_heads = v
                explicit_heads_source = k
                break

    if explicit_head_dim is None:
        for k, v in head_hints.items():
            lk = k.lower()
            if (lk.endswith("head_dim") or lk.endswith("dim_head")) and isinstance(v, int) and v > 0:
                explicit_head_dim = v
                explicit_head_dim_source = k
                break

    if explicit_heads is None:
        for k, v in head_hints.items():
            lk = k.lower()
            if (
                lk.endswith("encoder_heads")
                or lk.endswith("decoder_heads")
                or lk.endswith("attn_heads")
                or lk.endswith("heads")
            ) and isinstance(v, int) and v > 0:
                explicit_heads = v
                explicit_heads_source = k
                break

    # Explicit saved head_dim from options
    if explicit_head_dim is not None:
        out["configured_head_dim"] = explicit_head_dim
        out["configured_head_dim_source"] = explicit_head_dim_source

        if explicit_heads is not None:
            out["configured_num_heads"] = explicit_heads
            out["configured_num_heads_source"] = explicit_heads_source
        elif model_dim and model_dim % explicit_head_dim == 0:
            out["configured_num_heads"] = model_dim // explicit_head_dim
            out["configured_num_heads_source"] = f"model_dim / {explicit_head_dim_source}"

        if q_proj_out_dim and out.get("configured_num_heads"):
            nh = out["configured_num_heads"]
            if q_proj_out_dim % nh == 0:
                out["trained_head_dim"] = q_proj_out_dim // nh
                out["trained_head_dim_source"] = f"q_proj_out_dim / configured_num_heads"

        out.pop("likely_head_dim_old", None)
        out.pop("likely_num_heads_old", None)
        return out

    # Explicit saved heads from options
    if explicit_heads is not None:
        out["configured_num_heads"] = explicit_heads
        out["configured_num_heads_source"] = explicit_heads_source

        if model_dim and model_dim % explicit_heads == 0:
            out["calculated_head_dim"] = model_dim // explicit_heads
            out["calculated_head_dim_source"] = f"model_dim / {explicit_heads_source}"

        if q_proj_out_dim and q_proj_out_dim % explicit_heads == 0:
            out["trained_head_dim"] = q_proj_out_dim // explicit_heads
            out["trained_head_dim_source"] = f"q_proj_out_dim / {explicit_heads_source}"

        out.pop("likely_head_dim_old", None)
        out.pop("likely_num_heads_old", None)
        return out

    # Old fallback only when no explicit info exists
    if "likely_num_heads_old" in out:
        out["configured_num_heads"] = out["likely_num_heads_old"]
        out["configured_num_heads_source"] = "old_x_transformers_default"

    if "likely_head_dim_old" in out:
        out["trained_head_dim"] = out["likely_head_dim_old"]
        out["trained_head_dim_source"] = "old_x_transformers_default"

    return out

def add_head_dim_warnings(summary: dict[str, Any]) -> dict[str, Any]:
    if "warnings" not in summary:
        summary["warnings"] = []

    global_cfg = summary.get("global", {})
    if not isinstance(global_cfg, dict):
        return summary

    calc_hd = global_cfg.get("calculated_head_dim")
    trained_hd = global_cfg.get("trained_head_dim")

    if calc_hd is not None and trained_hd is not None and calc_hd != trained_hd:
        global_cfg["requires_legacy_mammoth"] = True
        global_cfg["legacy_mammoth_reason"] = (
            f"calculated_head_dim={calc_hd} but trained_head_dim={trained_hd}"
        )
        summary["warnings"].append(
            f"global: calculated_head_dim ({calc_hd}) differs from trained_head_dim ({trained_hd})")
        summary["warnings"].append(
            f"use older Mammoth compatible with trained_head_dim={trained_hd}")

    return summary

def select_representative_files(paths):
    roles = {
        "encoder": [],
        "decoder": [],
        "encoder_wrapper": [],
        "decoder_wrapper": [],
        "src_embeddings": [],
        "tgt_embeddings": [],
        "frame": [],
    }

    for path in sorted(paths):
        base = os.path.basename(path).lower()

        if base.endswith("_optim.pt"):
            continue
        if base.endswith(".json"):
            continue

        if "frame" in base:
            roles["frame"].append(path)
        elif "encoder_wrapper" in base:
            roles["encoder_wrapper"].append(path)
        elif "decoder_wrapper" in base:
            roles["decoder_wrapper"].append(path)
        elif "src_embeddings" in base:
            roles["src_embeddings"].append(path)
        elif "tgt_embeddings" in base:
            roles["tgt_embeddings"].append(path)
        elif "encoder" in base:
            roles["encoder"].append(path)
        elif "decoder" in base:
            roles["decoder"].append(path)

    keep = []

    if roles["encoder"]:
        keep.append(roles["encoder"][0])
    if roles["decoder"]:
        keep.append(roles["decoder"][0])
    if roles["encoder_wrapper"]:
        keep.append(roles["encoder_wrapper"][0])

    # keep all decoder wrappers, because vocab_size may differ across targets
    keep.extend(roles["decoder_wrapper"])

    if roles["src_embeddings"]:
        keep.append(roles["src_embeddings"][0])
    if roles["tgt_embeddings"]:
        keep.append(roles["tgt_embeddings"][0])

    if roles["frame"]:
        keep.append(roles["frame"][0])

    return keep

def infer_config_from_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, Any]:
    cfg = {}

    keys = list(sd.keys())

    layer_ids = set()
    for k in keys:
        parts = k.split(".")
        if len(parts) >= 2 and parts[0] == "_base_layers":
            try:
                layer_ids.add(int(parts[1]))
            except ValueError:
                pass
    if layer_ids:
        cfg["num_layers"] = max(layer_ids) + 1

    q_w = None
    k_w = None
    v_w = None
    ff_w = None
    rope = None
    out_w = None
    norm = None
    cross_attn = False

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        lk = k.lower()

        if q_w is None and ".to_q.weight" in lk:
            q_w = v
            cfg["_q_key"] = k

        if k_w is None and ".to_k.weight" in lk:
            k_w = v
            cfg["_k_key"] = k

        if v_w is None and ".to_v.weight" in lk:
            v_w = v
            cfg["_v_key"] = k

        if ff_w is None and (
            ".ff." in lk or "ffn" in lk or "feed_forward" in lk or "mlp" in lk
        ) and lk.endswith("weight"):
            ff_w = v
            cfg["_ff_key"] = k

        if rope is None and ("rotary" in lk or "inv_freq" in lk):
            rope = v
            cfg["_rope_key"] = k

        if out_w is None and (
            "to_logits.weight" in lk or "generator" in lk or "lm_head.weight" in lk
        ):
            out_w = v
            cfg["_out_key"] = k

        if norm is None and (
            "post_emb_norm.gamma" in lk
            or lk.endswith("layer_norm.weight")
            or lk.endswith("norm.weight")
            or lk.endswith("post_emb_norm.beta")
        ):
            norm = v
            cfg["_norm_key"] = k

        if "cross_attn" in lk or "encoder_attn" in lk:
            cross_attn = True

    cfg["has_cross_attn"] = cross_attn

    # d_model
    if q_w is not None and q_w.ndim == 2:
        cfg["model_dim"] = int(q_w.shape[1])
        cfg["q_proj_out_dim"] = int(q_w.shape[0])
        cfg["dtype"] = str(q_w.dtype).replace("torch.", "")
    elif norm is not None and norm.ndim == 1:
        cfg["model_dim"] = int(norm.shape[0])
        cfg["dtype"] = str(norm.dtype).replace("torch.", "")

    if k_w is not None and k_w.ndim == 2:
        cfg["k_proj_out_dim"] = int(k_w.shape[0])

    if v_w is not None and v_w.ndim == 2:
        cfg["v_proj_out_dim"] = int(v_w.shape[0])

    # FF dimension
    if ff_w is not None and ff_w.ndim == 2:
        cfg["ff_hidden_dim"] = int(ff_w.shape[0])
        if "model_dim" not in cfg:
            cfg["model_dim"] = int(ff_w.shape[1])

    # IMPORTANT: this is rotary_dim, not necessarily head_dim
    if rope is not None and rope.ndim == 1:
        cfg["rotary_dim"] = int(rope.shape[0]) * 2

    # Wrapper vocab
    if out_w is not None and out_w.ndim == 2:
        cfg["vocab_size"] = int(out_w.shape[0])

    if "model_dim" in cfg and "ff_hidden_dim" in cfg and cfg["model_dim"] > 0:
        cfg["ff_mult"] = cfg["ff_hidden_dim"] / cfg["model_dim"]

    # Heuristic fallback for older x-transformers models:
    # many checkpoints were trained with trained_head_dim = 64.
    # Keep this only as a fallback hint; it is not the same as
    # calculated_head_dim = model_dim / configured_num_heads.
    if "q_proj_out_dim" in cfg:
        qout = cfg["q_proj_out_dim"]
        if qout % 64 == 0:
            cfg["likely_head_dim_old"] = 64
            cfg["likely_num_heads_old"] = qout // 64

    return cfg

def classify_checkpoint_file(path: str, cfg: dict[str, Any]) -> str:
    base = os.path.basename(path).lower()

    if base.endswith("_optim.pt"):
        return "optimizer"

    if "frame" in base:
        return "frame"

    if "encoder_wrapper" in base:
        return "encoder_wrapper"

    if "decoder_wrapper" in base:
        return "decoder_wrapper"

    if "encoder" in base:
        return "encoder"

    if "decoder" in base:
        return "decoder"

    if "vocab_size" in cfg and "num_layers" not in cfg:
        return "wrapper"

    return "other"

from collections import defaultdict, Counter

from collections import defaultdict
from typing import Any

from collections import defaultdict
from typing import Any

# Replace merge_model_configs(...) with this raw version

def merge_model_configs_raw(per_file: list[dict[str, Any]]) -> dict[str, Any]:
    buckets = defaultdict(list)

    for item in per_file:
        role = item["role"]
        cfg = item["cfg"]
        for k, v in cfg.items():
            if not k.startswith("_"):
                buckets[(role, k)].append(v)

    def uniq(vals):
        out = []
        for v in vals:
            if v not in out:
                out.append(v)
        return out

    summary = {
        "global": {},
        "encoder": {},
        "decoder": {},
        "encoder_wrapper": {},
        "decoder_wrapper": {},
        "warnings": [],
    }

    for role in ["encoder", "decoder", "encoder_wrapper", "decoder_wrapper"]:
        role_keys = sorted({
            key
            for (r, key) in buckets.keys()
            if r == role
        })

        for key in role_keys:
            vals = uniq(buckets[(role, key)])
            if len(vals) == 1:
                summary[role][key] = vals[0]
            elif len(vals) > 1:
                summary[role][key] = vals
                summary["warnings"].append(f"ambiguous {role}.{key}: {vals}")

    return summary


def print_model_summary(summary: dict[str, Any], yaml_like: bool = False):
    order = ["global", "core", "encoder", "decoder", "encoder_wrapper", "decoder_wrapper"]

    if yaml_like:
        print("model_summary:")
        for section in order:
            print(f"  {section}:")
            if not summary.get(section):
                print("    {}")
            else:
                for k in sorted(summary[section]):
                    v = summary[section][k]
                    print(f"    {k}: {v}")
        if summary.get("head_hints"):
            print("  head_hints:")
            for k in sorted(summary["head_hints"]):
                v = summary["head_hints"][k]
                print(f"    {k}: {v}")
        if summary.get("warnings"):
            print("  warnings:")
            for w in sorted(summary["warnings"]):
                print(f"    - {w}")
        return

    print("\n## Inferred model summary")
    for section in order:
        print(f"\n### {section}")
        rows = [[k, summary[section][k]] for k in sorted(summary.get(section, {}))]
        if rows:
            print_table(["field", "value"], rows)
        else:
            print("(none)")

    if summary.get("head_hints"):
        print("\n### head_hints")
        rows = [[k, summary["head_hints"][k]] for k in sorted(summary["head_hints"])]
        print_table(["field", "value"], rows)

    if summary.get("warnings"):
        print("\n### warnings")
        for w in sorted(summary["warnings"]):
            print(f"- {w}")

def hoist_shared_keys(summary: dict[str, Any], sections: list[str], target: str, allowed_keys=None) -> dict[str, Any]:
    if target not in summary:
        summary[target] = {}

    existing = [s for s in sections if s in summary and isinstance(summary[s], dict)]
    if len(existing) < 2:
        return summary

    common_keys = set(summary[existing[0]].keys())
    for s in existing[1:]:
        common_keys &= set(summary[s].keys())

    if allowed_keys is not None:
        common_keys &= set(allowed_keys)

    for key in sorted(common_keys):
        vals = [summary[s][key] for s in existing]
        first = vals[0]
        if all(v == first for v in vals[1:]):
            summary[target][key] = first
            for s in existing:
                del summary[s][key]

    return summary

def compress_model_summary(summary: dict[str, Any]) -> dict[str, Any]:
    # Hoist fields shared by all component families
    summary = hoist_shared_keys(
        summary,
        ["encoder", "decoder", "encoder_wrapper", "decoder_wrapper"],
        "global",
        allowed_keys=[
            "dtype",
            "has_cross_attn",
            "model_dim",
            "configured_num_heads",
            "configured_num_heads_source",
            "calculated_head_dim",
            "calculated_head_dim_source",
        ],
    )

    # Hoist head information shared by encoder+decoder only
    summary = hoist_shared_keys(
        summary,
        ["encoder", "decoder"],
        "global",
        allowed_keys=[
            "trained_head_dim",
            "trained_head_dim_source",
        ],
    )

    if "core" not in summary:
        summary["core"] = {}

    summary = hoist_shared_keys(
        summary,
        ["encoder", "decoder"],
        "core",
        allowed_keys=[
            "ff_hidden_dim",
            "ff_mult",
            "q_proj_out_dim",
            "k_proj_out_dim",
            "v_proj_out_dim",
            "rotary_dim",
        ],
    )

    return summary

def summarize_model(files: list[str], args):
    per_file = []
    model_head_hints = {}

    for path in files:
        try:
            obj = load_checkpoint(path, args.unsafe)
        except Exception as e:
            if not args.quiet:
                print(f"Skipping {path}: load_error={type(e).__name__}: {e}")
            continue

        hints = extract_head_hints_from_object(obj)
        for k, v in hints.items():
            model_head_hints[k] = v

        sd = get_state_dict_like(obj)
        if sd is None:
            continue

        cfg = infer_config_from_state_dict(sd)
        role = classify_checkpoint_file(path, cfg)
        if role in ("encoder", "decoder"):
            cfg.update(infer_logical_layer_count(sd, role))
        per_file.append({
            "path": path,
            "role": role,
            "cfg": cfg,
        })

    # 1. Merge raw per-role values
    summary = merge_model_configs_raw(per_file)

    # 2. Resolve heads/head_dim BEFORE compression, while model_dim is still
    #    present in encoder/decoder sections
    for section in ["encoder", "decoder", "encoder_wrapper", "decoder_wrapper"]:
        if section in summary and isinstance(summary[section], dict):
            summary[section] = resolve_head_info(summary[section], model_head_hints)

    # 3. Compress after resolution so shared resolved values get hoisted
    summary = compress_model_summary(summary)
    summary = drop_stale_fallback_hints(summary)
    if model_head_hints:
        summary["head_hints"] = model_head_hints
    summary = add_head_dim_warnings(summary)
    
    print_model_summary(summary, yaml_like=args.yaml_like)

def drop_stale_fallback_hints(summary: dict[str, Any]) -> dict[str, Any]:
    explicit_head_info = False

    for section in ["global", "core", "encoder", "decoder"]:
        sec = summary.get(section, {})
        if not isinstance(sec, dict):
            continue

        if "configured_num_heads_source" in sec:
            explicit_head_info = True
            break
        if "configured_head_dim_source" in sec:
            explicit_head_info = True
            break
        if "calculated_head_dim_source" in sec:
            explicit_head_info = True
            break
        if "trained_head_dim_source" in sec and sec["trained_head_dim_source"] != "old_x_transformers_default":
            explicit_head_info = True
            break

    if explicit_head_info:
        for section in ["global", "core", "encoder", "decoder"]:
            sec = summary.get(section, {})
            if not isinstance(sec, dict):
                continue
            sec.pop("likely_head_dim_old", None)
            sec.pop("likely_num_heads_old", None)

    return summary

def summarize(path: str, args):
    try:
        obj = load_checkpoint(path, args.unsafe)
    except Exception as e:
        msg = str(e)
        print(f"\n=== {path} ===")
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            print("Could not load with weights_only=True.")
            print("This file contains pickled Python objects.")
            print("Use --unsafe only if you trust the checkpoint source.")
            print("\nError:")
            print(msg)
            return
        raise

    sd = get_state_dict_like(obj)

    if args.inspect_patterns:
        print(f"\n=== {path} ===")
        if sd is None:
            print("(No tensor state_dict found; likely metadata/config/optimizer file.)")
            return
        inspect_layer_patterns(sd, top_keys_per_layer=args.pattern_top_keys)
        return

    if args.config_summary or args.one_line:
        if sd is None:
            base = os.path.basename(path)
            if args.one_line:
                print(f"{base}: no_tensor_state_dict")
            else:
                print(f"\n=== {path} ===")
                print("(No tensor state_dict found; likely metadata/config/optimizer file.)")
            return

        cfg = infer_config_from_state_dict(sd)
        base = os.path.basename(path)

        if args.one_line:
            if sd is None:
                print(f"{base}: no_tensor_state_dict")
                return

            cfg = infer_config_from_state_dict(sd)
            if not cfg:
                print(f"{base}: no_config_fields")
                return

            fields = []
            for k in [
                    "num_base_layers",
                    "num_layers",
                    "model_dim",
                    "q_proj_out_dim",
                    "k_proj_out_dim",
                    "v_proj_out_dim",
                    "ff_hidden_dim",
                    "ff_mult",
                    "rotary_dim",
                    "configured_num_heads",
                    "calculated_head_dim",
                    "trained_head_dim",
                    "vocab_size",
                    "has_cross_attn",
                    "dtype",
            ]:
                if k in cfg:
                    fields.append(f"{k}={cfg[k]}")

            if fields:
                print(f"{base}: " + ", ".join(fields))
            else:
                print(f"{base}: no_config_fields")
            return
        
        print_config_summary(path, cfg)
        return

    top_keys = []
    if isinstance(obj, dict):
        top_keys = [str(k) for k in list(obj.keys())[:8]]
    top_keys_s = ", ".join(top_keys) if top_keys else "—"

    n_tensors = 0
    n_params = 0
    n_bytes = 0
    largest = None

    groups = defaultdict(lambda: {"tensors": 0, "params": 0, "bytes": 0})
    tensor_stats = []
    all_rows = []

    if sd is not None:
        for k, v in sd.items():
            if not torch.is_tensor(v):
                continue

            numel = v.numel()
            b = numel * bytes_per_elem(v.dtype)
            dtype = str(v.dtype).replace("torch.", "")
            shape = tuple(v.shape)
            label = label_key(k)

            n_tensors += 1
            n_params += numel
            n_bytes += b

            tensor_stats.append({
                "key": k,
                "label": label,
                "shape": shape,
                "dtype": dtype,
                "params": numel,
                "bytes": b,
            })

            g = group_key(k, args.depth)
            groups[g]["tensors"] += 1
            groups[g]["params"] += numel
            groups[g]["bytes"] += b

            if largest is None or numel > largest["params"]:
                largest = tensor_stats[-1]

            if args.all_keys:
                all_rows.append([
                    k, label, str(shape), dtype, f"{numel:,}", f"{mb(b):.1f}"
                ])

    print(f"\n=== {path} ===")

    if not args.no_summary:
        largest_s = "—"
        if largest:
            largest_s = (
                f"{largest['key']} {largest['shape']} {largest['dtype']}"
            )

        print("\n## Summary")
        print_table(
            [
                "file", "type", "top-level keys", "#tensors",
                "#params", "est_MB", "file_MB", "largest_tensor"
            ],
            [[
                base,
                type(obj).__name__,
                top_keys_s,
                str(n_tensors),
                f"{n_params:,}",
                f"{mb(n_bytes):.1f}",
                f"{file_mb:.1f}",
                largest_s,
            ]],
        )

    if not args.no_modules:
        print(f"\n## Module grouping (depth={args.depth}, top {args.top_mods})")
        if sd is None:
            print("(No tensors found; likely metadata/config.)")
        else:
            rows = []
            for name, st in sorted(
                groups.items(), key=lambda kv: kv[1]["params"], reverse=True
            )[: args.top_mods]:
                rows.append([
                    name,
                    label_key(name),
                    str(st["tensors"]),
                    f"{st['params']:,}",
                    f"{mb(st['bytes']):.1f}",
                ])
            print_table(
                ["module_prefix", "label", "#tensors", "#params", "est_MB"],
                rows,
            )

    if args.top_tensors > 0:
        if sd is None:
            print("(No tensors found in this file.)")
            if isinstance(obj, dict):
                print("Top-level keys:", top_keys_s)
        else:
            top = sorted(
                tensor_stats, key=lambda x: x["params"], reverse=True
            )[: args.top_tensors]
            rows = [[
                t["key"],
                t["label"],
                str(t["shape"]),
                t["dtype"],
                f"{t['params']:,}",
                f"{mb(t['bytes']):.1f}",
            ] for t in top]

            print(f"\n## Top {args.top_tensors} tensors by parameter count")
            print_table(
                ["key", "label", "shape", "dtype", "#params", "est_MB"],
                rows,
            )

    if args.all_keys and sd is not None:
        print("\n## All tensor keys")
        print_table(
            ["key", "label", "shape", "dtype", "#params", "est_MB"],
            all_rows,
        )


    
def main():
    ap = argparse.ArgumentParser(
        description="Inspect PyTorch checkpoint/module files."
    )
    ap.add_argument("files", nargs="+")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--top-mods", type=int, default=30)
    ap.add_argument("--top-tensors", type=int, default=0)
    ap.add_argument("--all-keys", action="store_true")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--no-modules", action="store_true")
    ap.add_argument("--config-summary",action="store_true",
        help="Print only compact config-like fields inferred from tensor shapes.",)
    ap.add_argument("--one-line",action="store_true",
        help="Print exactly one compact summary line per file.",)
    ap.add_argument("--model-summary", action="store_true",
                    help="Merge all given checkpoint files into one inferred model summary.")
    ap.add_argument("--yaml-like", action="store_true",
                    help="Print the model summary in YAML-like form.")
    ap.add_argument("--inspect-patterns", action="store_true",
                    help="Inspect _base_layers.<i> key patterns to understand saved layer structure.",)
    ap.add_argument("--pattern-top-keys", type=int, default=12,
                    help="How many example keys to print per saved layer in --inspect-patterns mode.",)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--unsafe", action="store_true",
        help="Use torch.load(..., weights_only=False). Only for trusted files.",)
    args = ap.parse_args()

    if args.model_summary:
        files = args.files
        files = select_representative_files(args.files)
        if not args.quiet:
            print("Selected representative files:")
            for p in files:
                print("  ", p)
        summarize_model(files, args)
        return

    for p in args.files:
        summarize(p, args)


if __name__ == "__main__":
    main()

