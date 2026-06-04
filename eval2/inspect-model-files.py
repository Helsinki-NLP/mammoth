#!/usr/bin/env python3
"""
inspect-model-files.py

Utility for inspecting PyTorch checkpoint/module files, especially modular
Mammoth / Transformer NMT checkpoints.

The script summarizes what is stored in each .pt file:
  - top-level object type and keys
  - number of tensors and parameters
  - estimated tensor memory size
  - largest tensors
  - grouped module prefixes
  - optional full tensor key listing

It is useful for understanding how model components are split across files,
for example:
  - source/target embeddings
  - shared encoder stack
  - language-specific decoder stack
  - task-specific output/logits heads
  - metadata/frame files
  - optimizer/training-state files

Common usage:

  python inspect-model-files.py model_best_decoder_0_eng.pt

  python inspect-model-files.py --top-tensors 20 model_best_decoder_0_eng.pt

  python inspect-model-files.py --depth 4 --top-mods 80 model_best_encoder_0_shared.pt

  python inspect-model-files.py --no-summary --no-modules --top-tensors 20 *.pt

  python inspect-model-files.py --unsafe model_best_frame.pt

Safety note:
  By default, torch.load is used in safe weights-only mode when available.
  Use --unsafe only for trusted checkpoint files, because it may unpickle
  arbitrary Python objects.

The script is intended for inspection/debugging only. It does not modify
checkpoint files.
"""
```

import argparse
import os
from collections import defaultdict
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


def load_checkpoint(path: str, unsafe: bool):
    if unsafe:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu")


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
    base = os.path.basename(path)
    file_mb = os.path.getsize(path) / (1024 * 1024)

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
    ap.add_argument(
        "--unsafe",
        action="store_true",
        help="Use torch.load(..., weights_only=False). Only for trusted files.",
    )
    args = ap.parse_args()

    for p in args.files:
        summarize(p, args)


if __name__ == "__main__":
    main()

