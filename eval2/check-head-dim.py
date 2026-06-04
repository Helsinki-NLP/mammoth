#!/usr/bin/env python3
"""
check-head-dim.py

Inspect Mammoth / Transformer checkpoint files and infer the attention head_dim
used when the model was trained.

For attention projection weights such as:

    *.to_q.weight

the output dimension is usually:

    inner_dim = heads * head_dim

So:

    head_dim = inner_dim / heads

If old checkpoints used hard-coded head_dim=64, this script helps determine
whether modern Mammoth config should explicitly set:

    head_dim: 64

Use:
python infer-head-dim.py /path/to/model_dir --heads 12
python infer-head-dim.py /path/to/model_dir --yaml /path/to/train.yaml
python infer-head-dim.py --heads 12 \
  model_best_encoder_0_shared.pt \
  model_best_decoder_0_eng.pt \
  model_best_decoder_0_fin.pt
python infer-head-dim.py --yaml train.yaml \
  model_best_encoder_0_shared.pt \
  model_best_decoder_0_eng.pt
"""

import argparse
import os
import re
from collections import defaultdict

import torch


def expand_inputs(paths):
    out = []
    wanted = (
        "model_best_encoder_0_shared.pt",
        "model_best_decoder_0_eng.pt",
        "model_best_decoder_0_fin.pt",
    )

    for p in paths:
        if os.path.isdir(p):
            for name in wanted:
                q = os.path.join(p, name)
                if os.path.exists(q):
                    out.append(q)

            # fallback: include any base encoder/decoder module files
            if not out:
                for name in sorted(os.listdir(p)):
                    if (
                        name.endswith(".pt")
                        and (
                            "encoder_0_shared" in name
                            or "decoder_0_" in name
                        )
                        and "optim" not in name
                        and "task" not in name
                        and "embeddings" not in name
                        and "frame" not in name
                    ):
                        out.append(os.path.join(p, name))
        else:
            out.append(p)

    return out

def load_yaml(path):
    try:
        import yaml
    except ImportError:
        return {}

    if not path:
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def find_heads_in_yaml(obj):
    found = []

    def walk(x, path=""):
        if isinstance(x, dict):
            for k, v in x.items():
                p = f"{path}.{k}" if path else str(k)
                if k in {"heads", "num_heads", "n_heads"}:
                    found.append((p, v))
                walk(v, p)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)
    return found


def load_pt(path, unsafe=False):
    if unsafe:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu")


def state_dict_like(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if isinstance(obj, dict) and any(torch.is_tensor(v) for v in obj.values()):
        return obj
    return None


def attention_group(key):
    """
    Convert e.g.
      _base_layers.0.1.to_q.weight
    to
      _base_layers.0.1
    """
    return re.sub(r"\.to_[qkv]\.weight$", "", key)


def inspect_file(path, heads):
    obj = load_pt(path)
    sd = state_dict_like(obj)
    if sd is None:
        return []

    groups = defaultdict(dict)

    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if k.endswith(".to_q.weight") or k.endswith(".to_k.weight") or k.endswith(".to_v.weight"):
            g = attention_group(k)
            proj = k.split(".")[-2]  # to_q, to_k, to_v
            groups[g][proj] = tuple(v.shape)

    rows = []
    for g, projs in sorted(groups.items()):
        q_shape = projs.get("to_q")
        k_shape = projs.get("to_k")
        v_shape = projs.get("to_v")

        if q_shape is None:
            continue

        inner_dim = q_shape[0]
        input_dim = q_shape[1] if len(q_shape) == 2 else None

        inferred = None
        status = "unknown"
        if heads:
            if inner_dim % heads == 0:
                inferred = inner_dim // heads
                status = "ok"
            else:
                status = "not divisible by heads"

        rows.append({
            "file": os.path.basename(path),
            "attn_block": g,
            "q_shape": str(q_shape),
            "k_shape": str(k_shape) if k_shape else "-",
            "v_shape": str(v_shape) if v_shape else "-",
            "model_dim_in": input_dim,
            "inner_dim": inner_dim,
            "heads": heads if heads else "-",
            "head_dim": inferred if inferred else "-",
            "status": status,
        })

    return rows


def print_table(rows):
    if not rows:
        print("No attention projection tensors found.")
        return

    headers = [
        "file",
        "attn_block",
        "q_shape",
        "inner_dim",
        "heads",
        "head_dim",
        "status",
    ]

    widths = {h: len(h) for h in headers}
    for r in rows:
        for h in headers:
            widths[h] = max(widths[h], len(str(r[h])))

    def line(vals):
        return " | ".join(str(vals[h]).ljust(widths[h]) for h in headers)

    print(line({h: h for h in headers}))
    print("-+-".join("-" * widths[h] for h in headers))

    for r in rows:
        print(line(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Checkpoint files or model directories")
    ap.add_argument("--heads", type=int, default=None, help="Number of attention heads")
    ap.add_argument("--yaml", default=None, help="Optional Mammoth train.yaml")
    args = ap.parse_args()

    yaml_obj = load_yaml(args.yaml)
    yaml_heads = find_heads_in_yaml(yaml_obj)

    heads = args.heads
    if heads is None and yaml_heads:
        # Use the first scalar-looking heads value found.
        for _, v in yaml_heads:
            if isinstance(v, int):
                heads = v
                break

    if heads is None:
        print("WARNING: no --heads provided and no integer heads found in YAML.")
        print("         I can show inner_dim, but cannot uniquely infer head_dim.")
        print()

    if yaml_heads:
        print("Heads values found in YAML:")
        for p, v in yaml_heads:
            print(f"  {p}: {v}")
        print()

    rows = []
    files = expand_inputs(args.files)
    print("Inspecting files:")
    for f in files:
        print(f"  {f}")
    print()
    for p in files:
        rows.extend(inspect_file(p, heads))

    print_table(rows)

    if heads:
        unique = sorted({r["head_dim"] for r in rows if r["head_dim"] != "-"})
        print()
        print("Inferred head_dim values:", unique)

        if len(unique) == 1:
            print(f"Suggested override: head_dim: {unique[0]}")
        elif len(unique) > 1:
            print("Multiple head_dim values found; inspect per component before overriding.")


if __name__ == "__main__":
    main()

