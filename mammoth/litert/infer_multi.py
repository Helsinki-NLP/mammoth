#!/usr/bin/env python3
"""
Greedy autoregressive inference with a multi-task Mammoth .tflite model.

The .tflite must have been produced by convert_multi.py and is accompanied by a
_manifest.json that maps task-ids to (src, tgt) language pairs.

Signature naming convention (from convert_multi.py):
    encode_{src}   — run once on source tokens
    prefill_{tgt}  — run once with BOS to seed the KV cache
    decode_{tgt}   — run per autoregressive step

Usage:
    python mammoth/litert/infer_multi.py \\
        --tflite  mammoth_multi.tflite \\
        --manifest mammoth_multi_manifest.json \\
        --task    fin-swe \\
        --frame   /path/to/_best_frame.pt \\
        --src     "Hello world ."

    # With saved tokenizer directories:
    python mammoth/litert/infer_multi.py \\
        --tflite  mammoth_multi.tflite \\
        --manifest mammoth_multi_manifest.json \\
        --task    fin-swe \\
        --src-tok ./hf_fin_swe/src_tokenizer \\
        --tgt-tok ./hf_fin_swe/tgt_tokenizer \\
        --src     "Hei maailma ."

Requires: ai_edge_litert  (pip install ai-edge-litert)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Reuse tokenizer-loading and buffer helpers from the single-task infer module.
from mammoth.litert.infer import (
    _load_tokenizers_from_frame,
    _load_tokenizers_from_dirs,
    _buf_read,
    _buf_write,
    _causal_mask_prefill,
    _causal_mask_decode,
    translate,
)


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _default_manifest_path(tflite_path: str) -> str:
    """Derive manifest path from .tflite path (mirrors write_manifest in convert_multi.py)."""
    base = tflite_path[:-7] if tflite_path.endswith(".tflite") else tflite_path
    return base + "_manifest.json"


# ── Shape discovery (multi-signature variant) ─────────────────────────────────

def discover_shapes_multi(model, src: str, tgt: str) -> dict:
    """Read enc_max_len, dec_max_len, n_dec_layers, vocab from per-language sigs."""
    enc_name = f"encode_{src}"
    pre_name = f"prefill_{tgt}"
    dec_name = f"decode_{tgt}"

    enc_idx = model.get_signature_index(enc_name)
    pre_idx = model.get_signature_index(pre_name)
    dec_idx = model.get_signature_index(dec_name)

    enc_in = model.create_input_buffers(enc_idx)
    pre_out = model.create_output_buffers(pre_idx)
    dec_in = model.create_input_buffers(dec_idx)

    # encode: inputs are (input_ids [1, enc_max], pad_mask [1, enc_max])
    enc_max = enc_in[0].get_tensor_details()["shape"][1]

    # decode: input[2] = causal_mask (1, 1, 1, dec_max)
    dec_max = dec_in[2].get_tensor_details()["shape"][3]

    # prefill outputs: logits + 4*n_layers KV tensors
    n_out = len(pre_out)
    n_dec_layers = (n_out - 1) // 4

    vocab = pre_out[0].get_tensor_details()["shape"][2]

    return {
        "enc_idx": enc_idx,
        "pre_idx": pre_idx,
        "dec_idx": dec_idx,
        "enc_max": enc_max,
        "dec_max": dec_max,
        "n_dec_layers": n_dec_layers,
        "vocab": vocab,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Greedy translation with a multi-task Mammoth LiteRT .tflite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tflite", required=True,
                        help=".tflite produced by convert_multi.py")
    parser.add_argument("--manifest", default=None,
                        help="_manifest.json (auto-detected from --tflite path if omitted)")
    parser.add_argument("--task", required=True,
                        help="Task id, e.g. fin-swe (must be listed in manifest)")
    parser.add_argument("--src", required=True,
                        help="Source sentence to translate")
    parser.add_argument("--frame", default=None,
                        help="Mammoth _best_frame.pt (for loading tokenizers)")
    parser.add_argument("--src-tok", default=None,
                        help="Path to source tokenizer dir (alternative to --frame)")
    parser.add_argument("--tgt-tok", default=None,
                        help="Path to target tokenizer dir (alternative to --frame)")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU delegate")
    parser.add_argument("--npu", action="store_true",
                        help="Enable NPU delegate — requires an AOT-compiled .tflite "
                             "(see compile_npu.py); use with --gpu for CPU/GPU/NPU "
                             "fallback chain")
    args = parser.parse_args()

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest_path = args.manifest or _default_manifest_path(args.tflite)
    if not os.path.exists(manifest_path):
        parser.error(f"Manifest not found: {manifest_path}\nPass --manifest explicitly.")
    manifest = load_manifest(manifest_path)

    tasks = manifest["tasks"]
    if args.task not in tasks:
        parser.error(
            f"Task {args.task!r} not in manifest. Available: {sorted(tasks.keys())}"
        )
    task_entry = tasks[args.task]
    src_lang = task_entry["src"]
    tgt_lang = task_entry["tgt"]
    print(f"Task: {args.task}  →  encode_{src_lang} / prefill_{tgt_lang} / decode_{tgt_lang}")

    # ── Tokenizers ────────────────────────────────────────────────────────────
    if args.src_tok and args.tgt_tok:
        src_tok, tgt_tok, bos_id, eos_id, pad_id = _load_tokenizers_from_dirs(
            args.src_tok, args.tgt_tok
        )
    elif args.frame:
        src_tok, tgt_tok, bos_id, eos_id, pad_id = _load_tokenizers_from_frame(
            args.frame, args.task
        )
    else:
        parser.error("Provide either --frame or both --src-tok and --tgt-tok")

    print(f"BOS={bos_id}  EOS={eos_id}  PAD={pad_id}")

    # ── Tokenize source ───────────────────────────────────────────────────────
    src_ids = [bos_id] + src_tok.encode(args.src, add_special_tokens=False) + [eos_id]
    print(f"Source: {args.src!r}  →  {src_ids}")

    # ── Load model ────────────────────────────────────────────────────────────
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    hw = HardwareAccelerator.CPU
    if args.gpu:
        hw |= HardwareAccelerator.GPU
    if args.npu:
        hw |= HardwareAccelerator.NPU

    print(f"Loading {args.tflite} …")
    model = CompiledModel.from_file(args.tflite, hw)

    shapes = discover_shapes_multi(model, src_lang, tgt_lang)
    print(
        f"Model: enc_max={shapes['enc_max']}  dec_max={shapes['dec_max']}"
        f"  n_dec_layers={shapes['n_dec_layers']}  vocab={shapes['vocab']}"
    )

    # ── Translate ─────────────────────────────────────────────────────────────
    print("Translating …")
    pred_ids = translate(
        model, src_ids, bos_id, eos_id, pad_id,
        max_new_tokens=args.max_new_tokens,
        shapes=shapes,
    )
    print(f"Output IDs: {pred_ids}")
    print(f"\nTranslation: {tgt_tok.decode(pred_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
