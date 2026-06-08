#!/usr/bin/env python3
"""
Greedy autoregressive inference with a Mammoth multi-signature .tflite model.

Three-step loop:
  1. encode   — run once on the padded source token ids
  2. prefill  — run once with BOS to seed the KV cache
  3. decode   — run until EOS or max_len, one token per step

Usage:
    python mammoth/litert/infer.py \\
        --tflite  mammoth_eng_spa.tflite \\
        --frame   /path/to/model/_best_frame.pt \\
        --src     "Hello world ." \\
        --task    eng-spa

    # GPU backend (on-device NPU):
    python mammoth/litert/infer.py --tflite ... --frame ... --src "..." --gpu

    # If you already ran convert_mammoth_to_hf.py and have saved tokenizers:
    python mammoth/litert/infer.py \\
        --tflite  mammoth_eng_spa.tflite \\
        --src-tok ./hf_eng_spa/src_tokenizer \\
        --tgt-tok ./hf_eng_spa/tgt_tokenizer \\
        --src     "Hello world ."

Requires: ai_edge_litert   (pip install ai-edge-litert)
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# ── Tokenizer loading ─────────────────────────────────────────────────────────

def _load_tokenizers_from_frame(frame_path: str, task: str):
    """Load src/tgt tokenizers from a Mammoth _best_frame.pt."""
    import torch
    from transformers import PreTrainedTokenizerFast

    try:
        frame = torch.load(frame_path, map_location="cpu", weights_only=True)
    except Exception:
        frame = torch.load(frame_path, map_location="cpu", weights_only=False)

    vocab = frame["vocab"]
    src, _, tgt = task.partition("-")

    src_vocab = vocab.get(("src", src)) or next(
        v for k, v in vocab.items() if k[0] == "src"
    )
    tgt_vocab = vocab.get(("tgt", tgt)) or next(
        v for k, v in vocab.items() if k[0] == "tgt"
    )

    def _make_tok(vocab_obj, add_bos_eos=False):
        sp = vocab_obj.specials
        rev = {v: k for k, v in sp.items()}
        bos_id = sp.get("<s>", 2)
        eos_id = sp.get("</s>", 0)
        tok = PreTrainedTokenizerFast(
            tokenizer_object=vocab_obj.tokenizer,
            bos_token=rev.get(bos_id, "<s>"),
            eos_token=rev.get(eos_id, "</s>"),
            pad_token=rev.get(sp.get("<blank>", 1), "<blank>"),
        )
        if add_bos_eos:
            from tokenizers.processors import TemplateProcessing
            bos_str = rev.get(bos_id, "<s>")
            eos_str = rev.get(eos_id, "</s>")
            tok._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos_str}:0 $A:0 {eos_str}:0",
                special_tokens=[(bos_str, bos_id), (eos_str, eos_id)],
            )
        return tok, sp

    src_tok, src_sp = _make_tok(src_vocab, add_bos_eos=True)
    tgt_tok, tgt_sp = _make_tok(tgt_vocab, add_bos_eos=False)

    bos_id = tgt_sp.get("<s>", 2)
    eos_id = tgt_sp.get("</s>", 0)
    pad_id = src_sp.get("<blank>", 1)
    return src_tok, tgt_tok, bos_id, eos_id, pad_id


def _load_tokenizers_from_dirs(src_tok_dir: str, tgt_tok_dir: str):
    """Load tokenizers from a directory or a single tokenizer.json file."""
    from transformers import PreTrainedTokenizerFast

    def _load_one(path: str) -> PreTrainedTokenizerFast:
        if os.path.isfile(path):
            return PreTrainedTokenizerFast(tokenizer_file=path)
        return PreTrainedTokenizerFast.from_pretrained(path)

    src_tok = _load_one(src_tok_dir)
    tgt_tok = _load_one(tgt_tok_dir)
    bos_id = tgt_tok.bos_token_id or 2
    eos_id = tgt_tok.eos_token_id or 0
    pad_id = src_tok.pad_token_id or 1
    return src_tok, tgt_tok, bos_id, eos_id, pad_id


# ── Buffer helpers ────────────────────────────────────────────────────────────

def _buf_read(buf) -> np.ndarray:
    details = buf.get_tensor_details()
    n = math.prod(details["shape"])
    return buf.read(n, details["dtype"]).reshape(details["shape"])


def _buf_write(buf, arr: np.ndarray):
    buf.write(arr.reshape(-1))


# ── Mask builders ─────────────────────────────────────────────────────────────

def _causal_mask_prefill(dec_max: int) -> np.ndarray:
    """Lower-triangular additive causal mask for the prefill step."""
    m = np.full((1, 1, dec_max, dec_max), float("-inf"), dtype=np.float32)
    for i in range(dec_max):
        m[0, 0, i, : i + 1] = 0.0
    return m


def _causal_mask_decode(step: int, dec_max: int) -> np.ndarray:
    """Single-row causal mask for the decode step at position `step`."""
    m = np.full((1, 1, 1, dec_max), float("-inf"), dtype=np.float32)
    m[0, 0, 0, : step + 1] = 0.0
    return m


def _cross_mask_prefill(dec_max: int, enc_max: int) -> np.ndarray:
    return np.zeros((1, 1, dec_max, enc_max), dtype=np.float32)


def _cross_mask_decode(enc_max: int) -> np.ndarray:
    return np.zeros((1, 1, 1, enc_max), dtype=np.float32)


# ── Model shape discovery ─────────────────────────────────────────────────────

def _discover_shapes(model) -> dict:
    """Read enc_max_len, dec_max_len, n_dec_layers from the model signatures."""
    enc_idx = model.get_signature_index("encode")
    pre_idx = model.get_signature_index("prefill")
    dec_idx = model.get_signature_index("decode")

    enc_in = model.create_input_buffers(enc_idx)
    pre_out = model.create_output_buffers(pre_idx)
    dec_in = model.create_input_buffers(dec_idx)

    # enc_in[0] = input_ids (1, enc_max), enc_in[1] = pad_mask (1, enc_max)
    enc_max = enc_in[0].get_tensor_details()["shape"][1]

    # dec_in[1] = input_ids (1, 1), dec_in[2] = causal_mask (1,1,1,dec_max)
    dec_max = dec_in[2].get_tensor_details()["shape"][3]

    # pre_out: logits + 4N KV tensors; logits shape (1, dec_max, vocab)
    # self-KV outputs: 2N tensors at indices 1..2N
    # cross-KV outputs: 2N tensors at indices 2N+1..4N
    n_out = len(pre_out)   # 1 + 4*n_layers
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


# ── Inference loop ────────────────────────────────────────────────────────────

def translate(
    model,
    src_ids: list[int],
    bos_id: int,
    eos_id: int,
    pad_id: int,
    max_new_tokens: int = 50,
    shapes: dict | None = None,
) -> list[int]:
    """
    Full encode → prefill → decode greedy decode loop.

    Args:
        model        : CompiledModel loaded from the .tflite
        src_ids      : source token IDs (including BOS/EOS from the tokenizer)
        bos_id       : target BOS token ID (decoder start token)
        eos_id       : target EOS token ID (stop condition)
        pad_id       : source PAD token ID (used to fill up to enc_max_len)
        max_new_tokens: stop after this many generated tokens

    Returns:
        list of predicted target token IDs (not including BOS, stops at EOS)
    """
    if shapes is None:
        shapes = _discover_shapes(model)

    enc_idx = shapes["enc_idx"]
    pre_idx = shapes["pre_idx"]
    dec_idx = shapes["dec_idx"]
    enc_max = shapes["enc_max"]
    dec_max = shapes["dec_max"]
    n = shapes["n_dec_layers"]

    # ── 1. Encode ──────────────────────────────────────────────────────────
    enc_in_bufs = model.create_input_buffers(enc_idx)
    enc_out_bufs = model.create_output_buffers(enc_idx)

    # Pad / truncate source to enc_max_len
    src_ids_padded = (list(src_ids) + [pad_id] * enc_max)[:enc_max]
    pad_mask = np.where(
        np.array(src_ids_padded) == pad_id, float("-inf"), 0.0
    ).astype(np.float32).reshape(1, enc_max)

    # Cross-attention mask: same padding positions as the encoder input mask.
    # Shape (1,1,1,enc_max) for decode; broadcast to (1,1,dec_max,enc_max) for prefill.
    cross_mask_1d = pad_mask.reshape(1, 1, 1, enc_max)
    cross_mask_prefill = np.broadcast_to(
        cross_mask_1d, (1, 1, dec_max, enc_max)
    ).copy()

    _buf_write(enc_in_bufs[0], np.array(src_ids_padded, dtype=np.int32).reshape(1, enc_max))
    _buf_write(enc_in_bufs[1], pad_mask)

    model.run_by_index(enc_idx, enc_in_bufs, enc_out_bufs)
    # enc_out_bufs[0] = encoder_hidden_states (1, enc_max, model_dim)

    # ── 2. Prefill with BOS ────────────────────────────────────────────────
    pre_in_bufs = model.create_input_buffers(pre_idx)
    pre_out_bufs = model.create_output_buffers(pre_idx)

    # Zero-copy: share encoder_hidden_states buffer
    pre_in_bufs[0] = enc_out_bufs[0]

    # Decoder input: BOS at position 0, PAD elsewhere
    dec_input = np.full((1, dec_max), pad_id, dtype=np.int32)
    dec_input[0, 0] = bos_id
    _buf_write(pre_in_bufs[1], dec_input)
    _buf_write(pre_in_bufs[2], _causal_mask_prefill(dec_max))
    _buf_write(pre_in_bufs[3], cross_mask_prefill)
    # Placeholders for KV inputs (prefill ignores values, only shapes matter)
    for i in range(4, len(pre_in_bufs)):
        details = pre_in_bufs[i].get_tensor_details()
        n_elems = math.prod(details["shape"])
        pre_in_bufs[i].write(np.zeros(n_elems, dtype=details["dtype"]))

    model.run_by_index(pre_idx, pre_in_bufs, pre_out_bufs)
    # pre_out_bufs[0]         = logits (1, dec_max, vocab)
    # pre_out_bufs[1..2N]     = self_k/v per layer  (1, H, dec_max, d)
    # pre_out_bufs[2N+1..4N]  = cross_k/v per layer (1, H, enc_max, d)

    # First predicted token (position 0 logits)
    first_logits = _buf_read(pre_out_bufs[0])[0, 0, :]
    predicted_ids: list[int] = []
    tok = int(np.argmax(first_logits))
    if tok == eos_id:
        return predicted_ids
    predicted_ids.append(tok)

    # ── 3. Decode loop ─────────────────────────────────────────────────────
    dec_in_bufs = model.create_input_buffers(dec_idx)
    dec_out_bufs = model.create_output_buffers(dec_idx)

    # Zero-copy: encoder_hidden_states stays the same throughout
    dec_in_bufs[0] = enc_out_bufs[0]

    # Cross-KV is fixed after prefill; point decode inputs at prefill outputs
    # dec_in_bufs layout: [enc_hid, ids, causal_mask, cross_mask, step_idx,
    #                       self_k0, self_v0, ..., cross_k0, cross_v0, ...]
    for i in range(2 * n):
        dec_in_bufs[5 + 2 * n + i] = pre_out_bufs[1 + 2 * n + i]

    # Initially point self-KV inputs at prefill self-KV outputs
    for i in range(2 * n):
        dec_in_bufs[5 + i] = pre_out_bufs[1 + i]

    cross_mask_np = cross_mask_1d  # encoder-padding-aware mask, shape (1,1,1,enc_max)

    for step in range(1, min(dec_max, max_new_tokens + 1)):
        current_token = predicted_ids[-1]

        _buf_write(dec_in_bufs[1],
                   np.array([[current_token]], dtype=np.int32))
        _buf_write(dec_in_bufs[2],
                   _causal_mask_decode(step, dec_max))
        _buf_write(dec_in_bufs[3], cross_mask_np)
        _buf_write(dec_in_bufs[4],
                   np.array(step, dtype=np.int32))

        model.run_by_index(dec_idx, dec_in_bufs, dec_out_bufs)
        # dec_out_bufs[0]      = logits (1, 1, vocab)
        # dec_out_bufs[1..2N]  = updated self_k/v

        logits = _buf_read(dec_out_bufs[0])[0, 0, :]
        tok = int(np.argmax(logits))
        if tok == eos_id:
            break
        predicted_ids.append(tok)

        # Swap self-KV: outputs of this step become inputs for the next
        for i in range(2 * n):
            dec_in_bufs[5 + i], dec_out_bufs[1 + i] = (
                dec_out_bufs[1 + i],
                dec_in_bufs[5 + i],
            )

    return predicted_ids


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Greedy translation with a Mammoth LiteRT .tflite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--tflite", required=True,
                        help=".tflite file produced by convert.py")
    parser.add_argument("--src", required=True,
                        help="Source sentence to translate (already tokenized text)")
    parser.add_argument("--task", default="eng-spa",
                        help="Task name (src-tgt), used for loading from --frame")
    parser.add_argument("--frame", default=None,
                        help="Mammoth _best_frame.pt (used to load tokenizers)")
    parser.add_argument("--src-tok", default=None,
                        help="Path to src_tokenizer dir (alternative to --frame)")
    parser.add_argument("--tgt-tok", default=None,
                        help="Path to tgt_tokenizer dir (alternative to --frame)")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum number of output tokens to generate")
    parser.add_argument("--gpu", action="store_true",
                        help="Enable GPU delegate (in addition to CPU)")
    args = parser.parse_args()

    # Tokenizers
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

    # Tokenize source.  The tokenizer.json format doesn't record bos/eos tokens,
    # so add_special_tokens=True is a no-op; wrap manually with <s>…</s> to match
    # the training format (both tokenizers share the same special-token IDs).
    src_ids = [bos_id] + src_tok.encode(args.src, add_special_tokens=False) + [eos_id]
    print(f"Source: {args.src!r}  →  {src_ids}")

    # Load model
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    hw = HardwareAccelerator.CPU
    if args.gpu:
        hw |= HardwareAccelerator.GPU

    print(f"Loading {args.tflite} …")
    model = CompiledModel.from_file(args.tflite, hw)

    # Discover shapes from signature metadata
    shapes = _discover_shapes(model)
    print(
        f"Model: enc_max={shapes['enc_max']}  dec_max={shapes['dec_max']}  "
        f"n_dec_layers={shapes['n_dec_layers']}  vocab={shapes['vocab']}"
    )

    # Run inference
    print("Translating …")
    pred_ids = translate(
        model, src_ids, bos_id, eos_id, pad_id,
        max_new_tokens=args.max_new_tokens,
        shapes=shapes,
    )
    print(f"Output IDs: {pred_ids}")

    translation = tgt_tok.decode(pred_ids, skip_special_tokens=True)
    print(f"\nTranslation: {translation}")


if __name__ == "__main__":
    main()
