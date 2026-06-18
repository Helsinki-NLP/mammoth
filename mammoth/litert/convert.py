#!/usr/bin/env python3
"""
Convert a Mammoth checkpoint to a multi-signature LiteRT (.tflite) model.

Usage (single task, best checkpoint):
    python mammoth/litert/convert.py \\
        --checkpoint-dir /path/to/model \\
        --task eng-spa \\
        --output mammoth_nmt.tflite

Options:
    --checkpoint-dir DIR   Mammoth checkpoint directory (contains _best_frame.pt etc.)
    --task SRC-TGT         Task to convert, e.g. eng-spa (default: first task found)
    --step N               Load step-N checkpoint instead of best
    --enc-max-len N        Encoder sequence length baked into the graph (default: 128)
    --dec-max-len N        Decoder KV-cache length baked into the graph (default: 64)
    --output FILE          Output path for the .tflite file (default: mammoth_nmt.tflite)
    --no-quant             Skip quantization (export float32 weights)
    --export-only          Stop after torch.export (skip litert_torch lowering)
    --dtype {fp32,bf16}    Cast model weights before export (default: fp32)

On macOS (where litert_torch is unavailable) --export-only is set automatically.
The exported program is saved as <output>.ep alongside the .tflite path.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
    assemble_state_dict,
    config_from_opts,
    resolve_prefix,
    _discover_tasks,
    _task_xcoder_ids,
    _infer_xcoder_ids_from_shards,
    _load,
)
from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import (
    MammothForConditionalGeneration,
)
from mammoth.litert.wrapper import (
    MammothLiteRTEncoder,
    MammothLiteRTPrefill,
    MammothLiteRTDecode,
    make_encoder_sample_inputs,
    make_prefill_sample_inputs,
    make_decode_sample_inputs,
    patch_rms_norms,
)


# ── Detect litert_torch availability ──────────────────────────────────────────

try:
    import litert_torch  # noqa: F401
    LITERT_AVAILABLE = True
except ImportError:
    LITERT_AVAILABLE = False


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_hf_model(
    ckpt_dir: str,
    task: str | None,
    step: int | None,
) -> tuple[MammothForConditionalGeneration, MammothConfig, str, str]:
    """Load a Mammoth checkpoint and return (hf_model, config, src, tgt)."""
    prefix = resolve_prefix(ckpt_dir, step)
    frame = _load(os.path.join(ckpt_dir, f"{prefix}_frame.pt"))
    opts = frame["opts"]
    vocab = frame["vocab"]

    # Resolve task
    if task is not None:
        src, _, tgt = task.partition("-")
    else:
        pairs = _discover_tasks(opts)
        if not pairs:
            raise ValueError("No tasks found in checkpoint; pass --task SRC-TGT")
        src, tgt = pairs[0]
        print(f"No --task specified; using first task: {src}-{tgt}")

    # Vocab sizes
    src_vocab_obj = vocab.get(("src", src)) or next(
        v for k, v in vocab.items() if k[0] == "src"
    )
    tgt_vocab_obj = vocab.get(("tgt", tgt)) or next(
        v for k, v in vocab.items() if k[0] == "tgt"
    )
    src_vocab_size = len(src_vocab_obj)
    tgt_vocab_size = len(tgt_vocab_obj)
    print(f"Vocab: src={src_vocab_size} tgt={tgt_vocab_size}")

    # Special token IDs
    from mammoth.constants import DefaultTokens
    sp = src_vocab_obj.specials
    bos_id = sp.get(DefaultTokens.BOS, 2)
    eos_id = sp.get(DefaultTokens.EOS, 0)
    pad_id = sp.get(DefaultTokens.PAD, 1)

    # Sharing groups
    try:
        encoder_id, decoder_id = _task_xcoder_ids(opts, src, tgt)
    except ValueError:
        encoder_id, decoder_id = _infer_xcoder_ids_from_shards(
            ckpt_dir, prefix, src, tgt
        )
    print(f"Components: encoder={encoder_id}  decoder={decoder_id}")

    config = config_from_opts(
        opts,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        encoder_sharing_groups=encoder_id,
        decoder_sharing_groups=decoder_id,
    )
    print(
        f"Config: dim={config.model_dim}  heads={config.heads}  "
        f"enc={config.enc_layers}  dec={config.dec_layers}  "
        f"ff_mult={config.ff_mult}  swiglu={config.ff_swiglu}"
    )

    print("Assembling state dict from shards …")
    sd = assemble_state_dict(
        ckpt_dir, prefix, src, tgt, encoder_id, decoder_id
    )
    print(f"  {len(sd)} tensors loaded")

    print("Building model and loading weights …")
    model = MammothForConditionalGeneration(config)
    missing = [k for k in model.state_dict() if k not in sd]
    unexpected = [k for k in sd if k not in model.state_dict()]
    if missing:
        print(f"  [WARN] missing keys: {missing}")
    if unexpected:
        print(f"  [WARN] unexpected keys: {unexpected}")
    model.load_state_dict(sd, strict=not (missing or unexpected))
    print("  Weights loaded.")
    return model, config, src, tgt


# ── Wrapper construction ──────────────────────────────────────────────────────

def build_wrappers(
    model: MammothForConditionalGeneration,
    enc_max_len: int,
    dec_max_len: int,
    dtype: torch.dtype,
) -> tuple[MammothLiteRTEncoder, MammothLiteRTPrefill, MammothLiteRTDecode]:
    model = model.to(dtype).eval()
    encoder = MammothLiteRTEncoder(model, enc_max_len).eval()
    prefill = MammothLiteRTPrefill(model, enc_max_len, dec_max_len).eval()
    decode = MammothLiteRTDecode(model, enc_max_len, dec_max_len).eval()
    return encoder, prefill, decode


# ── Sample inputs ─────────────────────────────────────────────────────────────

def build_sample_inputs(
    encoder: MammothLiteRTEncoder,
    prefill: MammothLiteRTPrefill,
    config: MammothConfig,
    enc_max_len: int,
    dec_max_len: int,
) -> tuple[tuple, tuple, tuple]:
    with torch.no_grad():
        enc_inputs = make_encoder_sample_inputs(enc_max_len, config.src_vocab_size)
        enc_out = encoder(*enc_inputs)
        prefill_inputs = make_prefill_sample_inputs(
            enc_out, enc_max_len, dec_max_len, config
        )
        prefill_out = prefill(*prefill_inputs)
        decode_inputs = make_decode_sample_inputs(
            enc_out, prefill_out, enc_max_len, dec_max_len, config
        )
    return enc_inputs, prefill_inputs, decode_inputs


# ── torch.export ──────────────────────────────────────────────────────────────

def export_wrappers(
    encoder, prefill, decode,
    enc_inputs, prefill_inputs, decode_inputs,
    output_path: str,
) -> None:
    print("\n── torch.export ─────────────────────────────────────────────────")
    print("  Exporting encoder …")
    ep_enc = torch.export.export(encoder, enc_inputs)
    print("  Exporting prefill …")
    ep_pre = torch.export.export(prefill, prefill_inputs)
    print("  Exporting decode …")
    ep_dec = torch.export.export(decode, decode_inputs)
    print("  All three signatures exported successfully.")

    ep_path = output_path + ".ep"
    torch.export.save(ep_enc, ep_path)
    print(f"  Encoder exported program saved → {ep_path}")

    return ep_enc, ep_pre, ep_dec


# ── litert_torch conversion ───────────────────────────────────────────────────

def convert_to_litert(
    encoder, prefill, decode,
    enc_inputs, prefill_inputs, decode_inputs,
    output_path: str,
    quant: bool,
) -> None:
    import litert_torch
    from litert_torch.generative.quantize import quant_attrs, quant_recipe
    from litert_torch.quantize import quant_config

    print("\n── litert_torch conversion ──────────────────────────────────────")

    q_config = None
    if quant:
        recipe = quant_recipe.LayerQuantRecipe(
            activation_dtype=quant_attrs.Dtype.FP32,
            weight_dtype=quant_attrs.Dtype.INT8,
            mode=quant_attrs.Mode.DYNAMIC_RANGE,
            algorithm=quant_attrs.Algorithm.MIN_MAX,
            granularity=quant_attrs.Granularity.CHANNELWISE,
        )
        q_config = quant_config.QuantConfig(
            generative_recipe=quant_recipe.GenerativeQuantRecipe(default=recipe)
        )
        print("  Quantization: dynamic int8 (CHANNELWISE)")
    else:
        print("  Quantization: disabled (--no-quant)")

    edge_model = (
        litert_torch.signature("encode", encoder, enc_inputs)
        .signature("prefill", prefill, prefill_inputs)
        .signature("decode", decode, decode_inputs)
        .convert(enable_x64=False, quant_config=q_config)
    )
    edge_model.export(output_path)
    print(f"  Saved → {output_path}")


# ── Quick sanity check ────────────────────────────────────────────────────────

def sanity_check(
    encoder, prefill, decode,
    enc_inputs, prefill_inputs, decode_inputs,
) -> None:
    print("\n── Sanity check (forward passes) ────────────────────────────────")
    with torch.no_grad():
        enc_out = encoder(*enc_inputs)
        print(f"  encoder output: {enc_out[0].shape}  "
              f"norm={enc_out[0].norm():.4f}")

        pre_out = prefill(*prefill_inputs)
        logits_pre = pre_out[0]
        print(f"  prefill logits: {logits_pre.shape}  "
              f"argmax(pos=0)={logits_pre[0, 0].argmax().item()}")

        dec_out = decode(*decode_inputs)
        logits_dec = dec_out[0]
        print(f"  decode  logits: {logits_dec.shape}  "
              f"argmax={logits_dec[0, 0].argmax().item()}")
    print("  All forward passes OK.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert Mammoth checkpoint → LiteRT .tflite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory containing Mammoth checkpoint shards")
    parser.add_argument("--task", default=None,
                        help="Task to convert, e.g. eng-spa")
    parser.add_argument("--step", type=int, default=None,
                        help="Load specific step instead of best checkpoint")
    parser.add_argument("--enc-max-len", type=int, default=128,
                        help="Encoder sequence length baked into the graph")
    parser.add_argument("--dec-max-len", type=int, default=64,
                        help="Decoder KV-cache length baked into the graph")
    parser.add_argument("--output", default="mammoth_nmt.tflite",
                        help="Output .tflite path")
    parser.add_argument("--no-quant", action="store_true",
                        help="Disable dynamic int8 quantization")
    parser.add_argument("--export-only", action="store_true",
                        help="Stop after torch.export (skip litert_torch lowering)")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32",
                        help="Cast model weights before export")
    parser.add_argument("--patch-norms", action="store_true",
                        help=(
                            "Replace nn.RMSNorm with HLFB composite ops "
                            "(odml.rms_norm) before export — required for "
                            "GPU/NPU delegate norm fusion"
                        ))
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    # litert_torch is Linux-only; force export-only on macOS/Windows
    if not LITERT_AVAILABLE and not args.export_only:
        print("[INFO] litert_torch not available on this platform "
              "— running in --export-only mode.")
        args.export_only = True

    # 1. Load checkpoint
    model, config, src, tgt = load_hf_model(
        args.checkpoint_dir, args.task, args.step
    )
    print(f"\nTask: {src} → {tgt}")
    print(f"Shapes: enc_max_len={args.enc_max_len}  dec_max_len={args.dec_max_len}")

    # 2. Build wrappers
    print("\n── Building LiteRT wrappers ─────────────────────────────────────")
    encoder, prefill, decode = build_wrappers(
        model, args.enc_max_len, args.dec_max_len, dtype
    )
    if args.patch_norms:
        print("  Patching RMSNorm → HLFB composite (odml.rms_norm) …")
        for m in (encoder, prefill, decode):
            patch_rms_norms(m)
    n_dec = sum(config.dec_layers)
    print(f"  Encoder stacks: {len(encoder.stacks)}  "
          f"(layers={config.enc_layers})")
    print(f"  Decoder layers: {n_dec}  (stacks={config.dec_layers})")
    print(f"  KV shapes: "
          f"self ({config.heads}h × {args.dec_max_len} × {config.model_dim // config.heads}d), "
          f"cross ({config.heads}h × {args.enc_max_len} × {config.model_dim // config.heads}d)")

    # 3. Sample inputs
    enc_inputs, prefill_inputs, decode_inputs = build_sample_inputs(
        encoder, prefill, config, args.enc_max_len, args.dec_max_len
    )
    print(f"  Prefill input count : {len(prefill_inputs)}  "
          f"(enc_hidden + ids + 2 masks + {4 * n_dec} KV placeholders)")
    print(f"  Decode  input count : {len(decode_inputs)}  "
          f"(enc_hidden + id + 2 masks + step_idx + {4 * n_dec} KV tensors)")

    # 4. Sanity check
    sanity_check(encoder, prefill, decode, enc_inputs, prefill_inputs, decode_inputs)

    # 5. torch.export
    export_wrappers(
        encoder, prefill, decode,
        enc_inputs, prefill_inputs, decode_inputs,
        args.output,
    )

    if args.export_only:
        print(f"\nDone (export-only).  Re-run on Linux with litert_torch to get {args.output}")
        return

    # 6. LiteRT conversion
    convert_to_litert(
        encoder, prefill, decode,
        enc_inputs, prefill_inputs, decode_inputs,
        args.output,
        quant=not args.no_quant,
    )
    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
