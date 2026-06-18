#!/usr/bin/env python3
"""
Convert a multi-task Mammoth checkpoint to a single multi-signature LiteRT .tflite.

The script discovers all source/target languages and their weight-sharing structure
directly from opts.tasks — no assumptions about language set, stack count, or
sharing group names.

Signature layout (one per unique language, not per task-pair):
    encode_{src}   × n_unique_src   — encoder with source-language weights
    prefill_{tgt}  × n_unique_tgt   — decoder prefill with target-language weights
    decode_{tgt}   × n_unique_tgt   — decoder single-step with target-language weights

At inference time, task "fin-swe" → run encode_fin → prefill_swe → decode_swe loop.

Weight sharing: transformer stacks with the same (stack_idx, xcoder_id) in different
language models reference the same nn.Parameter objects so litert_torch deduplicates
them in the flatbuffer (e.g. a shared "all" stack appears only once in the .tflite).

A manifest JSON (same path with .tflite → _manifest.json suffix) records the task →
(src, tgt) mapping needed by infer_multi.py.

Usage:
    python mammoth/litert/convert_multi.py \\
        --checkpoint-dir /path/to/checkpoint \\
        --output mammoth_multi.tflite

On macOS (where litert_torch is unavailable) --export-only is set automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
import torch.nn as nn

from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
    config_from_opts,
    resolve_prefix,
    _load,
)
from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration
from mammoth.litert.wrapper import (
    MammothLiteRTEncoder,
    MammothLiteRTPrefill,
    MammothLiteRTDecode,
    make_encoder_sample_inputs,
    make_prefill_sample_inputs,
    make_decode_sample_inputs,
)
from mammoth.litert.gpu_patches import replace_rms_norms, sdpa_patch


try:
    import litert_torch  # noqa: F401
    LITERT_AVAILABLE = True
except ImportError:
    LITERT_AVAILABLE = False


# ── Discovery ─────────────────────────────────────────────────────────────────

def discover_languages(opts) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Scan opts.tasks and return (src_info, tgt_info).

    src_info: {src_lang: enc_sharing_group_list}
    tgt_info: {tgt_lang: dec_sharing_group_list}
    """
    src_info: dict[str, list[str]] = {}
    tgt_info: dict[str, list[str]] = {}
    for tcfg in opts.tasks.values():
        src_tgt = tcfg["src_tgt"]
        src, _, tgt = src_tgt.partition("-")
        src_info[src] = list(tcfg["enc_sharing_group"])
        tgt_info[tgt] = list(tcfg["dec_sharing_group"])
    return src_info, tgt_info


def discover_tasks(opts) -> dict[str, tuple[str, str]]:
    """Return {task_id: (src, tgt)} for all tasks in opts."""
    tasks = {}
    for tid, tcfg in opts.tasks.items():
        src, _, tgt = tcfg["src_tgt"].partition("-")
        tasks[tid] = (src, tgt)
    return tasks


def shared_stack_keys(lang_info: dict[str, list[str]]) -> set[tuple[int, str]]:
    """Return (stack_idx, xcoder_id) pairs that appear in more than one language.

    These are the stacks whose nn.Parameter objects should be shared across
    language-specific models so litert_torch can deduplicate them.
    """
    counter: Counter = Counter()
    for sg in lang_info.values():
        for stack_idx, xcoder_id in enumerate(sg):
            counter[(stack_idx, xcoder_id)] += 1
    return {k for k, n in counter.items() if n > 1}


# ── Shard loading ─────────────────────────────────────────────────────────────

def _load_shard(ckpt_dir: str, prefix: str, filename: str) -> dict:
    path = os.path.join(ckpt_dir, filename)
    if not os.path.exists(path):
        print(f"  [WARN] missing shard: {path}")
        return {}
    return _load(path)


# ── Per-language model loading ────────────────────────────────────────────────

def _base_config(opts, src_vocab_size: int, tgt_vocab_size: int, vocab, src: str) -> MammothConfig:
    from mammoth.constants import DefaultTokens
    sp = vocab[("src", src)].specials
    return config_from_opts(
        opts,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        bos_token_id=sp.get(DefaultTokens.BOS, 2),
        eos_token_id=sp.get(DefaultTokens.EOS, 0),
        pad_token_id=sp.get(DefaultTokens.PAD, 1),
        encoder_sharing_groups=None,
        decoder_sharing_groups=None,
    )


def load_encoder_model(
    ckpt_dir: str,
    prefix: str,
    src: str,
    encoder_id: list[str],
    opts,
    vocab: dict,
    shared_enc: dict[tuple[int, str], nn.Module],
    enc_shared_keys: set[tuple[int, str]],
    dtype: torch.dtype,
    any_tgt_vocab_size: int,
) -> MammothForConditionalGeneration:
    """Build a model with only encoder weights loaded.

    The decoder has random weights; MammothLiteRTEncoder never touches it.
    Shared stacks are injected from/into shared_enc so all encoder models point
    to the same nn.Parameter objects for those stacks.
    """
    src_vocab_size = len(vocab[("src", src)])
    config = _base_config(opts, src_vocab_size, any_tgt_vocab_size, vocab, src)
    model = MammothForConditionalGeneration(config).to(dtype)

    sd: dict[str, torch.Tensor] = {}

    shard = _load_shard(ckpt_dir, prefix, f"{prefix}_src_embeddings_{src}.pt")
    if "weight" in shard:
        sd["encoder.token_emb.weight"] = shard["weight"].to(dtype)

    enc_key = "_".join(encoder_id)
    shard = _load_shard(ckpt_dir, prefix, f"{prefix}_encoder_wrapper_{enc_key}.pt")
    if "post_emb_norm.weight" in shard:
        sd["encoder.post_emb_norm.weight"] = shard["post_emb_norm.weight"].to(dtype)

    for stack_idx, xcoder_id in enumerate(encoder_id):
        shard = _load_shard(ckpt_dir, prefix, f"{prefix}_encoder_{stack_idx}_{xcoder_id}.pt")
        for k, v in shard.items():
            sd[f"encoder.stacks.{stack_idx}.{k}"] = v.to(dtype)

    model.load_state_dict(sd, strict=False)

    # Share stack modules whose parameters should be deduplicated across languages.
    for stack_idx, xcoder_id in enumerate(encoder_id):
        key = (stack_idx, xcoder_id)
        if key in enc_shared_keys:
            if key not in shared_enc:
                shared_enc[key] = model.encoder.stacks[stack_idx]
            else:
                model.encoder.stacks[stack_idx] = shared_enc[key]

    return model.eval()


def load_decoder_model(
    ckpt_dir: str,
    prefix: str,
    tgt: str,
    decoder_id: list[str],
    opts,
    vocab: dict,
    shared_dec: dict[tuple[int, str], nn.Module],
    dec_shared_keys: set[tuple[int, str]],
    dtype: torch.dtype,
    any_src_vocab_size: int,
    any_src_lang: str,
) -> MammothForConditionalGeneration:
    """Build a model with only decoder weights loaded."""
    tgt_vocab_size = len(vocab[("tgt", tgt)])
    config = _base_config(opts, any_src_vocab_size, tgt_vocab_size, vocab, any_src_lang)
    model = MammothForConditionalGeneration(config).to(dtype)

    sd: dict[str, torch.Tensor] = {}

    shard = _load_shard(ckpt_dir, prefix, f"{prefix}_tgt_embeddings_{tgt}.pt")
    if "weight" in shard:
        sd["decoder.token_emb.weight"] = shard["weight"].to(dtype)

    dec_key = "_".join(decoder_id)
    shard = _load_shard(ckpt_dir, prefix, f"{prefix}_decoder_wrapper_{dec_key}.pt")
    if "post_emb_norm.weight" in shard:
        sd["decoder.post_emb_norm.weight"] = shard["post_emb_norm.weight"].to(dtype)
    if "to_logits.weight" in shard:
        sd["decoder.to_logits.weight"] = shard["to_logits.weight"].to(dtype)

    for stack_idx, xcoder_id in enumerate(decoder_id):
        shard = _load_shard(ckpt_dir, prefix, f"{prefix}_decoder_{stack_idx}_{xcoder_id}.pt")
        for k, v in shard.items():
            sd[f"decoder.stacks.{stack_idx}.{k}"] = v.to(dtype)

    model.load_state_dict(sd, strict=False)

    for stack_idx, xcoder_id in enumerate(decoder_id):
        key = (stack_idx, xcoder_id)
        if key in dec_shared_keys:
            if key not in shared_dec:
                shared_dec[key] = model.decoder.stacks[stack_idx]
            else:
                model.decoder.stacks[stack_idx] = shared_dec[key]

    return model.eval()


# ── Sample inputs ─────────────────────────────────────────────────────────────

def build_sample_inputs(
    encoders: dict[str, MammothLiteRTEncoder],
    prefills: dict[str, MammothLiteRTPrefill],
    decodes: dict[str, MammothLiteRTDecode],
    src_configs: dict[str, MammothConfig],
    tgt_configs: dict[str, MammothConfig],
    enc_max_len: int,
    dec_max_len: int,
) -> tuple[dict, dict, dict]:
    enc_inputs_map: dict[str, tuple] = {}
    prefill_inputs_map: dict[str, tuple] = {}
    decode_inputs_map: dict[str, tuple] = {}

    # Generic encoder output: shapes are identical across all src languages
    # (model_dim is shared); we use the first encoder to produce the hidden states.
    first_src = next(iter(encoders))
    first_enc = encoders[first_src]
    first_src_cfg = src_configs[first_src]
    generic_enc_inputs = make_encoder_sample_inputs(enc_max_len, first_src_cfg.src_vocab_size)
    with torch.no_grad():
        generic_enc_out = first_enc(*generic_enc_inputs)

    for src, enc in encoders.items():
        enc_inputs_map[src] = make_encoder_sample_inputs(enc_max_len, src_configs[src].src_vocab_size)

    for tgt in prefills:
        cfg = tgt_configs[tgt]
        pi = make_prefill_sample_inputs(generic_enc_out, enc_max_len, dec_max_len, cfg)
        prefill_inputs_map[tgt] = pi
        with torch.no_grad():
            prefill_out = prefills[tgt](*pi)
        decode_inputs_map[tgt] = make_decode_sample_inputs(
            generic_enc_out, prefill_out, enc_max_len, dec_max_len, cfg
        )

    return enc_inputs_map, prefill_inputs_map, decode_inputs_map


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(encoders, prefills, decodes, enc_inputs_map, prefill_inputs_map, decode_inputs_map):
    print("\n── Sanity check ─────────────────────────────────────────────────")
    with torch.no_grad():
        for src, enc in encoders.items():
            out = enc(*enc_inputs_map[src])
            print(f"  encode_{src}: {out[0].shape}  norm={out[0].norm():.3f}")
        for tgt, prefill in prefills.items():
            out = prefill(*prefill_inputs_map[tgt])
            print(f"  prefill_{tgt}: logits {out[0].shape}")
        for tgt, decode in decodes.items():
            out = decode(*decode_inputs_map[tgt])
            print(f"  decode_{tgt}: logits {out[0].shape}")
    print("  All forward passes OK.")


# ── torch.export ──────────────────────────────────────────────────────────────

def export_all(encoders, prefills, decodes, enc_inputs_map, prefill_inputs_map, decode_inputs_map):
    print("\n── torch.export ─────────────────────────────────────────────────")
    with sdpa_patch():
        for src, enc in encoders.items():
            print(f"  encode_{src} …")
            torch.export.export(enc, enc_inputs_map[src])
        for tgt, prefill in prefills.items():
            print(f"  prefill_{tgt} …")
            torch.export.export(prefill, prefill_inputs_map[tgt])
        for tgt, decode in decodes.items():
            print(f"  decode_{tgt} …")
            torch.export.export(decode, decode_inputs_map[tgt])
    n = len(encoders) + 2 * len(prefills)
    print(f"  All {n} signatures exported.")


# ── litert_torch conversion ───────────────────────────────────────────────────

def convert_to_litert(
    encoders, prefills, decodes,
    enc_inputs_map, prefill_inputs_map, decode_inputs_map,
    output_path: str,
    quant: bool,
):
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

    # Collect all (name, module, inputs) triples in insertion order.
    signatures: list[tuple] = []
    for src, enc in encoders.items():
        signatures.append((f"encode_{src}", enc, enc_inputs_map[src]))
    for tgt in prefills:
        signatures.append((f"prefill_{tgt}", prefills[tgt], prefill_inputs_map[tgt]))
    for tgt in decodes:
        signatures.append((f"decode_{tgt}", decodes[tgt], decode_inputs_map[tgt]))

    for name, _, _ in signatures:
        print(f"  + {name}")

    name0, mod0, inp0 = signatures[0]
    builder = litert_torch.signature(name0, mod0, inp0)
    for name, mod, inp in signatures[1:]:
        builder = builder.signature(name, mod, inp)

    n = len(encoders) + 2 * len(prefills)
    print(f"  Converting {n} signatures …")
    edge_model = builder.convert(enable_x64=False, quant_config=q_config)
    edge_model.export(output_path)
    print(f"  Saved → {output_path}")


# ── Manifest ──────────────────────────────────────────────────────────────────

def write_manifest(output_path: str, tasks, src_vocab_sizes, tgt_vocab_sizes, enc_max_len, dec_max_len):
    # Derive manifest path by replacing .tflite suffix.
    base = output_path[:-7] if output_path.endswith(".tflite") else output_path
    manifest_path = base + "_manifest.json"
    manifest = {
        "model_type": "mammoth_litert_multi",
        "enc_max_len": enc_max_len,
        "dec_max_len": dec_max_len,
        "tasks": {tid: {"src": src, "tgt": tgt} for tid, (src, tgt) in tasks.items()},
        "src_vocab_sizes": src_vocab_sizes,
        "tgt_vocab_sizes": tgt_vocab_sizes,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest → {manifest_path}")
    return manifest_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert multi-task Mammoth checkpoint → single LiteRT .tflite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Directory containing Mammoth checkpoint shards")
    parser.add_argument("--step", type=int, default=None,
                        help="Load specific step instead of best checkpoint")
    parser.add_argument("--enc-max-len", type=int, default=128)
    parser.add_argument("--dec-max-len", type=int, default=64)
    parser.add_argument("--output", default="mammoth_multi.tflite")
    parser.add_argument("--no-quant", action="store_true")
    parser.add_argument("--export-only", action="store_true",
                        help="Stop after torch.export (skip litert_torch lowering)")
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if not LITERT_AVAILABLE and not args.export_only:
        print("[INFO] litert_torch not available — running in --export-only mode.")
        args.export_only = True

    # ── Load frame ────────────────────────────────────────────────────────────
    prefix = resolve_prefix(args.checkpoint_dir, args.step)
    frame = _load(os.path.join(args.checkpoint_dir, f"{prefix}_frame.pt"))
    opts, vocab = frame["opts"], frame["vocab"]

    src_info, tgt_info = discover_languages(opts)
    all_tasks = discover_tasks(opts)
    enc_shared_keys = shared_stack_keys(src_info)
    dec_shared_keys = shared_stack_keys(tgt_info)

    print(f"Sources  ({len(src_info)}): {sorted(src_info)}")
    print(f"Targets  ({len(tgt_info)}): {sorted(tgt_info)}")
    print(f"Tasks    ({len(all_tasks)}): {sorted(all_tasks)}")
    print(f"Shared encoder stacks: {sorted(enc_shared_keys)}")
    print(f"Shared decoder stacks: {sorted(dec_shared_keys)}")
    n_sigs = len(src_info) + 2 * len(tgt_info)
    print(f"Signatures to pack: {len(src_info)} encode + {len(tgt_info)} prefill"
          f" + {len(tgt_info)} decode = {n_sigs} total")

    # For constructing decoder-side configs we need any valid src lang/vocab.
    any_src = next(iter(src_info))
    any_src_vocab_size = len(vocab[("src", any_src)])
    any_tgt_vocab_size = len(vocab[("tgt", next(iter(tgt_info)))])

    # ── Build encoders ────────────────────────────────────────────────────────
    print("\n── Loading encoders ─────────────────────────────────────────────")
    shared_enc: dict[tuple[int, str], nn.Module] = {}
    src_models: dict[str, MammothForConditionalGeneration] = {}
    src_configs: dict[str, MammothConfig] = {}
    encoders: dict[str, MammothLiteRTEncoder] = {}

    for src in sorted(src_info):
        encoder_id = src_info[src]
        print(f"  {src}: sharing_group={encoder_id}")
        model = load_encoder_model(
            args.checkpoint_dir, prefix, src, encoder_id,
            opts, vocab, shared_enc, enc_shared_keys, dtype, any_tgt_vocab_size,
        )
        src_models[src] = model
        src_configs[src] = model.config
        encoders[src] = MammothLiteRTEncoder(model, args.enc_max_len).eval()
        replace_rms_norms(encoders[src])

    # ── Build decoders ────────────────────────────────────────────────────────
    print("\n── Loading decoders ─────────────────────────────────────────────")
    shared_dec: dict[tuple[int, str], nn.Module] = {}
    tgt_models: dict[str, MammothForConditionalGeneration] = {}
    tgt_configs: dict[str, MammothConfig] = {}
    prefills: dict[str, MammothLiteRTPrefill] = {}
    decodes: dict[str, MammothLiteRTDecode] = {}

    for tgt in sorted(tgt_info):
        decoder_id = tgt_info[tgt]
        print(f"  {tgt}: sharing_group={decoder_id}")
        model = load_decoder_model(
            args.checkpoint_dir, prefix, tgt, decoder_id,
            opts, vocab, shared_dec, dec_shared_keys, dtype,
            any_src_vocab_size, any_src,
        )
        tgt_models[tgt] = model
        tgt_configs[tgt] = model.config
        prefills[tgt] = MammothLiteRTPrefill(model, args.enc_max_len, args.dec_max_len).eval()
        replace_rms_norms(prefills[tgt])
        decodes[tgt] = MammothLiteRTDecode(model, args.enc_max_len, args.dec_max_len).eval()
        replace_rms_norms(decodes[tgt])

    # ── Sample inputs ─────────────────────────────────────────────────────────
    print("\n── Building sample inputs ───────────────────────────────────────")
    enc_inputs_map, prefill_inputs_map, decode_inputs_map = build_sample_inputs(
        encoders, prefills, decodes, src_configs, tgt_configs,
        args.enc_max_len, args.dec_max_len,
    )

    # ── Sanity check ──────────────────────────────────────────────────────────
    sanity_check(encoders, prefills, decodes, enc_inputs_map, prefill_inputs_map, decode_inputs_map)

    # ── torch.export ──────────────────────────────────────────────────────────
    export_all(encoders, prefills, decodes, enc_inputs_map, prefill_inputs_map, decode_inputs_map)

    src_vocab_sizes = {src: len(vocab[("src", src)]) for src in src_info}
    tgt_vocab_sizes = {tgt: len(vocab[("tgt", tgt)]) for tgt in tgt_info}

    if args.export_only:
        print(f"\nDone (export-only). Re-run on Linux with litert_torch to produce {args.output}")
        write_manifest(args.output, all_tasks, src_vocab_sizes, tgt_vocab_sizes,
                       args.enc_max_len, args.dec_max_len)
        return

    # ── LiteRT conversion ─────────────────────────────────────────────────────
    convert_to_litert(
        encoders, prefills, decodes,
        enc_inputs_map, prefill_inputs_map, decode_inputs_map,
        args.output, quant=not args.no_quant,
    )
    write_manifest(args.output, all_tasks, src_vocab_sizes, tgt_vocab_sizes,
                   args.enc_max_len, args.dec_max_len)
    print(f"\nDone → {args.output}")


if __name__ == "__main__":
    main()
