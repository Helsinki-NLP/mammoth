#!/usr/bin/env python3
"""
Convert a sharded Mammoth checkpoint → HuggingFace MammothForConditionalGeneration.

Usage:
    # Load the best checkpoint (default — requires *_best_frame.pt):
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --src es --tgt en --output-dir path/to/hf_output

    # Load the checkpoint with the highest step number:
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --load-last \
        --src es --tgt en --output-dir path/to/hf_output

    # Load a specific step:
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --step 500 \
        --src es --tgt en --output-dir path/to/hf_output

The checkpoint directory must contain files produced by Mammoth's model saver.
Best checkpoint naming:   _best_frame.pt, _best_src_embeddings_{src}.pt, ...
Step checkpoint naming:   _step_{N}_frame.pt, _step_{N}_src_embeddings_{src}.pt, ...

Outputs saved with save_pretrained() plus copies of configuration_mammoth.py and
modeling_mammoth.py so that trust_remote_code=True loading works.
"""

import argparse
import glob as glob_module
import os
import re
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import PreTrainedTokenizerFast

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def config_from_opts(
    opts,
    src_vocab_size: int,
    tgt_vocab_size: int,
    tie_word_embeddings: bool,
    enc_attn_dim_head_override: int | None = None,
    dec_attn_dim_head_override: int | None = None,
) -> MammothConfig:
    """Build MammothConfig from a Mammoth opts Namespace."""
    xt = getattr(opts, 'x_transformers_opts', {}) or {}

    enc_layers = opts.enc_layers
    if isinstance(enc_layers, (list, tuple)):
        enc_layers = sum(enc_layers)  # total depth across all components
    dec_layers = opts.dec_layers
    if isinstance(dec_layers, (list, tuple)):
        dec_layers = sum(dec_layers)  # total depth across all components

    model_dim = getattr(opts, 'model_dim', None)
    if model_dim is not None:
        enc_model_dim = model_dim
        dec_model_dim = model_dim
    else:
        enc_model_dim = getattr(opts, 'enc_model_dim', 768)
        dec_model_dim = getattr(opts, 'dec_model_dim', enc_model_dim)

    enc_heads = xt.get('heads', xt.get('enc_heads', 12))
    dec_heads = xt.get('heads', xt.get('dec_heads', 12))
    enc_attn_dim_head = enc_attn_dim_head_override if enc_attn_dim_head_override is not None \
        else xt.get('enc_attn_dim_head', enc_model_dim // enc_heads)
    dec_attn_dim_head = dec_attn_dim_head_override if dec_attn_dim_head_override is not None \
        else xt.get('dec_attn_dim_head', dec_model_dim // dec_heads)

    return MammothConfig(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        enc_model_dim=enc_model_dim,
        dec_model_dim=dec_model_dim,
        enc_layers=enc_layers,
        dec_layers=dec_layers,
        enc_max_seq_len=getattr(opts, 'src_seq_length_max', 1024),
        dec_max_seq_len=getattr(opts, 'tgt_seq_length_max', 1024),
        # Encoder attention
        enc_heads=enc_heads,
        enc_attn_dim_head=enc_attn_dim_head,
        enc_attn_dropout=xt.get('enc_attn_dropout', 0.0),
        enc_attn_qkv_bias=xt.get('enc_attn_qkv_bias', True),
        enc_attn_flash=xt.get('attn_flash', False),
        # Encoder FFN
        enc_ff_mult=xt.get('enc_ff_mult', 4.0),
        enc_ff_glu=xt.get('enc_ff_glu', False),
        enc_ff_no_bias=xt.get('enc_ff_no_bias', False),
        enc_ff_dropout=xt.get('enc_ff_dropout', 0.0),
        # Encoder norm/pos
        enc_pre_norm=xt.get('enc_pre_norm', False),
        enc_use_rmsnorm=xt.get('enc_use_rmsnorm', False),
        enc_layernorm_bias=xt.get('enc_layernorm_bias', True),
        enc_norm_add_unit_offset=xt.get('enc_norm_add_unit_offset', False),
        enc_rotary_pos_emb=xt.get('enc_rotary_pos_emb', False),
        enc_use_abs_pos_emb=xt.get('use_abs_pos_emb', True),
        enc_post_emb_norm=xt.get('enc_post_emb_norm', True),
        enc_post_emb_norm_bias=xt.get('enc_post_emb_norm_bias', True),
        enc_scaled_embeddings=xt.get('enc_scaled_embeddings', False),
        enc_emb_dropout=xt.get('enc_emb_dropout', 0.0),
        enc_sliding_window=getattr(opts, 'enc_sliding_window', -1),
        # Decoder attention
        dec_heads=dec_heads,
        dec_attn_dim_head=dec_attn_dim_head,
        dec_attn_dropout=xt.get('dec_attn_dropout', 0.0),
        dec_attn_qkv_bias=xt.get('dec_attn_qkv_bias', True),
        dec_attn_flash=xt.get('attn_flash', False),
        dec_attn_kv_heads=xt.get('dec_attn_kv_heads', None),
        dec_attn_qk_norm=xt.get('dec_attn_qk_norm', False),
        dec_attn_qk_norm_dim_scale=xt.get('dec_attn_qk_norm_dim_scale', False),
        dec_cross_attn_dim_context=xt.get('dec_cross_attn_dim_context', None),
        # Decoder FFN
        dec_ff_mult=xt.get('dec_ff_mult', 4.0),
        dec_ff_glu=xt.get('dec_ff_glu', False),
        dec_ff_no_bias=xt.get('dec_ff_no_bias', False),
        dec_ff_dropout=xt.get('dec_ff_dropout', 0.0),
        # Decoder norm/pos
        dec_pre_norm=xt.get('dec_pre_norm', False),
        dec_use_rmsnorm=xt.get('dec_use_rmsnorm', False),
        dec_layernorm_bias=xt.get('dec_layernorm_bias', True),
        dec_norm_add_unit_offset=xt.get('dec_norm_add_unit_offset', False),
        dec_rotary_pos_emb=xt.get('dec_rotary_pos_emb', False),
        dec_use_abs_pos_emb=xt.get('use_abs_pos_emb', True),
        dec_post_emb_norm=xt.get('dec_post_emb_norm', True),
        dec_post_emb_norm_bias=xt.get('dec_post_emb_norm_bias', True),
        dec_scaled_embeddings=xt.get('dec_scaled_embeddings', False),
        dec_emb_dropout=xt.get('dec_emb_dropout', 0.0),
        dec_sandwich_norm=xt.get('dec_sandwich_norm', False),
        dec_sliding_window=getattr(opts, 'dec_sliding_window', -1),
        dec_global_attn_every_n_layers=getattr(opts, 'dec_global_attn_every_n_layers', 0),
        dec_global_rope_theta=getattr(opts, 'dec_global_rope_theta', 10000.0),
        dec_local_rope_theta=getattr(opts, 'dec_local_rope_theta', 10000.0),
        # Misc
        tie_word_embeddings=tie_word_embeddings,
        model_dtype=getattr(opts, 'model_dtype', 'bf16'),
    )


# ---------------------------------------------------------------------------
# Checkpoint prefix resolution
# ---------------------------------------------------------------------------

def resolve_prefix(ckpt_dir: str, step: int | None) -> str:
    """Return the shard filename prefix to use (_step_N or _best).

    Priority:
      1. --step N  → _step_N  (explicit, always wins)
      2. best checkpoint (*_best_frame.pt) if it exists
      3. fallback  → _step_<largest N found>
    """
    if step is not None:
        return f'_step_{step}'

    best_frames = glob_module.glob(os.path.join(ckpt_dir, '*_best_frame.pt'))
    if best_frames:
        print(f"Loading best checkpoint: {os.path.basename(best_frames[0])}")
        return '_best'

    frames = glob_module.glob(os.path.join(ckpt_dir, '*_step_*_frame.pt'))
    if not frames:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir}")
    steps = [
        int(m.group(1))
        for f in frames
        if (m := re.search(r'_step_(\d+)_frame\.pt$', os.path.basename(f)))
    ]
    if not steps:
        raise FileNotFoundError(f"Could not parse step numbers from frame files in {ckpt_dir}")
    best_step = max(steps)
    print(f"No best checkpoint found, loading last checkpoint: step {best_step}")
    return f'_step_{best_step}'


# ---------------------------------------------------------------------------
# State dict assembly from shards
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def _task_xcoder_ids(opts, src: str, tgt: str):
    """Return (encoder_id, decoder_id) lists for the given src→tgt task from opts.

    Each list has one xcoder_id per layer stack index, e.g. ['eng', 'all'] for a
    2-stack encoder where the first stack is language-specific and the second shared.
    These map directly to the shard filenames: encoder_{i}_{encoder_id[i]}.pt.
    """
    tasks = getattr(opts, 'tasks', None) or {}
    for corpus_opts in tasks.values():
        src_tgt = corpus_opts.get('src_tgt', '')
        task_src, _, task_tgt = src_tgt.partition('-')
        if task_src == src and task_tgt == tgt:
            enc_id = corpus_opts.get('enc_sharing_group', [src])
            dec_id = corpus_opts.get('dec_sharing_group', [tgt])
            return list(enc_id), list(dec_id)
    available = [v.get('src_tgt') for v in tasks.values()]
    raise ValueError(f"Task {src}-{tgt} not found in opts.tasks. Available: {available}")


def assemble_state_dict(
    ckpt_dir: str, prefix: str, src: str, tgt: str,
    encoder_id: list, decoder_id: list,
) -> dict:
    """
    Reassemble a flat HF-model state dict from Mammoth's per-component shards.

    HF model key structure:
        model.encoder.<x-transformers key>
        model.decoder.<x-transformers key>

    Shard → HF prefix mapping:
        src_embeddings_{src}.pt                        → model.encoder.token_emb.*
        encoder_wrapper_{'_'.join(encoder_id)}.pt      → model.encoder.*
        encoder_{i}_{encoder_id[i]}.pt  _base_layers.{j}.* → model.encoder.attn_layers.layers.{offset+j}.*
        tgt_embeddings_{tgt}.pt                        → model.decoder.token_emb.*
        decoder_wrapper_{'_'.join(decoder_id)}.pt      → model.decoder.*
        decoder_{i}_{decoder_id[i]}.pt  _base_layers.{j}.* → model.decoder.attn_layers.layers.{offset+j}.*

    encoder_id / decoder_id are the xcoder_id lists from opts.tasks (enc_sharing_group /
    dec_sharing_group), which determine both the wrapper filename and the per-stack shards.
    """
    sd = {}

    def prefix_load(filename: str, hf_prefix: str):
        path = os.path.join(ckpt_dir, filename)
        if not os.path.exists(path):
            print(f"  [WARN] missing shard: {path}")
            return
        shard = _load(path)
        for k, v in shard.items():
            sd[f'{hf_prefix}.{k}'] = v

    def attn_load(filename: str, hf_prefix: str, layer_offset: int) -> int:
        """Load one attention-component shard, offsetting _base_layers indices.

        Returns the number of layers contributed by this shard (0 if file missing).
        """
        path = os.path.join(ckpt_dir, filename)
        if not os.path.exists(path):
            print(f"  [WARN] missing shard: {path}")
            return 0
        shard = _load(path)
        max_local_idx = -1
        for k, v in shard.items():
            if k.startswith('_base_layers.'):
                rest = k[len('_base_layers.'):]
                dot_pos = rest.index('.')
                local_idx = int(rest[:dot_pos])
                global_idx = local_idx + layer_offset
                new_k = f'{hf_prefix}.layers.{global_idx}{rest[dot_pos:]}'
                if local_idx > max_local_idx:
                    max_local_idx = local_idx
            else:
                new_k = f'{hf_prefix}.{k}'
            sd[new_k] = v
        return max_local_idx + 1 if max_local_idx >= 0 else 0

    p = prefix

    prefix_load(f'{p}_src_embeddings_{src}.pt', 'model.encoder.token_emb')
    prefix_load(f'{p}_encoder_wrapper_{"_".join(encoder_id)}.pt', 'model.encoder')
    prefix_load(f'{p}_tgt_embeddings_{tgt}.pt', 'model.decoder.token_emb')
    prefix_load(f'{p}_decoder_wrapper_{"_".join(decoder_id)}.pt', 'model.decoder')

    enc_offset = 0
    for i, xcoder_id in enumerate(encoder_id):
        fname = f'{p}_encoder_{i}_{xcoder_id}.pt'
        n = attn_load(fname, 'model.encoder.attn_layers', enc_offset)
        print(f"  encoder component {i} ({xcoder_id}): {n} layer(s) at offset {enc_offset}")
        enc_offset += n

    dec_offset = 0
    for i, xcoder_id in enumerate(decoder_id):
        fname = f'{p}_decoder_{i}_{xcoder_id}.pt'
        n = attn_load(fname, 'model.decoder.attn_layers', dec_offset)
        print(f"  decoder component {i} ({xcoder_id}): {n} layer(s) at offset {dec_offset}")
        dec_offset += n

    return sd


def _infer_attn_dim_head(hf_sd: dict, side: str, heads: int) -> int | None:
    """Read the first to_q.weight for encoder/decoder and back-calculate dim_head."""
    for k, v in hf_sd.items():
        if k.startswith(f'model.{side}.attn_layers.layers.') and k.endswith('.to_q.weight'):
            return v.shape[0] // heads
    return None


def _patch_config_from_sd(config, hf_sd: dict) -> None:
    """Correct config flags that can be reliably inferred from the assembled checkpoint.

    Handles cases where training opts don't store every x_transformers flag
    and the converter's defaults don't match the actual trained architecture.
    """
    def _has(side: str, suffix: str) -> bool:
        prefix = f'model.{side}.attn_layers.layers.'
        return any(k.startswith(prefix) and k.endswith(suffix) for k in hf_sd)

    config.enc_attn_qkv_bias = _has('encoder', '.to_q.bias')
    config.dec_attn_qkv_bias = _has('decoder', '.to_q.bias')

    # Layer norm bias: x-transformers uses 'beta' for the bias term
    config.enc_layernorm_bias = _has('encoder', '.beta')
    config.dec_layernorm_bias = _has('decoder', '.beta')

    # FFN bias
    config.enc_ff_no_bias = not _has('encoder', '.ff.0.0.bias') and not _has('encoder', '.ff.2.bias')
    config.dec_ff_no_bias = not _has('decoder', '.ff.0.0.bias') and not _has('decoder', '.ff.2.bias')

    # post_emb_norm presence/bias
    config.enc_post_emb_norm = 'model.encoder.post_emb_norm.gamma' in hf_sd
    config.dec_post_emb_norm = 'model.decoder.post_emb_norm.gamma' in hf_sd
    config.enc_post_emb_norm_bias = 'model.encoder.post_emb_norm.beta' in hf_sd
    config.dec_post_emb_norm_bias = 'model.decoder.post_emb_norm.beta' in hf_sd


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(ckpt_dir: str, prefix: str, src: str, tgt: str, output_dir: str):
    frame_path = os.path.join(ckpt_dir, f'{prefix}_frame.pt')
    frame = _load(frame_path)
    opts = frame['opts']
    vocab = frame['vocab']

    # vocab keys are tuples: ('src', lang) and ('tgt', lang)
    src_vocab_obj = vocab.get(('src', src)) or next(
        v for k, v in vocab.items() if k[0] == 'src'
    )
    tgt_vocab_obj = vocab.get(('tgt', tgt)) or next(
        v for k, v in vocab.items() if k[0] == 'tgt'
    )
    src_vocab_size = len(src_vocab_obj)
    tgt_vocab_size = len(tgt_vocab_obj)
    print(f"Vocab sizes: src={src_vocab_size}, tgt={tgt_vocab_size}")

    encoder_id, decoder_id = _task_xcoder_ids(opts, src, tgt)
    print(f"Task components: encoder={encoder_id}, decoder={decoder_id}")

    # When tied, x-transformers uses a lambda for to_logits (no saved weights).
    # Scan the decoder wrapper + component shards for any 'to_logits' key.
    p = prefix
    dec_wrapper_key = '_'.join(decoder_id)
    decoder_shards = [f'{p}_decoder_wrapper_{dec_wrapper_key}.pt'] + [
        f'{p}_decoder_{i}_{xcoder_id}.pt' for i, xcoder_id in enumerate(decoder_id)
    ]
    tie = not any(
        'to_logits' in k
        for fname in decoder_shards
        if os.path.exists(os.path.join(ckpt_dir, fname))
        for k in _load(os.path.join(ckpt_dir, fname))
    )
    print(f"tie_word_embeddings: {tie}")

    print("Assembling state dict from shards...")
    hf_sd = assemble_state_dict(ckpt_dir, prefix, src, tgt, encoder_id, decoder_id)
    print(f"  Assembled {len(hf_sd)} tensors")

    # Infer actual attention dim_head from checkpoint weights to avoid size mismatches
    # when opts defaults don't match the trained model.
    xt = getattr(opts, 'x_transformers_opts', {}) or {}
    enc_heads = xt.get('heads', xt.get('enc_heads', 12))
    dec_heads = xt.get('heads', xt.get('dec_heads', 12))
    enc_attn_dim_head_override = _infer_attn_dim_head(hf_sd, 'encoder', enc_heads)
    dec_attn_dim_head_override = _infer_attn_dim_head(hf_sd, 'decoder', dec_heads)
    if enc_attn_dim_head_override is not None:
        print(f"  Inferred enc_attn_dim_head={enc_attn_dim_head_override} "
              f"(inner_dim={enc_heads * enc_attn_dim_head_override})")
    if dec_attn_dim_head_override is not None:
        print(f"  Inferred dec_attn_dim_head={dec_attn_dim_head_override} "
              f"(inner_dim={dec_heads * dec_attn_dim_head_override})")

    config = config_from_opts(
        opts, src_vocab_size, tgt_vocab_size, tie,
        enc_attn_dim_head_override=enc_attn_dim_head_override,
        dec_attn_dim_head_override=dec_attn_dim_head_override,
    )
    _patch_config_from_sd(config, hf_sd)
    print(f"Config: enc {config.enc_layers}×{config.enc_model_dim}d "
          f"(heads={config.enc_heads}, dim_head={config.enc_attn_dim_head}, "
          f"qkv_bias={config.enc_attn_qkv_bias}, ln_bias={config.enc_layernorm_bias}), "
          f"dec {config.dec_layers}×{config.dec_model_dim}d "
          f"(heads={config.dec_heads}, dim_head={config.dec_attn_dim_head}, "
          f"qkv_bias={config.dec_attn_qkv_bias}, ln_bias={config.dec_layernorm_bias})")

    print("Building HF model...")
    model = MammothForConditionalGeneration(config)

    missing, unexpected = [], []
    model_sd = model.state_dict()
    for k in model_sd:
        if k not in hf_sd:
            missing.append(k)
    for k in hf_sd:
        if k not in model_sd:
            unexpected.append(k)
    if missing:
        print(f"  [WARN] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.load_state_dict(hf_sd, strict=False)
    print("Weights loaded.")

    os.makedirs(output_dir, exist_ok=True)
    config.auto_map = {
        "AutoConfig": "configuration_mammoth.MammothConfig",
        "AutoModelForSeq2SeqLM": "modeling_mammoth.MammothForConditionalGeneration",
    }
    model.save_pretrained(output_dir)

    # Save tokenizer — Mammoth uses an HF-compatible tokenizers.Tokenizer,
    # so we just wrap it with PreTrainedTokenizerFast.
    specials = src_vocab_obj.specials  # {'<s>': 0, '<pad>': 1, '</s>': 2, ...}
    _rev = {v: k for k, v in specials.items()}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=src_vocab_obj.path,
        bos_token=_rev.get(config.bos_token_id, '<s>'),
        eos_token=_rev.get(config.eos_token_id, '</s>'),
        unk_token=_rev.get(getattr(config, 'unk_token_id', 3), '<unk>'),
        pad_token=_rev.get(config.pad_token_id, '<pad>'),
    )
    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved ({len(tokenizer)} tokens).")

    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("configuration_mammoth.py", "modeling_mammoth.py"):
        shutil.copy(os.path.join(here, fname), os.path.join(output_dir, fname))

    # Vendor the x-transformers fork as flat sibling .py files so trust_remote_code
    # users do not need `pip install mammoth`. HF's dynamic-module loader only
    # supports relative imports of sibling .py files, not sub-packages.
    import mammoth.x_transformers as _mxt
    xt_src_dir = os.path.dirname(_mxt.__file__)
    for fname in ("x_transformers.py", "attend.py", "autoregressive_wrapper.py"):
        shutil.copy(os.path.join(xt_src_dir, fname), os.path.join(output_dir, fname))

    print(f"Saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert Mammoth checkpoint to HuggingFace format")
    parser.add_argument('--checkpoint-dir', required=True,
                        help="Directory containing Mammoth shard files")
    parser.add_argument('--src', required=True,
                        help="Source language id (e.g. 'es')")
    parser.add_argument('--tgt', required=True,
                        help="Target language id (e.g. 'en')")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for HF model")

    parser.add_argument('--step', type=int, default=None,
                        help="Load a specific checkpoint step (e.g. --step 500). "
                             "Default: best checkpoint, or last step if no best exists.")

    args = parser.parse_args()
    prefix = resolve_prefix(args.checkpoint_dir, args.step)
    convert(args.checkpoint_dir, prefix, args.src, args.tgt, args.output_dir)


if __name__ == '__main__':
    main()
