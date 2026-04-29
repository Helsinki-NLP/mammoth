#!/usr/bin/env python3
"""
Convert a sharded Mammoth checkpoint → HuggingFace MammothForConditionalGeneration.

Usage:
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/converted_model \
        --step 0 \
        --src es \
        --tgt en \
        --output-dir path/to/hf_output

The checkpoint directory must contain files produced by Mammoth's model saver:
    _step_{step}_frame.pt
    _step_{step}_src_embeddings_{src}.pt
    _step_{step}_encoder_wrapper_{src}.pt
    _step_{step}_encoder_0_{src}.pt
    _step_{step}_tgt_embeddings_{tgt}.pt
    _step_{step}_decoder_wrapper_{tgt}.pt
    _step_{step}_decoder_0_{tgt}.pt

Outputs saved with save_pretrained() plus copies of configuration_mammoth.py and
modeling_mammoth.py so that trust_remote_code=True loading works.
"""

import argparse
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import PreTrainedTokenizerFast

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------

def config_from_opts(opts, src_vocab_size: int, tgt_vocab_size: int) -> MammothConfig:
    """Build MammothConfig from a Mammoth opts Namespace."""
    xt = getattr(opts, 'x_transformers_opts', {}) or {}

    enc_layers = opts.enc_layers
    if isinstance(enc_layers, (list, tuple)):
        enc_layers = enc_layers[0]
    dec_layers = opts.dec_layers
    if isinstance(dec_layers, (list, tuple)):
        dec_layers = dec_layers[0]

    enc_model_dim = getattr(opts, 'enc_model_dim', 768)
    dec_model_dim = getattr(opts, 'dec_model_dim', enc_model_dim)

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
        enc_heads=xt.get('enc_heads', 12),
        enc_attn_dim_head=xt.get('enc_attn_dim_head', enc_model_dim // xt.get('enc_heads', 12)),
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
        dec_heads=xt.get('dec_heads', 12),
        dec_attn_dim_head=xt.get('dec_attn_dim_head', dec_model_dim // xt.get('dec_heads', 12)),
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
        tie_word_embeddings=True,
        model_dtype=getattr(opts, 'model_dtype', 'bf16'),
    )


# ---------------------------------------------------------------------------
# State dict assembly from shards
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        return torch.load(path, map_location='cpu', weights_only=False)


def assemble_state_dict(ckpt_dir: str, step: int, src: str, tgt: str) -> dict:
    """
    Reassemble a flat HF-model state dict from Mammoth's per-component shards.

    HF model key structure:
        model.encoder.<x-transformers key>
        model.decoder.<x-transformers key>

    Shard → HF prefix mapping:
        src_embeddings_{src}.pt         emb.weight → model.encoder.token_emb.emb.weight
        encoder_wrapper_{src}.pt        post_emb_norm.*, pos_emb.* → model.encoder.*
        encoder_0_{src}.pt              _base_layers.* → model.encoder.attn_layers.attention_layers_stack.0.*
        tgt_embeddings_{tgt}.pt         emb.weight → model.decoder.token_emb.emb.weight
        decoder_wrapper_{tgt}.pt        post_emb_norm.*, pos_emb.* → model.decoder.*
        decoder_0_{tgt}.pt              _base_layers.* → model.decoder.attn_layers.attention_layers_stack.0.*
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

    def attn_load(filename: str, hf_prefix: str):
        """Map _base_layers.* → {hf_prefix}.layers.* (x-transformers AttentionLayers key name)."""
        path = os.path.join(ckpt_dir, filename)
        if not os.path.exists(path):
            print(f"  [WARN] missing shard: {path}")
            return
        shard = _load(path)
        for k, v in shard.items():
            if k.startswith('_base_layers.'):
                new_k = f'{hf_prefix}.layers.' + k[len('_base_layers.'):]
            else:
                new_k = f'{hf_prefix}.{k}'
            sd[new_k] = v

    p = f'_step_{step}'

    prefix_load(f'{p}_src_embeddings_{src}.pt', 'model.encoder.token_emb')
    prefix_load(f'{p}_encoder_wrapper_{src}.pt', 'model.encoder')
    attn_load(f'{p}_encoder_0_{src}.pt',         'model.encoder.attn_layers')
    prefix_load(f'{p}_tgt_embeddings_{tgt}.pt',  'model.decoder.token_emb')
    prefix_load(f'{p}_decoder_wrapper_{tgt}.pt', 'model.decoder')
    attn_load(f'{p}_decoder_0_{tgt}.pt',         'model.decoder.attn_layers')

    return sd


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(ckpt_dir: str, step: int, src: str, tgt: str, output_dir: str):
    frame_path = os.path.join(ckpt_dir, f'_step_{step}_frame.pt')
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

    config = config_from_opts(opts, src_vocab_size, tgt_vocab_size)
    print(f"Config: enc {config.enc_layers}×{config.enc_model_dim}d, "
          f"dec {config.dec_layers}×{config.dec_model_dim}d")

    print("Building HF model...")
    model = MammothForConditionalGeneration(config)

    print("Assembling state dict from shards...")
    hf_sd = assemble_state_dict(ckpt_dir, step, src, tgt)
    print(f"  Assembled {len(hf_sd)} tensors")

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
    parser.add_argument('--step', type=int, default=0,
                        help="Checkpoint step number (default: 0)")
    parser.add_argument('--src', required=True,
                        help="Source language id (e.g. 'es')")
    parser.add_argument('--tgt', required=True,
                        help="Target language id (e.g. 'en')")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for HF model")
    args = parser.parse_args()

    convert(args.checkpoint_dir, args.step, args.src, args.tgt, args.output_dir)


if __name__ == '__main__':
    main()
