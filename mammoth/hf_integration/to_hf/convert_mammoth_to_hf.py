#!/usr/bin/env python3
"""
Convert a sharded Mammoth checkpoint → HuggingFace MammothForConditionalGeneration.

The script discovers src/tgt language pairs from the checkpoint's stored opts.tasks —
users do not pass --src/--tgt. Single-task checkpoints write flat into --output-dir;
multi-task checkpoints write one subdirectory per task: <output_dir>/<src>-<tgt>/.

Usage:
    # Convert all tasks (default; best checkpoint):
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --output-dir path/to/hf_output

    # Convert a specific task only (multi-task checkpoint):
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --task eng-spa \
        --output-dir path/to/hf_output

    # Load a specific step instead of the best checkpoint:
    python convert_mammoth_to_hf.py \
        --checkpoint-dir path/to/checkpoint_dir \
        --step 500 \
        --output-dir path/to/hf_output

Checkpoint shard naming:
    Best:  _best_frame.pt, _best_src_embeddings_{src}.pt, ...
    Step:  _step_{N}_frame.pt, _step_{N}_src_embeddings_{src}.pt, ...

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

from mammoth.constants import DefaultTokens
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
    enc_layers_override: int | None = None,
    dec_layers_override: int | None = None,
    bos_token_id: int = 2,
    eos_token_id: int = 0,
    decoder_start_token_id: int = 2,
    pad_token_id: int = 1,
) -> MammothConfig:
    """Build MammothConfig from a Mammoth opts Namespace."""
    xt = getattr(opts, 'x_transformers_opts', {}) or {}

    if enc_layers_override is not None:
        enc_layers = enc_layers_override
    else:
        enc_layers = opts.enc_layers
        if isinstance(enc_layers, (list, tuple)):
            enc_layers = sum(enc_layers)
    if dec_layers_override is not None:
        dec_layers = dec_layers_override
    else:
        dec_layers = opts.dec_layers
        if isinstance(dec_layers, (list, tuple)):
            dec_layers = sum(dec_layers)

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
        enc_attn_dropout=xt.get('enc_attn_dropout', xt.get('attn_dropout', 0.0)),
        enc_attn_qkv_bias=xt.get('enc_attn_qkv_bias', True),
        enc_attn_flash=xt.get('attn_flash', False),
        # Encoder FFN
        enc_ff_mult=xt.get('enc_ff_mult', xt.get('ff_mult', 4.0)),
        enc_ff_glu=xt.get('enc_ff_glu', xt.get('ff_glu', False)),
        enc_ff_no_bias=xt.get('enc_ff_no_bias', xt.get('ff_no_bias', False)),
        enc_ff_dropout=xt.get('enc_ff_dropout', xt.get('ff_dropout', 0.0)),
        # Encoder norm/pos
        enc_pre_norm=xt.get('enc_pre_norm', xt.get('pre_norm', False)),
        # use_fused_rmsnorm is a separate flag in x-transformers (FusedRMSNorm backed by nn.RMSNorm).
        # For unit_offset=True (the default), both FusedRMSNorm and RMSNorm store the scale delta
        # as `.g` and compute x/rms(x)*(g+1), so the HF model can use use_rmsnorm=True safely.
        enc_use_rmsnorm=xt.get('enc_use_rmsnorm', xt.get('use_rmsnorm', False)) or xt.get('enc_use_fused_rmsnorm', xt.get('use_fused_rmsnorm', False)),
        enc_layernorm_bias=xt.get('enc_layernorm_bias', True),
        enc_norm_add_unit_offset=xt.get('enc_norm_add_unit_offset', True),
        enc_rotary_pos_emb=xt.get('enc_rotary_pos_emb', xt.get('rotary_pos_emb', False)),
        enc_use_abs_pos_emb=xt.get('use_abs_pos_emb', True),
        enc_post_emb_norm=xt.get('enc_post_emb_norm', xt.get('post_emb_norm', True)),
        enc_post_emb_norm_bias=xt.get('enc_post_emb_norm_bias', True),
        enc_scaled_embeddings=xt.get('enc_scaled_embeddings', False),
        enc_emb_dropout=xt.get('enc_emb_dropout', xt.get('emb_dropout', 0.0)),
        enc_sliding_window=xt.get('enc_sliding_window', getattr(opts, 'enc_sliding_window', -1)),
        # Decoder attention
        dec_heads=dec_heads,
        dec_attn_dim_head=dec_attn_dim_head,
        dec_attn_dropout=xt.get('dec_attn_dropout', xt.get('attn_dropout', 0.0)),
        dec_attn_qkv_bias=xt.get('dec_attn_qkv_bias', True),
        dec_attn_flash=xt.get('attn_flash', False),
        dec_attn_kv_heads=xt.get('dec_attn_kv_heads', None),
        dec_attn_qk_norm=xt.get('dec_attn_qk_norm', False),
        dec_attn_qk_norm_dim_scale=xt.get('dec_attn_qk_norm_dim_scale', False),
        dec_cross_attn_dim_context=xt.get('dec_cross_attn_dim_context', None),
        # Decoder FFN
        dec_ff_mult=xt.get('dec_ff_mult', xt.get('ff_mult', 4.0)),
        dec_ff_glu=xt.get('dec_ff_glu', xt.get('ff_glu', False)),
        dec_ff_no_bias=xt.get('dec_ff_no_bias', xt.get('ff_no_bias', False)),
        dec_ff_dropout=xt.get('dec_ff_dropout', xt.get('ff_dropout', 0.0)),
        # Decoder norm/pos
        dec_pre_norm=xt.get('dec_pre_norm', xt.get('pre_norm', False)),
        dec_use_rmsnorm=xt.get('dec_use_rmsnorm', xt.get('use_rmsnorm', False)) or xt.get('dec_use_fused_rmsnorm', xt.get('use_fused_rmsnorm', False)),
        dec_layernorm_bias=xt.get('dec_layernorm_bias', True),
        dec_norm_add_unit_offset=xt.get('dec_norm_add_unit_offset', True),
        dec_rotary_pos_emb=xt.get('dec_rotary_pos_emb', xt.get('rotary_pos_emb', False)),
        dec_use_abs_pos_emb=xt.get('use_abs_pos_emb', True),
        dec_post_emb_norm=xt.get('dec_post_emb_norm', xt.get('post_emb_norm', True)),
        dec_post_emb_norm_bias=xt.get('dec_post_emb_norm_bias', True),
        dec_scaled_embeddings=xt.get('dec_scaled_embeddings', False),
        dec_emb_dropout=xt.get('dec_emb_dropout', xt.get('emb_dropout', 0.0)),
        dec_sandwich_norm=xt.get('dec_sandwich_norm', False),
        dec_sliding_window=xt.get('dec_sliding_window', getattr(opts, 'dec_sliding_window', -1)),
        dec_global_attn_every_n_layers=xt.get('dec_global_attn_every_n_layers', getattr(opts, 'dec_global_attn_every_n_layers', 0)),
        dec_global_rope_theta=xt.get('dec_global_rope_theta', getattr(opts, 'dec_global_rope_theta', 10000.0)),
        dec_local_rope_theta=xt.get('dec_local_rope_theta', getattr(opts, 'dec_local_rope_theta', 10000.0)),
        # Misc
        tie_word_embeddings=tie_word_embeddings,
        model_dtype=getattr(opts, 'model_dtype', 'bf16'),
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        decoder_start_token_id=decoder_start_token_id,
        pad_token_id=pad_token_id,
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


def _infer_xcoder_ids_from_shards(ckpt_dir: str, prefix: str, src: str, tgt: str):
    """Infer (encoder_id, decoder_id) lists by scanning shard filenames.

    Falls back to ([src], [tgt]) when no matching shards are found.
    Shard pattern: {prefix}_encoder_{i}_{id}.pt  (and decoder equivalent).
    """
    import re
    enc_pattern = re.compile(rf'^{re.escape(prefix)}_encoder_(\d+)_(.+?)(?:_optim)?\.pt$')
    dec_pattern = re.compile(rf'^{re.escape(prefix)}_decoder_(\d+)_(.+?)(?:_optim)?\.pt$')

    enc_map, dec_map = {}, {}
    for fname in os.listdir(ckpt_dir):
        m = enc_pattern.match(fname)
        if m:
            enc_map[int(m.group(1))] = m.group(2)
        m = dec_pattern.match(fname)
        if m:
            dec_map[int(m.group(1))] = m.group(2)

    if enc_map and dec_map:
        encoder_id = [enc_map[i] for i in sorted(enc_map)]
        decoder_id = [dec_map[i] for i in sorted(dec_map)]
        print(f"[WARN] opts.tasks empty — inferred from shards: encoder={encoder_id}, decoder={decoder_id}")
        return encoder_id, decoder_id

    print(f"[WARN] opts.tasks empty and no shards found — falling back to encoder=['{src}'], decoder=['{tgt}']")
    return [src], [tgt]


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
            elif k.startswith('layers.'):
                # AdaptedAttentionLayers stores layers under both 'layers.*' and '_base_layers.*'
                # (same tensors, different names). Skip 'layers.*' here — '_base_layers.*' already
                # handles the offset remapping. Without this skip, component N's 'layers.0-K.*'
                # overwrites the correctly-offset weights from all earlier components.
                continue
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
        depth = n // 2  # encoder: 2 sub-layers per depth block (self-attn + FFN)
        print(f"  encoder component {i} ({xcoder_id}): depth={depth} ({n} sub-layers) at offset {enc_offset}")
        enc_offset += n

    dec_offset = 0
    for i, xcoder_id in enumerate(decoder_id):
        fname = f'{p}_decoder_{i}_{xcoder_id}.pt'
        n = attn_load(fname, 'model.decoder.attn_layers', dec_offset)
        depth = n // 3  # decoder: 3 sub-layers per depth block (self-attn + cross-attn + FFN)
        print(f"  decoder component {i} ({xcoder_id}): depth={depth} ({n} sub-layers) at offset {dec_offset}")
        dec_offset += n

    # total depth = sum over components
    enc_depth = enc_offset // 2
    dec_depth = dec_offset // 3
    print(f"  Total depth from checkpoint: enc_layers={enc_depth}, dec_layers={dec_depth}")

    return sd, enc_depth, dec_depth


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

    # post_emb_norm presence/bias: LayerNorm uses .gamma, RMSNorm uses .g
    config.enc_post_emb_norm = (
        'model.encoder.post_emb_norm.gamma' in hf_sd or 'model.encoder.post_emb_norm.g' in hf_sd
    )
    config.dec_post_emb_norm = (
        'model.decoder.post_emb_norm.gamma' in hf_sd or 'model.decoder.post_emb_norm.g' in hf_sd
    )
    config.enc_post_emb_norm_bias = 'model.encoder.post_emb_norm.beta' in hf_sd
    config.dec_post_emb_norm_bias = 'model.decoder.post_emb_norm.beta' in hf_sd

    # norm_add_unit_offset: norms init their scale param to 0 when unit_offset=True (effective=1).
    # LayerNorm uses .gamma, RMSNorm/FusedRMSNorm (unit_offset mode) uses .g — check both.
    def _mean_gamma(side: str) -> float | None:
        prefix = f'model.{side}.attn_layers.layers.'
        gammas = [v for k, v in hf_sd.items()
                  if k.startswith(prefix) and (k.endswith('.0.2.gamma') or k.endswith('.0.2.g'))]
        if not gammas:
            return None
        return sum(g.mean().item() for g in gammas) / len(gammas)

    enc_gamma_mean = _mean_gamma('encoder')
    dec_gamma_mean = _mean_gamma('decoder')
    if enc_gamma_mean is not None:
        config.enc_norm_add_unit_offset = abs(enc_gamma_mean) < 0.1
    if dec_gamma_mean is not None:
        config.dec_norm_add_unit_offset = abs(dec_gamma_mean) < 0.1


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

    # Derive BOS/EOS/PAD token IDs from the src vocab specials.
    # Mammoth trains with source wrapped as [BOS, tokens..., EOS].
    # DefaultTokens.BOS = '<s>', DefaultTokens.EOS = '</s>'.
    src_specials = src_vocab_obj.specials
    bos_str = DefaultTokens.BOS
    eos_str = DefaultTokens.EOS
    bos_id = src_specials.get(bos_str, 2)
    eos_id = src_specials.get(eos_str, 0)
    pad_id = src_specials.get(DefaultTokens.PAD, 1)
    print(f"Token IDs: BOS={bos_str!r}={bos_id}, EOS={eos_str!r}={eos_id}, PAD={pad_id}")

    try:
        encoder_id, decoder_id = _task_xcoder_ids(opts, src, tgt)
    except ValueError:
        encoder_id, decoder_id = _infer_xcoder_ids_from_shards(ckpt_dir, prefix, src, tgt)
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
    hf_sd, enc_depth, dec_depth = assemble_state_dict(ckpt_dir, prefix, src, tgt, encoder_id, decoder_id)
    # inv_freq is a computed buffer (not a trained weight); strip it so load_state_dict
    # doesn't warn about unexpected keys — it will be recomputed from config on init.
    hf_sd = {k: v for k, v in hf_sd.items() if not k.endswith('rotary_pos_emb.inv_freq')}
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
        enc_layers_override=enc_depth,
        dec_layers_override=dec_depth,
        bos_token_id=bos_id,
        eos_token_id=eos_id,
        decoder_start_token_id=bos_id,
        pad_token_id=pad_id,
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

    def _save_tokenizer(vocab_obj, subdir: str, label: str, add_bos_eos: bool = False):
        specials = vocab_obj.specials
        _rev = {v: k for k, v in specials.items()}
        tok = PreTrainedTokenizerFast(
            tokenizer_object=vocab_obj.tokenizer,
            bos_token=_rev.get(config.bos_token_id, bos_str),
            eos_token=_rev.get(config.eos_token_id, eos_str),
            unk_token=_rev.get(getattr(config, 'unk_token_id', 3), '<unk>'),
            pad_token=_rev.get(config.pad_token_id, DefaultTokens.PAD),
        )
        if add_bos_eos:
            # Mirror Mammoth's _maybe_numericalize: source is always fed as [BOS, tokens..., EOS].
            from tokenizers.processors import TemplateProcessing
            tok._tokenizer.post_processor = TemplateProcessing(
                single=f"{bos_str}:0 $A:0 {eos_str}:0",
                special_tokens=[(bos_str, bos_id), (eos_str, eos_id)],
            )
        save_path = os.path.join(output_dir, subdir)
        os.makedirs(save_path, exist_ok=True)
        tok.save_pretrained(save_path)
        print(f"{label} tokenizer saved → {subdir}/ ({len(tok)} tokens)"
              + (" [with BOS/EOS post-processor]" if add_bos_eos else ""))

    _save_tokenizer(src_vocab_obj, config.src_tokenizer_dir, 'src', add_bos_eos=True)
    _save_tokenizer(tgt_vocab_obj, config.tgt_tokenizer_dir, 'tgt')

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


def _discover_tasks(opts) -> list[tuple[str, str]]:
    """Return [(src, tgt), ...] for every task in opts.tasks, preserving definition order."""
    tasks = getattr(opts, 'tasks', None) or {}
    pairs = []
    for corpus_opts in tasks.values():
        src_tgt = corpus_opts.get('src_tgt', '')
        src, _, tgt = src_tgt.partition('-')
        if src and tgt:
            pairs.append((src, tgt))
    return pairs


def convert_multi_task_artifact(
    ckpt_dir: str,
    output_dir: str,
    step: int | None = None,
) -> None:
    """Convert all tasks from a multi-task checkpoint into a single bundled directory.

    Each task is converted via the existing convert() into a temp dir, then the
    weight files are renamed to {task}.safetensors and collected into output_dir
    alongside shared code files and a manifest config.json.
    """
    import json
    import tempfile

    prefix = resolve_prefix(ckpt_dir, step)
    frame = _load(os.path.join(ckpt_dir, f'{prefix}_frame.pt'))
    all_pairs = _discover_tasks(frame['opts'])
    if not all_pairs:
        raise ValueError(f"No tasks found in checkpoint opts at {ckpt_dir}")

    os.makedirs(output_dir, exist_ok=True)
    task_manifest = {}

    with tempfile.TemporaryDirectory(prefix="mammoth_convert_") as tmp:
        for src, tgt in all_pairs:
            task_name = f"{src}-{tgt}"
            task_dir = os.path.join(tmp, task_name)
            print(f"\n=== Converting task {task_name} ===")
            convert(ckpt_dir, prefix, src, tgt, task_dir)

            # Rename model.safetensors → {task_name}.safetensors in output_dir
            src_sf = os.path.join(task_dir, "model.safetensors")
            dst_sf = os.path.join(output_dir, f"{task_name}.safetensors")
            if os.path.exists(src_sf):
                shutil.move(src_sf, dst_sf)
            else:
                # Try .bin as fallback
                src_bin = os.path.join(task_dir, "pytorch_model.bin")
                if os.path.exists(src_bin):
                    dst_bin = os.path.join(output_dir, f"{task_name}.bin")
                    shutil.move(src_bin, dst_bin)

            # Read per-task config to extract task-specific info for the manifest
            task_config_path = os.path.join(task_dir, "config.json")
            with open(task_config_path) as f:
                task_config = json.load(f)

            # Store the FULL per-task config (has all architecture params)
            # plus convenience fields for the wrapper.
            task_manifest[task_name] = {
                **task_config,
                "_src": src,
                "_tgt": tgt,
            }

            # Copy per-task tokenizers into output_dir/{task_name}_src_tokenizer/
            for tok_side in ("src_tokenizer", "tgt_tokenizer"):
                tok_subdir = task_config.get(
                    f"{tok_side}_dir", f"{tok_side}"
                )
                tok_src = os.path.join(task_dir, tok_subdir)
                tok_dst = os.path.join(output_dir, f"{task_name}_{tok_side}")
                if os.path.isdir(tok_src) and not os.path.exists(tok_dst):
                    shutil.copytree(tok_src, tok_dst)

    # Vendor shared code files once
    here = os.path.dirname(os.path.abspath(__file__))
    for fname in ("configuration_mammoth.py", "modeling_mammoth.py", "mammoth_hub.py"):
        shutil.copy(os.path.join(here, fname), os.path.join(output_dir, fname))
    import mammoth.x_transformers as _mxt
    xt_src_dir = os.path.dirname(_mxt.__file__)
    for fname in ("x_transformers.py", "attend.py", "autoregressive_wrapper.py"):
        shutil.copy(os.path.join(xt_src_dir, fname), os.path.join(output_dir, fname))

    # Write manifest config.json
    manifest = {
        "model_type": "mammoth_hub",
        "tasks": task_manifest,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBundled {len(task_manifest)} tasks into {output_dir}")
    for t in task_manifest:
        print(f"  {t}")



def convert_checkpoint(
    ckpt_dir: str,
    output_dir: str,
    task: str | None = None,
    step: int | None = None,
) -> None:
    """Discover tasks from the checkpoint's stored opts and convert each one.

    - task=None on a multi-task checkpoint  → convert every task into output_dir/<src>-<tgt>/
    - task=None on a single-task checkpoint → convert flat into output_dir/
    - task="src-tgt"                        → convert only that task; flat layout
                                              (errors if pair is not in opts.tasks)
    """
    prefix = resolve_prefix(ckpt_dir, step)
    frame = _load(os.path.join(ckpt_dir, f'{prefix}_frame.pt'))
    all_pairs = _discover_tasks(frame['opts'])
    if not all_pairs:
        raise ValueError(
            f"No tasks found in checkpoint opts at {ckpt_dir}. "
            "Checkpoint must contain opts.tasks with at least one src_tgt entry."
        )

    if task is not None:
        src, _, tgt = task.partition('-')
        if not src or not tgt:
            raise ValueError(f"--task must be of the form 'SRC-TGT', got: {task!r}")
        if (src, tgt) not in all_pairs:
            available = [f"{s}-{t}" for s, t in all_pairs]
            raise ValueError(
                f"Task {task!r} not found in checkpoint. Available tasks: {available}"
            )
        pairs = [(src, tgt)]
    else:
        pairs = all_pairs

    # Subdir layout is determined by the checkpoint's task count, not the filtered
    # selection: this keeps the output layout stable across invocations so a user can
    # convert tasks one at a time into the same output dir without collisions.
    use_subdirs = len(all_pairs) > 1
    for src, tgt in pairs:
        out = os.path.join(output_dir, f"{src}-{tgt}") if use_subdirs else output_dir
        print(f"\n=== Converting task {src}-{tgt} → {out} ===")
        convert(ckpt_dir, prefix, src, tgt, out)


def main():
    parser = argparse.ArgumentParser(description="Convert Mammoth checkpoint to HuggingFace format")
    parser.add_argument('--checkpoint-dir', required=True,
                        help="Directory containing Mammoth shard files")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for HF model(s). For multi-task checkpoints, "
                             "one subdirectory per task is created (e.g. eng-spa/, eng-fra/).")
    parser.add_argument('--task', default=None,
                        help="Convert only this src-tgt pair (e.g. 'eng-spa'). "
                             "Default: convert every task found in the checkpoint.")
    parser.add_argument('--step', type=int, default=None,
                        help="Load a specific checkpoint step (e.g. --step 500). "
                             "Default: best checkpoint, or last step if no best exists.")
    parser.add_argument('--single-artifact', action='store_true',
                        help="Bundle all tasks into a single HF directory with per-task shards.")

    args = parser.parse_args()

    if args.single_artifact:
        convert_multi_task_artifact(
            ckpt_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            step=args.step,
        )
    else:
        convert_checkpoint(
            ckpt_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            task=args.task,
            step=args.step,
        )


if __name__ == '__main__':
    main()
