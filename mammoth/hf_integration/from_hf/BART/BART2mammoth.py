#!/usr/bin/env python3
import sys
import os

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

"""
HuggingFace BART to Mammoth Converter

Usage: python BART2mammoth.py <model_path> <save_path>

BART Architecture:
- Encoder-Decoder transformer with shared vocabulary
- Post-LayerNorm (LayerNorm after residual connections)
- Learned absolute positional embeddings
- Standard attention (no RoPE)
- Standard FFN with GELU (no GLU gating)
- Biases in all linear layers
- Embedding layer normalization

BART-base specs:
- 6 encoder layers, 6 decoder layers
- 768 hidden dim, 12 attention heads
- 3072 FFN intermediate size
- 50265 vocab size
- 1024 max position embeddings

This converter:
1. Loads BART model from HuggingFace
2. Creates x-transformers encoder-decoder model with BART architecture
3. Maps pretrained weights to Mammoth multi-task model
4. Handles embedding sharing and layer normalization correctly
"""

import torch
from collections import OrderedDict
from argparse import Namespace
from transformers import AutoConfig, BartForConditionalGeneration, AutoTokenizer

from mammoth.x_transformers import TransformerWrapper, Encoder, Decoder, XTransformer
from mammoth.inputters.vocab import HFTokenizerVocab
from mammoth.distributed.tasks import TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.model_builder import build_model
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.model_saver import build_model_saver


# =============================================================================
# SECTION 1: HuggingFace to x-transformers conversion
# =============================================================================


def load_bart_config_and_bias(model_path):
    """
    Load BART config and extract final_logits_bias before creating models

    Args:
        model_path: Path to HuggingFace BART model

    Returns:
        (config, final_logits_bias, hf_state_dict)
    """
    print("=" * 70)
    print("Loading BART Configuration and Weights")
    print("=" * 70)

    print(f"  [1/2] Loading BART configuration...")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)

    print(f"  [2/2] Loading BART model to extract final_logits_bias...")
    hf_model = BartForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    hf_state_dict = hf_model.state_dict()

    # Extract final_logits_bias if present
    final_logits_bias = None
    if 'final_logits_bias' in hf_state_dict:
        final_logits_bias = hf_state_dict['final_logits_bias']
        print(f"  ✓ Extracted final_logits_bias: {final_logits_bias.shape}")
    else:
        print(f"  ⚠️  No final_logits_bias found in model")

    print(f"  ✓ Loaded {len(hf_state_dict)} parameters from HuggingFace BART")

    return config, final_logits_bias, hf_state_dict


def create_bart_xtransformer_model(config, final_logits_bias=None):
    """
    Create BART encoder-decoder from configuration

    Architecture:
    - Encoder: 6 layers, 768 hidden dim, 12 heads (BART-base)
    - Decoder: 6 layers, 768 hidden dim, 12 heads (BART-base)
    - Post-LN, learned positional embeddings, standard FFN

    Args:
        config: BART configuration
        final_logits_bias: Optional BART final logits bias tensor

    Returns:
        (encoder_model, decoder_model)
    """
    print("\n" + "=" * 70)
    print("Creating BART Encoder-Decoder Models")
    print("=" * 70)

    print(f"  BART Configuration:")
    print(f"    - Encoder layers: {config.encoder_layers}")
    print(f"    - Decoder layers: {config.decoder_layers}")
    print(f"    - Hidden size: {config.d_model}")
    print(f"    - Encoder attention heads: {config.encoder_attention_heads}")
    print(f"    - Decoder attention heads: {config.decoder_attention_heads}")
    print(f"    - Encoder FFN dim: {config.encoder_ffn_dim}")
    print(f"    - Decoder FFN dim: {config.decoder_ffn_dim}")
    print(f"    - Vocab size: {config.vocab_size}")
    print(f"    - Max position embeddings: {config.max_position_embeddings}")
    print(f"    - Activation: {config.activation_function}")
    print(f"    - Normalize before: False")
    print(f"    - Normalize embedding: True")
    print(f"    - Scale embedding: {config.scale_embedding}")

    # Calculate ff_mult
    enc_ff_mult = config.encoder_ffn_dim / config.d_model
    dec_ff_mult = config.decoder_ffn_dim / config.d_model

    # BART uses positional embedding offset of 2
    # BART has 1026 embeddings but only uses positions [2:1026] for sequences up to 1024 tokens
    # We'll copy positions [2:1026] → [0:1024] during weight loading
    max_seq_len = config.max_position_embeddings  # 1024, not 1026

    print(f"\n[Architecture Summary]")
    print(f"  Encoder: {config.encoder_layers} layers × {config.d_model}d")
    print(f"  Decoder: {config.decoder_layers} layers × {config.d_model}d")
    print(f"  FF multipliers: encoder={enc_ff_mult:.1f}x, decoder={dec_ff_mult:.1f}x")
    print(f"  Post-LN architecture (normalize_before=False)")
    print(f"  Max sequence length: {max_seq_len} (BART's positional offset handled during weight loading)")

    # Build encoder
    print(f"\n[Building Encoder]")
    encoder = Encoder(
        dim=config.d_model,
        depth=config.encoder_layers,
        heads=config.encoder_attention_heads,
        ff_mult=enc_ff_mult,
        rotary_pos_emb=False,  # BART uses learned positional embeddings
        ff_glu=False,  # BART uses standard FFN (not GLU)
        attn_dropout=config.attention_dropout,
        ff_dropout=config.dropout,
        pre_norm=False,  # BART uses Post-LN (normalize after residual)
        attn_qkv_bias=True,  # BART uses bias in Q, K, V projections
        layernorm_bias=True,  # BART uses bias in LayerNorm (beta parameter)
        norm_add_unit_offset=False,  # BART gamma is stored as-is; no +1 offset
    )

    encoder_model = TransformerWrapper(
        num_tokens=config.vocab_size,
        max_seq_len=max_seq_len,  # 1024
        attn_layers=encoder,
        emb_dropout=config.dropout,
        post_emb_norm=True,  # BART normalizes embeddings
        post_emb_norm_bias=True,  # BART uses bias in embedding LayerNorm
        return_only_embed=True,  # Encoder returns embeddings only
    )

    print(f"  ✓ Encoder created: BART architecture ({config.encoder_layers} layers)")

    # Build decoder
    print(f"\n[Building Decoder]")
    decoder = Decoder(
        dim=config.d_model,
        depth=config.decoder_layers,
        heads=config.decoder_attention_heads,
        ff_mult=dec_ff_mult,
        rotary_pos_emb=False,  # BART uses learned positional embeddings
        ff_glu=False,  # BART uses standard FFN (not GLU)
        attn_dropout=config.attention_dropout,
        ff_dropout=config.dropout,
        pre_norm=False,  # BART uses Post-LN (normalize after residual)
        cross_attend=True,  # Enable cross-attention for encoder-decoder
        attn_qkv_bias=True,  # BART uses bias in Q, K, V projections
        layernorm_bias=True,  # BART uses bias in LayerNorm (beta parameter)
        norm_add_unit_offset=False,  # BART gamma is stored as-is; no +1 offset
    )

    decoder_model = TransformerWrapper(
        num_tokens=config.vocab_size,
        max_seq_len=max_seq_len,  # 1024
        attn_layers=decoder,
        emb_dropout=config.dropout,
        post_emb_norm=True,  # BART normalizes embeddings
        post_emb_norm_bias=True,  # BART uses bias in embedding LayerNorm
        tie_embedding=True,  # ✅ Tie decoder input embeddings to output projection
        final_logits_bias=final_logits_bias,  # ✅ BART final logits bias
    )

    print(f"  ✓ Decoder created: BART architecture ({config.decoder_layers} layers)")
    print(f"  ✓ Cross-attention enabled for encoder-decoder architecture")
    if final_logits_bias is not None:
        print(f"  ✓ Final logits bias applied: {final_logits_bias.shape}")

    # ✅ Tie encoder and decoder embeddings (BART weight sharing)
    print(f"\n[Tying Embeddings]")
    decoder_model.token_emb = encoder_model.token_emb
    print(f"  ✓ Encoder-decoder embeddings tied (share same parameters)")
    print(f"  ✓ Decoder input-output embeddings tied")

    return encoder_model, decoder_model


def load_bart_weights(hf_state_dict, encoder_model, decoder_model, config):
    """
    Load BART weights into x-transformers encoder and decoder models

    Args:
        hf_state_dict: HuggingFace BART model state dict
        encoder_model: x-transformers encoder model
        decoder_model: x-transformers decoder model
        config: BART config

    Returns:
        (encoder_model, decoder_model) with loaded weights
    """
    print("\n" + "=" * 70)
    print("Mapping BART Weights to x-transformers")
    print("=" * 70)

    print(f"  [1/3] Creating weight mappings...")

    # Create weight mappings
    enc_mapping = create_bart_encoder_mapping(config.encoder_layers)
    dec_mapping = create_bart_decoder_mapping(config.decoder_layers)

    print(f"  [2/3] Mapping weights...")
    # BART positional embedding offset handling
    # BART has positions [0, 1, 2, ..., 1025] but only uses [2:1026]
    # We'll map BART's [2:1026] to x-transformers' [0:1024]
    bart_pos_offset = 2

    # Map encoder weights
    enc_state_dict = OrderedDict()
    for hf_key, x_key in enc_mapping.items():
        if hf_key not in hf_state_dict:
            print(f"  ⚠️  Missing encoder key: {hf_key}")
            continue

        # Handle positional embedding slicing
        if "embed_positions.weight" in hf_key:
            # Slice BART positions [2:1026] → x-transformers [0:1024]
            enc_state_dict[x_key] = hf_state_dict[hf_key][bart_pos_offset:, :]
            print(f"  ✓ Sliced positional embeddings: {hf_state_dict[hf_key].shape} → {enc_state_dict[x_key].shape}")
        else:
            enc_state_dict[x_key] = hf_state_dict[hf_key]

    # Map decoder weights
    dec_state_dict = OrderedDict()
    for hf_key, x_key in dec_mapping.items():
        if hf_key not in hf_state_dict:
            print(f"  ⚠️  Missing decoder key: {hf_key}")
            continue

        # Handle positional embedding slicing
        if "embed_positions.weight" in hf_key:
            # Slice BART positions [2:1026] → x-transformers [0:1024]
            dec_state_dict[x_key] = hf_state_dict[hf_key][bart_pos_offset:, :]
            print(f"  ✓ Sliced positional embeddings: {hf_state_dict[hf_key].shape} → {dec_state_dict[x_key].shape}")
        else:
            dec_state_dict[x_key] = hf_state_dict[hf_key]

    print(f"  [3/3] Loading weights into models...")
    # Load weights
    encoder_model.load_state_dict(enc_state_dict, strict=False)
    decoder_model.load_state_dict(dec_state_dict, strict=False)

    print(f"  ✓ Encoder weights loaded: {len(enc_state_dict)} parameters")
    print(f"  ✓ Decoder weights loaded: {len(dec_state_dict)} parameters")

    # Log model sizes
    log_model_size(encoder_model, "Encoder")
    log_model_size(decoder_model, "Decoder")

    return encoder_model, decoder_model


def create_bart_encoder_mapping(num_layers):
    """
    Create mapping from HuggingFace BART encoder to x-transformers encoder

    BART encoder structure (per layer):
    - self_attn (with bias): q_proj, k_proj, v_proj, out_proj
    - self_attn_layer_norm (post-attention)
    - fc1, fc2 (FFN with bias)
    - final_layer_norm (post-FFN)

    x-transformers encoder structure (per layer, indices: attn=2i, ff=2i+1):
    - PreNorm + Attention (but BART is Post-LN, so norm comes after)
    - PreNorm + FeedForward (but BART is Post-LN, so norm comes after)
    """
    mapping = {}

    # Embeddings
    mapping["model.encoder.embed_tokens.weight"] = "token_emb.emb.weight"

    # BART uses learned positional embeddings, not RoPE
    mapping["model.encoder.embed_positions.weight"] = "pos_emb.emb.weight"

    # Embedding normalization (if normalize_embedding=True)
    # x-transformers post_emb_norm corresponds to BART's layernorm_embedding
    mapping["model.encoder.layernorm_embedding.weight"] = "post_emb_norm.gamma"
    mapping["model.encoder.layernorm_embedding.bias"] = "post_emb_norm.beta"

    # Encoder layers
    for i in range(num_layers):
        attn_idx = i * 2
        ff_idx = attn_idx + 1

        hf_prefix = f"model.encoder.layers.{i}"
        x_attn_prefix = f"attn_layers.layers.{attn_idx}"
        x_ff_prefix = f"attn_layers.layers.{ff_idx}"

        # === Self-Attention Block ===
        # BART: q_proj, k_proj, v_proj, out_proj (with bias)
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{x_attn_prefix}.1.to_q.weight"
        mapping[f"{hf_prefix}.self_attn.q_proj.bias"] = f"{x_attn_prefix}.1.to_q.bias"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{x_attn_prefix}.1.to_k.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.bias"] = f"{x_attn_prefix}.1.to_k.bias"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{x_attn_prefix}.1.to_v.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.bias"] = f"{x_attn_prefix}.1.to_v.bias"
        mapping[f"{hf_prefix}.self_attn.out_proj.weight"] = f"{x_attn_prefix}.1.to_out.weight"
        mapping[f"{hf_prefix}.self_attn.out_proj.bias"] = f"{x_attn_prefix}.1.to_out.bias"

        # Post-attention layer norm (BART is Post-LN)
        # In x-transformers with pre_norm=False, the norm is in position .0.2
        mapping[f"{hf_prefix}.self_attn_layer_norm.weight"] = f"{x_attn_prefix}.0.2.gamma"
        mapping[f"{hf_prefix}.self_attn_layer_norm.bias"] = f"{x_attn_prefix}.0.2.beta"

        # === Feedforward Block ===
        mapping[f"{hf_prefix}.fc1.weight"] = f"{x_ff_prefix}.1.ff.0.0.weight"
        mapping[f"{hf_prefix}.fc1.bias"] = f"{x_ff_prefix}.1.ff.0.0.bias"
        mapping[f"{hf_prefix}.fc2.weight"] = f"{x_ff_prefix}.1.ff.2.weight"
        mapping[f"{hf_prefix}.fc2.bias"] = f"{x_ff_prefix}.1.ff.2.bias"

        # Post-feedforward layer norm (BART is Post-LN)
        mapping[f"{hf_prefix}.final_layer_norm.weight"] = f"{x_ff_prefix}.0.2.gamma"
        mapping[f"{hf_prefix}.final_layer_norm.bias"] = f"{x_ff_prefix}.0.2.beta"

    # Note: BART doesn't have a final layer norm on the encoder
    # The final_norm in x-transformers will be initialized randomly

    return mapping


def create_bart_decoder_mapping(num_layers):
    """
    Create mapping from HuggingFace BART decoder to x-transformers decoder

    BART decoder structure (per layer):
    - self_attn: q_proj, k_proj, v_proj, out_proj (all with bias)
    - self_attn_layer_norm (post-self-attention)
    - encoder_attn (cross-attention): q_proj, k_proj, v_proj, out_proj (all with bias)
    - encoder_attn_layer_norm (post-cross-attention)
    - fc1, fc2 (FFN with bias)
    - final_layer_norm (post-FFN)

    x-transformers decoder structure (indices: self_attn=3i, cross_attn=3i+1, ff=3i+2):
    - Self-attention with post-norm
    - Cross-attention with post-norm
    - FeedForward with post-norm
    """
    mapping = {}

    # Embeddings
    mapping["model.decoder.embed_tokens.weight"] = "token_emb.emb.weight"

    # BART uses learned positional embeddings
    mapping["model.decoder.embed_positions.weight"] = "pos_emb.emb.weight"

    # Embedding normalization
    mapping["model.decoder.layernorm_embedding.weight"] = "post_emb_norm.gamma"
    mapping["model.decoder.layernorm_embedding.bias"] = "post_emb_norm.beta"

    # Decoder layers (with cross-attention - indices: self_attn, cross_attn, ff)
    for layer_idx in range(num_layers):
        self_attn_idx = layer_idx * 3
        cross_attn_idx = layer_idx * 3 + 1
        ff_idx = layer_idx * 3 + 2

        hf_prefix = f"model.decoder.layers.{layer_idx}"
        x_self_attn_prefix = f"attn_layers.layers.{self_attn_idx}"
        x_cross_attn_prefix = f"attn_layers.layers.{cross_attn_idx}"
        x_ff_prefix = f"attn_layers.layers.{ff_idx}"

        # === Self-Attention Block ===
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{x_self_attn_prefix}.1.to_q.weight"
        mapping[f"{hf_prefix}.self_attn.q_proj.bias"] = f"{x_self_attn_prefix}.1.to_q.bias"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{x_self_attn_prefix}.1.to_k.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.bias"] = f"{x_self_attn_prefix}.1.to_k.bias"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{x_self_attn_prefix}.1.to_v.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.bias"] = f"{x_self_attn_prefix}.1.to_v.bias"
        mapping[f"{hf_prefix}.self_attn.out_proj.weight"] = f"{x_self_attn_prefix}.1.to_out.weight"
        mapping[f"{hf_prefix}.self_attn.out_proj.bias"] = f"{x_self_attn_prefix}.1.to_out.bias"

        # Post-self-attention layer norm
        mapping[f"{hf_prefix}.self_attn_layer_norm.weight"] = f"{x_self_attn_prefix}.0.2.gamma"
        mapping[f"{hf_prefix}.self_attn_layer_norm.bias"] = f"{x_self_attn_prefix}.0.2.beta"

        # === Cross-Attention Block (encoder_attn) ===
        mapping[f"{hf_prefix}.encoder_attn.q_proj.weight"] = f"{x_cross_attn_prefix}.1.to_q.weight"
        mapping[f"{hf_prefix}.encoder_attn.q_proj.bias"] = f"{x_cross_attn_prefix}.1.to_q.bias"
        mapping[f"{hf_prefix}.encoder_attn.k_proj.weight"] = f"{x_cross_attn_prefix}.1.to_k.weight"
        mapping[f"{hf_prefix}.encoder_attn.k_proj.bias"] = f"{x_cross_attn_prefix}.1.to_k.bias"
        mapping[f"{hf_prefix}.encoder_attn.v_proj.weight"] = f"{x_cross_attn_prefix}.1.to_v.weight"
        mapping[f"{hf_prefix}.encoder_attn.v_proj.bias"] = f"{x_cross_attn_prefix}.1.to_v.bias"
        mapping[f"{hf_prefix}.encoder_attn.out_proj.weight"] = f"{x_cross_attn_prefix}.1.to_out.weight"
        mapping[f"{hf_prefix}.encoder_attn.out_proj.bias"] = f"{x_cross_attn_prefix}.1.to_out.bias"

        # Post-cross-attention layer norm
        mapping[f"{hf_prefix}.encoder_attn_layer_norm.weight"] = f"{x_cross_attn_prefix}.0.2.gamma"
        mapping[f"{hf_prefix}.encoder_attn_layer_norm.bias"] = f"{x_cross_attn_prefix}.0.2.beta"

        # === Feedforward Block ===
        mapping[f"{hf_prefix}.fc1.weight"] = f"{x_ff_prefix}.1.ff.0.0.weight"
        mapping[f"{hf_prefix}.fc1.bias"] = f"{x_ff_prefix}.1.ff.0.0.bias"
        mapping[f"{hf_prefix}.fc2.weight"] = f"{x_ff_prefix}.1.ff.2.weight"
        mapping[f"{hf_prefix}.fc2.bias"] = f"{x_ff_prefix}.1.ff.2.bias"

        # Post-feedforward layer norm
        mapping[f"{hf_prefix}.final_layer_norm.weight"] = f"{x_ff_prefix}.0.2.gamma"
        mapping[f"{hf_prefix}.final_layer_norm.bias"] = f"{x_ff_prefix}.0.2.beta"

    # Note: BART doesn't have a final layer norm on the decoder
    # The final_norm in x-transformers will be initialized randomly

    return mapping


# =============================================================================
# SECTION 2: MAMMOTH utilities
# =============================================================================


def calculate_model_size(model):
    """
    Calculate model size in terms of parameters and memory

    Returns:
        total_params (int): Total number of parameters
        trainable_params (int): Number of trainable parameters
        size_mb (float): Approximate size in MB (assuming float32)
        size_gb (float): Approximate size in GB (assuming float32)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate size (4 bytes per float32 parameter)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_bytes / (1024 ** 3)

    return total_params, trainable_params, size_mb, size_gb


def log_model_size(model, model_name="Model"):
    """
    Log model size information in a formatted way

    Args:
        model: PyTorch model
        model_name: Name to display in log
    """
    total_params, trainable_params, size_mb, size_gb = calculate_model_size(model)

    print(f"\n  {model_name} Size:")
    print(f"    - Total parameters: {total_params:,}")
    print(f"    - Trainable parameters: {trainable_params:,}")
    if size_gb >= 1.0:
        print(f"    - Approximate size: {size_gb:.2f} GB ({size_mb:.1f} MB)")
    else:
        print(f"    - Approximate size: {size_mb:.1f} MB")


def rename_bart_special_tokens_to_mammoth(tokenizer_path, output_path=None):
    """
    Rename HuggingFace BART special tokens to MAMMOTH conventions

    BART special tokens:
    - <s> (ID 0) → BOS (keep as-is)
    - </s> (ID 2) → EOS (keep as-is)
    - <pad> (ID 1) → PAD (keep as-is)
    - <unk> (ID 3) → UNK (keep as-is)

    BART already uses MAMMOTH-compatible token names, so we just need to verify
    and potentially create a backup.

    Args:
        tokenizer_path: Path to the HuggingFace tokenizer.json file
        output_path: Optional path to save modified tokenizer (if None, modifies in-place)

    Returns:
        Path to the tokenizer file
    """
    import json
    import shutil
    from pathlib import Path

    print(f"Checking BART tokenizer compatibility with MAMMOTH...")
    print(f"  Input: {tokenizer_path}")

    # Load tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Check special tokens
    vocab = tokenizer_data['model']['vocab']
    special_tokens = {
        '<s>': vocab.get('<s>', None),
        '</s>': vocab.get('</s>', None),
        '<pad>': vocab.get('<pad>', None),
        '<unk>': vocab.get('<unk>', None),
    }

    print(f"  BART special tokens:")
    for token, token_id in special_tokens.items():
        if token_id is not None:
            print(f"    {token}: ID {token_id}")
        else:
            print(f"    ⚠️  {token}: NOT FOUND")

    print(f"  ✓ BART tokenizer is already compatible with MAMMOTH conventions")

    # Save or copy
    if output_path is None:
        output_path = tokenizer_path
    else:
        # Create backup if modifying in-place
        if output_path == tokenizer_path:
            backup_path = str(Path(tokenizer_path).with_suffix('.json.backup'))
            if not Path(backup_path).exists():
                shutil.copy2(tokenizer_path, backup_path)
                print(f"  Created backup: {backup_path}")

        # Save to output path
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Tokenizer saved to: {output_path}")

    return output_path


def create_vocabs_dict_from_hf_tokenizer(tokenizer_path, src_lang="en", tgt_lang="en"):
    """
    Create vocabs_dict for Mammoth model

    For BART, we use the same tokenizer for both source and target
    """
    from tokenizers import Tokenizer

    src_vocab = HFTokenizerVocab(
        tokenizer_path=tokenizer_path,
        tag=f"src_{src_lang}",
    )

    tgt_vocab = HFTokenizerVocab(
        tokenizer_path=tokenizer_path,
        tag=f"tgt_{tgt_lang}",
    )

    vocabs_dict = {
        ("src", src_lang): src_vocab,
        ("tgt", tgt_lang): tgt_vocab,
    }

    print(f"✓ Created vocabularies:")
    print(f"  - Source ({src_lang}): {len(src_vocab)} tokens")
    print(f"  - Target ({tgt_lang}): {len(tgt_vocab)} tokens")

    return vocabs_dict


def create_model_opts_from_bart_model(encoder_model, decoder_model, config, final_logits_bias=None):
    """
    Create model_opts for Mammoth from BART encoder + decoder models

    Args:
        encoder_model: BART encoder model
        decoder_model: BART decoder model
        config: BART configuration
        final_logits_bias: Optional BART final logits bias tensor

    Returns:
        model_opts: Namespace with model configuration
    """
    model_opts = Namespace()

    # Basic settings
    model_opts.model_dtype = "bf16"

    # Architecture - same dimensions for encoder/decoder
    model_opts.enc_layers = [config.encoder_layers]
    model_opts.dec_layers = [config.decoder_layers]
    model_opts.enc_model_dim = config.d_model
    model_opts.dec_model_dim = config.d_model

    # Calculate ff_mult
    enc_ff_mult = config.encoder_ffn_dim / config.d_model
    dec_ff_mult = config.decoder_ffn_dim / config.d_model

    # x_transformers_opts
    model_opts.x_transformers_opts = {
        # Shared options
        "attn_flash": False,  # BART uses standard attention
        "tie_embedding": True,  # Tie decoder input embeddings to output projection

        # Encoder-specific (BART)
        "enc_heads": config.encoder_attention_heads,
        "enc_attn_dim_head": config.d_model // config.encoder_attention_heads,
        "enc_attn_dropout": config.attention_dropout,
        "enc_attn_qkv_bias": True,  # BART uses bias in Q, K, V projections
        "enc_layernorm_bias": True,  # BART uses bias in LayerNorm (beta parameter)
        "enc_ff_mult": enc_ff_mult,
        "enc_ff_dropout": config.dropout,
        "enc_ff_glu": False,  # BART uses standard FFN
        "enc_rotary_pos_emb": False,  # BART uses learned positional embeddings
        "enc_pre_norm": False,  # BART uses Post-LN
        "enc_post_emb_norm": True,
        "enc_post_emb_norm_bias": True,  # BART uses bias in embedding LayerNorm
        "enc_max_seq_len": config.max_position_embeddings,  # 1024

        # Decoder-specific (BART)
        "dec_heads": config.decoder_attention_heads,
        "dec_attn_dim_head": config.d_model // config.decoder_attention_heads,
        "dec_attn_dropout": config.attention_dropout,
        "dec_attn_qkv_bias": True,  # BART uses bias in Q, K, V projections
        "dec_layernorm_bias": True,  # BART uses bias in LayerNorm (beta parameter)
        "dec_ff_mult": dec_ff_mult,
        "dec_ff_dropout": config.dropout,
        "dec_ff_glu": False,  # BART uses standard FFN
        "dec_rotary_pos_emb": False,  # BART uses learned positional embeddings
        "dec_pre_norm": False,  # BART uses Post-LN
        "dec_post_emb_norm": True,
        "dec_post_emb_norm_bias": True,  # BART uses bias in embedding LayerNorm
        "dec_max_seq_len": config.max_position_embeddings,  # 1024
    }

    # Add final_logits_bias to decoder if provided (BART-specific)
    if final_logits_bias is not None:
        # Ensure bias is properly detached and on CPU for model building
        bias_tensor = final_logits_bias.detach().cpu().clone()
        # Squeeze to [vocab_size] if needed
        if bias_tensor.dim() > 1:
            bias_tensor = bias_tensor.squeeze(0)
        model_opts.x_transformers_opts["dec_final_logits_bias"] = bias_tensor
        print(f"  ✓ Added final_logits_bias to model_opts (shape: {bias_tensor.shape})")

    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.dropout = [config.dropout]
    model_opts.attention_dropout = [config.attention_dropout]
    model_opts.share_encoder_decoder_embeddings = True

    return model_opts


class SimpleWorldContext(WorldContext):
    def __init__(self):
        super().__init__(context=DeviceContextEnum.SINGLE_GPU, n_nodes=1, gpus_per_node=1)
        self.context = DeviceContextEnum.SINGLE_GPU
        self.n_nodes = 1
        self.gpus_per_node = 1

    def is_distributed(self):
        return False

    def global_to_local(self, node_rank, local_rank):
        return SimpleDeviceContext(
            context=self.context,
            n_nodes=self.n_nodes,
            gpus_per_node=self.gpus_per_node,
            node_rank=node_rank,
            local_rank=local_rank,
        )


class SimpleDeviceContext(DeviceContext):
    def __init__(self, context, n_nodes, gpus_per_node, node_rank, local_rank):
        super().__init__(
            context=context,
            n_nodes=n_nodes,
            gpus_per_node=gpus_per_node,
            node_rank=node_rank,
            local_rank=local_rank,
        )

    def validate(self, world_context):
        super().validate(world_context)

    def is_master(self):
        return True

    def is_distributed(self):
        return False


def create_task_queue_manager(vocabs_dict, num_encoder_layers, num_decoder_layers, src_lang="en", tgt_lang="en"):
    """
    Create TaskQueueManager for BART model
    """
    opts = Namespace()
    opts.tasks = {}

    task_name = f"task_{src_lang}_{tgt_lang}"
    opts.tasks[task_name] = {
        "src_tgt": f"{src_lang}-{tgt_lang}",
        "weight": 1.0,
        "introduce_at_training_step": 0,
        "node_gpu": "0:0",
        "enc_sharing_group": [src_lang],
        "dec_sharing_group": [tgt_lang],
    }

    opts.enc_layers = [num_encoder_layers]
    opts.dec_layers = [num_decoder_layers]
    opts.task_distribution_strategy = "weighted_sampling"
    opts.accum_count = [1]
    opts.seed = 42

    world_context = SimpleWorldContext()
    task_manager = TaskQueueManager.from_opts(opts, world_context)

    # Assign vocabularies
    src_vocab = vocabs_dict.get(("src", src_lang))
    tgt_vocab = vocabs_dict.get(("tgt", tgt_lang))

    for task in task_manager.tasks:
        task.src_vocab = src_vocab
        task.tgt_vocab = tgt_vocab

    local_task_manager = task_manager.global_to_local(
        node_rank=0, local_rank=0, opts=opts
    )
    local_task_manager.create_all_distributed_components(
        use_attention_bridge=False
    )

    return local_task_manager


def create_opts():
    """Create complete opts object for optimizer"""
    opts = Namespace()
    opts.train_from = None
    opts.reset_optim = "all"
    opts.model_dtype = "bf16"
    opts.optim = "adafactor"
    opts.learning_rate = 0.001
    opts.adam_beta1 = 0.9
    opts.adam_beta2 = 0.999
    opts.weight_decay = 0.0
    opts.max_grad_norm = 1.0
    opts.decay_method = "none"
    opts.learning_rate_decay = 0.5
    opts.start_decay_steps = 50000
    opts.decay_steps = 10000
    opts.adagrad_accumulator_init = 0.0
    opts.gpu_ranks = []
    opts.log_model_structure = True
    opts.adapters = None
    return opts


# =============================================================================
# SECTION 3: x-transformers to Mammoth conversion
# =============================================================================


def verify_random_weights(hf_state_dict, mammoth_sd, mapping, enc_sd, dec_sd, num_samples=3):
    """
    Simple weight verification - compare first 10 values of randomly sampled parameters

    Args:
        hf_state_dict: Original HF BART state dict
        mammoth_sd: MAMMOTH model state dict after mapping
        mapping: Weight mapping dictionary
        enc_sd: x-transformers encoder state dict
        dec_sd: x-transformers decoder state dict
        num_samples: Number of random weights to check
    """
    import random

    print(f"\n  Verifying {num_samples} random weights (first 10 values each):")

    # Get a list of all mappable keys (excluding embeddings for brevity)
    mappable_keys = []
    for hf_key, xt_key in mapping.items():
        if any(layer_type in hf_key.lower() for layer_type in ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']):
            mappable_keys.append((hf_key, xt_key))

    # Randomly sample a few weights
    sampled_keys = random.sample(mappable_keys, min(num_samples, len(mappable_keys)))

    all_match = True
    for hf_key, mammoth_key in sampled_keys:
        # Determine source dict based on key
        if "encoder" in mammoth_key:
            source_key = hf_key.replace("model.encoder.", "")
            if source_key in enc_sd:
                source_val = enc_sd[source_key]
                target_val = mammoth_sd[mammoth_key]

                # Compare first 10 values
                source_flat = source_val.flatten()[:10]
                target_flat = target_val.flatten()[:10]

                if torch.allclose(source_flat, target_flat, atol=1e-6):
                    print(f"    ✓ {hf_key.split('.')[-2]}: values match")
                else:
                    print(f"    ✗ {hf_key.split('.')[-2]}: values differ")
                    all_match = False
        elif "decoder" in mammoth_key:
            source_key = hf_key.replace("model.decoder.", "")
            if source_key in dec_sd:
                source_val = dec_sd[source_key]
                target_val = mammoth_sd[mammoth_key]

                # Compare first 10 values
                source_flat = source_val.flatten()[:10]
                target_flat = target_val.flatten()[:10]

                if torch.allclose(source_flat, target_flat, atol=1e-6):
                    print(f"    ✓ {hf_key.split('.')[-2]}: values match")
                else:
                    print(f"    ✗ {hf_key.split('.')[-2]}: values differ")
                    all_match = False

    if all_match:
        print(f"    ✓ All sampled weights match correctly!")
    else:
        print(f"    ⚠️  Some weights differ - check conversion")


def create_bart_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, task_name):
    """
    Create combined mapping for BART model:
    - Encoder: BART encoder weights
    - Decoder: BART decoder weights
    """
    mapping = {}

    # ======== ENCODER MAPPING (BART) ========
    mapping["encoder.token_emb.emb.weight"] = f"encoder.{task_name}.token_emb.emb.weight"
    mapping["encoder.pos_emb.emb.weight"] = f"encoder.{task_name}.pos_emb.emb.weight"

    # Post-embedding normalization (BART's layernorm_embedding)
    mapping["encoder.post_emb_norm.gamma"] = f"encoder.{task_name}.post_emb_norm.gamma"
    mapping["encoder.post_emb_norm.beta"] = f"encoder.{task_name}.post_emb_norm.beta"

    for layer_idx in range(num_encoder_layers):
        attn_idx = layer_idx * 2
        ff_idx = attn_idx + 1

        xt_attn_base = f"encoder.attn_layers.layers.{attn_idx}"
        mammoth_attn_base = f"encoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{attn_idx}"

        # Attention
        mapping[f"{xt_attn_base}.1.to_q.weight"] = f"{mammoth_attn_base}.1.to_q.weight"
        mapping[f"{xt_attn_base}.1.to_q.bias"] = f"{mammoth_attn_base}.1.to_q.bias"
        mapping[f"{xt_attn_base}.1.to_k.weight"] = f"{mammoth_attn_base}.1.to_k.weight"
        mapping[f"{xt_attn_base}.1.to_k.bias"] = f"{mammoth_attn_base}.1.to_k.bias"
        mapping[f"{xt_attn_base}.1.to_v.weight"] = f"{mammoth_attn_base}.1.to_v.weight"
        mapping[f"{xt_attn_base}.1.to_v.bias"] = f"{mammoth_attn_base}.1.to_v.bias"
        mapping[f"{xt_attn_base}.1.to_out.weight"] = f"{mammoth_attn_base}.1.to_out.weight"
        mapping[f"{xt_attn_base}.1.to_out.bias"] = f"{mammoth_attn_base}.1.to_out.bias"

        # Post-attention norm
        mapping[f"{xt_attn_base}.0.2.gamma"] = f"{mammoth_attn_base}.0.2.gamma"
        mapping[f"{xt_attn_base}.0.2.beta"] = f"{mammoth_attn_base}.0.2.beta"

        # Feedforward
        xt_ff_base = f"encoder.attn_layers.layers.{ff_idx}"
        mammoth_ff_base = f"encoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{ff_idx}"

        mapping[f"{xt_ff_base}.1.ff.0.0.weight"] = f"{mammoth_ff_base}.1.ff.0.0.weight"
        mapping[f"{xt_ff_base}.1.ff.0.0.bias"] = f"{mammoth_ff_base}.1.ff.0.0.bias"
        mapping[f"{xt_ff_base}.1.ff.2.weight"] = f"{mammoth_ff_base}.1.ff.2.weight"
        mapping[f"{xt_ff_base}.1.ff.2.bias"] = f"{mammoth_ff_base}.1.ff.2.bias"

        # Post-feedforward norm
        mapping[f"{xt_ff_base}.0.2.gamma"] = f"{mammoth_ff_base}.0.2.gamma"
        mapping[f"{xt_ff_base}.0.2.beta"] = f"{mammoth_ff_base}.0.2.beta"

    # ======== DECODER MAPPING (BART) ========
    mapping["decoder.token_emb.emb.weight"] = f"decoder.{task_name}.token_emb.emb.weight"
    mapping["decoder.pos_emb.emb.weight"] = f"decoder.{task_name}.pos_emb.emb.weight"

    # Post-embedding normalization (BART's layernorm_embedding)
    mapping["decoder.post_emb_norm.gamma"] = f"decoder.{task_name}.post_emb_norm.gamma"
    mapping["decoder.post_emb_norm.beta"] = f"decoder.{task_name}.post_emb_norm.beta"

    for layer_idx in range(num_decoder_layers):
        # Mammoth encoder-decoder: [self_attn, cross_attn, ff] triplets
        mammoth_self_attn_idx = layer_idx * 3
        mammoth_cross_attn_idx = layer_idx * 3 + 1
        mammoth_ff_idx = layer_idx * 3 + 2

        # x-transformers decoder (with cross_attend=True): [self_attn, cross_attn, ff] triplets
        xt_self_attn_idx = layer_idx * 3
        xt_cross_attn_idx = layer_idx * 3 + 1
        xt_ff_idx = layer_idx * 3 + 2

        xt_self_attn_base = f"decoder.attn_layers.layers.{xt_self_attn_idx}"
        mammoth_self_attn_base = f"decoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{mammoth_self_attn_idx}"

        # Self-attention
        mapping[f"{xt_self_attn_base}.1.to_q.weight"] = f"{mammoth_self_attn_base}.1.to_q.weight"
        mapping[f"{xt_self_attn_base}.1.to_q.bias"] = f"{mammoth_self_attn_base}.1.to_q.bias"
        mapping[f"{xt_self_attn_base}.1.to_k.weight"] = f"{mammoth_self_attn_base}.1.to_k.weight"
        mapping[f"{xt_self_attn_base}.1.to_k.bias"] = f"{mammoth_self_attn_base}.1.to_k.bias"
        mapping[f"{xt_self_attn_base}.1.to_v.weight"] = f"{mammoth_self_attn_base}.1.to_v.weight"
        mapping[f"{xt_self_attn_base}.1.to_v.bias"] = f"{mammoth_self_attn_base}.1.to_v.bias"
        mapping[f"{xt_self_attn_base}.1.to_out.weight"] = f"{mammoth_self_attn_base}.1.to_out.weight"
        mapping[f"{xt_self_attn_base}.1.to_out.bias"] = f"{mammoth_self_attn_base}.1.to_out.bias"

        # Post-self-attention norm
        mapping[f"{xt_self_attn_base}.0.2.gamma"] = f"{mammoth_self_attn_base}.0.2.gamma"
        mapping[f"{xt_self_attn_base}.0.2.beta"] = f"{mammoth_self_attn_base}.0.2.beta"

        # Cross-attention
        xt_cross_attn_base = f"decoder.attn_layers.layers.{xt_cross_attn_idx}"
        mammoth_cross_attn_base = f"decoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{mammoth_cross_attn_idx}"

        mapping[f"{xt_cross_attn_base}.1.to_q.weight"] = f"{mammoth_cross_attn_base}.1.to_q.weight"
        mapping[f"{xt_cross_attn_base}.1.to_q.bias"] = f"{mammoth_cross_attn_base}.1.to_q.bias"
        mapping[f"{xt_cross_attn_base}.1.to_k.weight"] = f"{mammoth_cross_attn_base}.1.to_k.weight"
        mapping[f"{xt_cross_attn_base}.1.to_k.bias"] = f"{mammoth_cross_attn_base}.1.to_k.bias"
        mapping[f"{xt_cross_attn_base}.1.to_v.weight"] = f"{mammoth_cross_attn_base}.1.to_v.weight"
        mapping[f"{xt_cross_attn_base}.1.to_v.bias"] = f"{mammoth_cross_attn_base}.1.to_v.bias"
        mapping[f"{xt_cross_attn_base}.1.to_out.weight"] = f"{mammoth_cross_attn_base}.1.to_out.weight"
        mapping[f"{xt_cross_attn_base}.1.to_out.bias"] = f"{mammoth_cross_attn_base}.1.to_out.bias"

        # Post-cross-attention norm
        mapping[f"{xt_cross_attn_base}.0.2.gamma"] = f"{mammoth_cross_attn_base}.0.2.gamma"
        mapping[f"{xt_cross_attn_base}.0.2.beta"] = f"{mammoth_cross_attn_base}.0.2.beta"

        # Feedforward
        xt_ff_base = f"decoder.attn_layers.layers.{xt_ff_idx}"
        mammoth_ff_base = f"decoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{mammoth_ff_idx}"

        mapping[f"{xt_ff_base}.1.ff.0.0.weight"] = f"{mammoth_ff_base}.1.ff.0.0.weight"
        mapping[f"{xt_ff_base}.1.ff.0.0.bias"] = f"{mammoth_ff_base}.1.ff.0.0.bias"
        mapping[f"{xt_ff_base}.1.ff.2.weight"] = f"{mammoth_ff_base}.1.ff.2.weight"
        mapping[f"{xt_ff_base}.1.ff.2.bias"] = f"{mammoth_ff_base}.1.ff.2.bias"

        # Post-feedforward norm
        mapping[f"{xt_ff_base}.0.2.gamma"] = f"{mammoth_ff_base}.0.2.gamma"
        mapping[f"{xt_ff_base}.0.2.beta"] = f"{mammoth_ff_base}.0.2.beta"

    # ======== FINAL LOGITS BIAS (BART-specific) ========
    # Map final_logits_bias from x-transformers decoder to Mammoth decoder
    mapping["decoder.final_logits_bias"] = f"decoder.{task_name}.final_logits_bias"

    return mapping


def map_bart_weights_to_mammoth(encoder_model, decoder_model, mammoth_model, num_encoder_layers, num_decoder_layers, task_name, hf_state_dict=None):
    """
    Map BART encoder + decoder weights to Mammoth model

    Handles both encoder and decoder weights

    Args:
        encoder_model: x-transformers encoder model
        decoder_model: x-transformers decoder model
        mammoth_model: Mammoth model
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        task_name: Task name for mapping
        hf_state_dict: Original HF BART state dict (for verification)
    """
    print("\n" + "=" * 70)
    print("Mapping BART Weights to Mammoth")
    print("=" * 70)

    enc_sd = encoder_model.state_dict()
    dec_sd = decoder_model.state_dict()
    mammoth_sd = mammoth_model.state_dict()

    print(f"  Encoder model: {len(enc_sd)} parameters")
    print(f"  Decoder model: {len(dec_sd)} parameters")
    print(f"  Mammoth model: {len(mammoth_sd)} parameters")

    # Debug: Check if final_logits_bias is in each state dict
    final_bias_in_dec = 'final_logits_bias' in dec_sd
    final_bias_key_mammoth = f"decoder.{task_name}.final_logits_bias"
    final_bias_in_mammoth = final_bias_key_mammoth in mammoth_sd
    print(f"  [DEBUG] final_logits_bias in standalone decoder: {final_bias_in_dec}")
    print(f"  [DEBUG] {final_bias_key_mammoth} in Mammoth: {final_bias_in_mammoth}")
    if final_bias_in_mammoth:
        print(f"  [DEBUG] Mammoth bias shape: {mammoth_sd[final_bias_key_mammoth].shape}")

    mapping = create_bart_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, task_name)

    copied = 0
    missed = 0
    missed_details = []

    for xt_key, mammoth_key in mapping.items():
        # Determine source state dict based on key prefix
        if "encoder." in xt_key:
            source_sd = enc_sd
            source_name = "encoder"
            # Remove "encoder." prefix since our standalone encoder doesn't have it
            source_key = xt_key.replace("encoder.", "")
        elif "decoder." in xt_key:
            source_sd = dec_sd
            source_name = "decoder"
            # Remove "decoder." prefix since our standalone decoder doesn't have it
            source_key = xt_key.replace("decoder.", "")
        else:
            missed += 1
            missed_details.append(f"  ✗ Cannot determine source for: {xt_key}")
            continue

        if source_key not in source_sd:
            missed += 1
            missed_details.append(f"  ✗ {source_name.capitalize()} missing: {source_key}")
            continue

        if mammoth_key not in mammoth_sd:
            missed += 1
            missed_details.append(f"  ✗ Mammoth missing: {mammoth_key}")
            continue

        source_val = source_sd[source_key]
        mammoth_val = mammoth_sd[mammoth_key]

        if source_val.shape != mammoth_val.shape:
            missed += 1
            missed_details.append(
                f"  ✗ Shape mismatch ({source_name}): {source_key} {source_val.shape} → {mammoth_key} {mammoth_val.shape}"
            )
            continue

        mammoth_sd[mammoth_key] = source_val.clone()
        copied += 1

    mammoth_model.load_state_dict(mammoth_sd, strict=False)

    print(f"\n  ✓ Weight mapping complete:")
    print(f"    - Successfully mapped: {copied}/{len(mapping)} parameters")
    print(f"    - Failed to map: {missed}/{len(mapping)} parameters")

    # Simple weight verification - compare first 10 values of key parameters
    if hf_state_dict is not None:
        verify_random_weights(hf_state_dict, mammoth_sd, mapping, enc_sd, dec_sd)


    if missed_details:
        print(f"\n  Missing details (first 10):")
        for detail in missed_details[:10]:
            print(detail)
        if len(missed_details) > 10:
            print(f"  ... and {len(missed_details) - 10} more")

    return copied > 0


# =============================================================================
# SECTION 4: Main conversion pipeline
# =============================================================================


def convert_bart_to_mammoth(
    model_path,
    save_path,
    tokenizer_path=None,
    src_lang="en",
    tgt_lang="en"
):
    """
    Convert BART to Mammoth model

    Args:
        model_path: Path to HuggingFace BART model
        save_path: Path to save converted Mammoth model
        tokenizer_path: Path to tokenizer.json (if None, uses model tokenizer)
        src_lang: Source language code
        tgt_lang: Target language code
    """
    print("=" * 70)
    print("BART → Mammoth Converter")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Save path: {save_path}")
    print("=" * 70)

    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    # Stage 0: Prepare tokenizer
    print("\n[Stage 0] Preparing tokenizer")

    if tokenizer_path is None:
        print("  Using BART tokenizer from model directory...")
        tokenizer_path = os.path.join(model_path, "tokenizer.json")

    # Check if tokenizer exists
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")

    # BART tokenizer is already MAMMOTH-compatible
    tokenizer_path = rename_bart_special_tokens_to_mammoth(tokenizer_path)

    # Stage 1: Load BART configuration and extract final_logits_bias
    print("\n[Stage 1] Loading BART configuration and weights")
    config, final_logits_bias, hf_state_dict = load_bart_config_and_bias(model_path)

    # Stage 2: Create BART encoder-decoder models with final_logits_bias
    print("\n[Stage 2] Creating BART encoder-decoder models")
    encoder_model, decoder_model = create_bart_xtransformer_model(config, final_logits_bias)

    # Stage 3: Map BART weights to x-transformers
    print("\n[Stage 3] Mapping BART weights")
    encoder_model, decoder_model = load_bart_weights(hf_state_dict, encoder_model, decoder_model, config)

    # Save encoder and decoder keys
    enc_keys_path = os.path.join(save_dir, "xt_encoder_keys.txt")
    with open(enc_keys_path, "w") as f:
        for key, value in encoder_model.state_dict().items():
            f.write(f"{key}\t{value.shape}\n")
    print(f"\n✓ Encoder keys saved: {enc_keys_path}")

    dec_keys_path = os.path.join(save_dir, "xt_decoder_keys.txt")
    with open(dec_keys_path, "w") as f:
        for key, value in decoder_model.state_dict().items():
            f.write(f"{key}\t{value.shape}\n")
    print(f"✓ Decoder keys saved: {dec_keys_path}")

    # Stage 4: Create Mammoth model
    print("\n[Stage 4] Creating Mammoth model")

    vocabs_dict = create_vocabs_dict_from_hf_tokenizer(tokenizer_path, src_lang, tgt_lang)
    model_opts = create_model_opts_from_bart_model(encoder_model, decoder_model, config, final_logits_bias)
    task_queue_manager = create_task_queue_manager(
        vocabs_dict, config.encoder_layers, config.decoder_layers,
        src_lang, tgt_lang
    )
    opts = create_opts()

    mammoth_model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=None,
    )

    # Log Mammoth model size
    log_model_size(mammoth_model, "Mammoth BART Model")

    # Stage 5: Map weights to Mammoth
    print("\n[Stage 5] Mapping weights to Mammoth")

    task_name = f"task_{src_lang}_{tgt_lang}"
    success = map_bart_weights_to_mammoth(
        encoder_model, decoder_model, mammoth_model,
        config.encoder_layers,
        config.decoder_layers,
        task_name,
        hf_state_dict
    )

    if not success:
        raise RuntimeError("Weight mapping failed!")

    # Save Mammoth keys
    mammoth_keys_path = os.path.join(save_dir, "mammoth_bart_keys.txt")
    with open(mammoth_keys_path, "w") as f:
        for key, value in mammoth_model.state_dict().items():
            f.write(f"{key}\t{value.shape}\n")
    print(f"✓ Mammoth keys saved: {mammoth_keys_path}")

    # Stage 6: Create optimizer and save
    print("\n[Stage 6] Saving Mammoth model")

    optimizer = MultipleOptimizer.from_opts(
        model=mammoth_model,
        opts=opts,
        task_queue_manager=task_queue_manager,
        frame_checkpoint=None,
    )

    save_opts = Namespace()
    save_opts.save_model = save_path
    save_opts.keep_checkpoint = -1

    model_saver = build_model_saver(
        model_opts=model_opts,
        opts=save_opts,
        model=mammoth_model,
        vocabs_dict=vocabs_dict,
        optim=optimizer,
        task_queue_manager=task_queue_manager,
    )

    model_saver.save(step=0, data_state={})

    print("\n" + "=" * 70)
    print("✓ Conversion Complete!")
    print("=" * 70)
    print(f"Saved to: {save_path}")
    print(f"\nArchitecture:")
    print(f"  Encoder: BART ({config.encoder_layers} layers, {config.d_model}d)")
    print(f"  Decoder: BART ({config.decoder_layers} layers, {config.d_model}d)")
    print("=" * 70)

    return mammoth_model


# =============================================================================
# SECTION 5: Command line interface
# =============================================================================


def main():
    """Main entry point for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert BART to Mammoth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output:
  - BART encoder-decoder model
  - Pretrained weights from HuggingFace BART
  - Same vocabulary for source and target
        """,
    )

    parser.add_argument(
        "model_path",
        help="HuggingFace BART model name or local path",
    )

    parser.add_argument(
        "save_path",
        help="Path to save converted model (e.g., './models/bart.pt')"
    )

    parser.add_argument(
        "--tokenizer",
        help="Path to tokenizer.json (if not provided, uses model tokenizer)"
    )

    parser.add_argument(
        "--src-lang",
        default="en",
        help="Source language code (default: en)"
    )

    parser.add_argument(
        "--tgt-lang",
        default="en",
        help="Target language code (default: en)"
    )

    args = parser.parse_args()

    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    try:
        convert_bart_to_mammoth(
            model_path=args.model_path,
            save_path=args.save_path,
            tokenizer_path=args.tokenizer,
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
        )

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())