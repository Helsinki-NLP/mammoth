#!/usr/bin/env python3
import sys
import os

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

"""
HuggingFace ModernBERT (Encoder) + Gemma3 (Decoder) to Mammoth Hybrid Model Converter

Usage: python modernBERT_gemma3_2mammoth.py <encoder_model_path> <decoder_model_path> <save_path>

Hybrid Architecture:
- Encoder: ModernBERT (HuggingFace pretrained, encoder-only BERT model)
  * Features: Pre-LN, GeGLU MLP, bias-free, fused QKV, RoPE, 768 hidden dim
  * 22 layers for ModernBERT-base

- Decoder: Gemma3 (HuggingFace pretrained, decoder-only causal LM)
  * Features: RMSNorm with unit_offset, sandwich norms (4 norms/layer), Q/K norm, MQA, gated MLP
  * Variable hidden dim (e.g., 256 for 270M, larger for 4B)
  * Alternating attention patterns, dual RoPE theta

- Cross-Attention Bridge: Automatic dimension matching if encoder_dim != decoder_dim
  * If dimensions match: Direct cross-attention
  * If dimensions differ: Linear projection bridge handles dimension conversion

This converter:
1. Loads ModernBERT encoder weights from HuggingFace
2. Loads Gemma3 decoder weights from HuggingFace
3. Creates x-transformers hybrid model
4. Maps both pretrained weights to Mammoth multi-task model
5. Handles dimension mismatches automatically via cross-attention context projection
"""

import torch
from collections import OrderedDict
from argparse import Namespace
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

from mammoth.x_transformers import TransformerWrapper, Encoder, Decoder
from mammoth.inputters.vocab import HFTokenizerVocab
from mammoth.distributed.tasks import TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.model_builder import build_model
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.model_saver import build_model_saver


# =============================================================================
# SECTION 1: HuggingFace to x-transformers conversion
# =============================================================================


def create_hybrid_xtransformer_model(encoder_model_path, decoder_model_path):
    """
    Create hybrid encoder + decoder from ModernBERT (encoder) + Gemma3 (decoder)

    Architecture:
    - Encoder: ModernBERT (22 layers, 768 hidden dim, 12 heads)
    - Decoder: Gemma3 270M (18 layers, 640 hidden dim, 4 heads)
    - Cross-attention: Mammoth will handle dimension bridging

    Args:
        encoder_model_path: Path to HuggingFace ModernBERT model
        decoder_model_path: Path to HuggingFace Gemma3 model

    Returns:
        (encoder_model, decoder_model, enc_config, dec_config)
    """
    print("=" * 70)
    print("Creating Hybrid Encoder + Decoder Models")
    print("=" * 70)

    # Load encoder config (ModernBERT)
    print("\n[1/2] Loading ModernBERT encoder configuration...")
    enc_config = AutoConfig.from_pretrained(encoder_model_path, local_files_only=True, trust_remote_code=False)

    # Extract ModernBERT sliding window and dual RoPE parameters
    enc_sliding_window = getattr(enc_config, "local_attention", -1)
    enc_global_attn_every_n_layers = getattr(enc_config, "global_attn_every_n_layers", 3)
    enc_global_rope_theta = getattr(enc_config, "global_rope_theta", 160000.0)
    enc_local_rope_theta = getattr(enc_config, "local_rope_theta", 10000.0)

    print(f"  ModernBERT Encoder:")
    print(f"    - Hidden size: {enc_config.hidden_size}")
    print(f"    - Layers: {enc_config.num_hidden_layers}")
    print(f"    - Attention heads: {enc_config.num_attention_heads}")
    print(f"    - Intermediate size: {getattr(enc_config, 'intermediate_size', enc_config.hidden_size * 4)}")
    print(f"    - Vocab size: {enc_config.vocab_size}")

    # Log sliding window configuration for encoder
    sliding_window_status = "enabled" if enc_sliding_window > 0 else "disabled"
    print(f"    - Sliding window: {sliding_window_status}")
    if enc_sliding_window > 0:
        print(f"      • Window size: {enc_sliding_window} tokens")
        print(f"      • Global attention every N layers: {enc_global_attn_every_n_layers}")
        print(f"      • Global RoPE theta: {enc_global_rope_theta}")
        print(f"      • Local RoPE theta: {enc_local_rope_theta}")

    # Load decoder config (Gemma3)
    print("\n[2/2] Loading Gemma3 decoder configuration...")
    dec_config = AutoConfig.from_pretrained(decoder_model_path, local_files_only=False, trust_remote_code=False)

    # Extract text config if multimodal
    if hasattr(dec_config, 'text_config'):
        dec_config = dec_config.text_config

    # Extract Gemma3 sliding window and dual RoPE parameters
    dec_sliding_window = getattr(dec_config, 'sliding_window', -1)
    dec_global_attn_every_n_layers = getattr(dec_config, 'global_attn_every_n_layers', 3)
    dec_global_rope_theta = getattr(dec_config, 'rope_theta', 1000000.0)  # Global theta (default 1M)
    dec_local_rope_theta = getattr(dec_config, 'rope_local_base_freq', 10000.0)  # Local theta (default 10K)

    print(f"  Gemma3 Decoder:")
    print(f"    - Hidden size: {dec_config.hidden_size}")
    print(f"    - Layers: {dec_config.num_hidden_layers}")
    print(f"    - Attention heads: {dec_config.num_attention_heads}")
    print(f"    - KV heads: {dec_config.num_key_value_heads} (MQA)")
    print(f"    - Head dim: {dec_config.head_dim}")
    print(f"    - Intermediate size: {dec_config.intermediate_size}")
    print(f"    - Vocab size: {dec_config.vocab_size}")
    print(f"    - RoPE global theta: {dec_global_rope_theta}")
    print(f"    - RoPE local theta: {dec_local_rope_theta}")

    # Log sliding window configuration for decoder
    dec_sliding_window_status = "enabled" if dec_sliding_window > 0 else "disabled"
    print(f"    - Sliding window: {dec_sliding_window_status}")
    if dec_sliding_window > 0:
        print(f"      • Window size: {dec_sliding_window} tokens")
        print(f"      • Global attention every N layers: {dec_global_attn_every_n_layers}")

    # Calculate ff_mult for both encoder and decoder
    enc_intermediate_size = getattr(enc_config, "intermediate_size", enc_config.hidden_size * 4)
    enc_ff_mult = enc_intermediate_size / enc_config.hidden_size

    dec_intermediate_size = dec_config.intermediate_size
    dec_ff_mult = dec_intermediate_size / dec_config.hidden_size

    print(f"\n[Architecture Summary]")
    print(f"  Encoder (ModernBERT): {enc_config.hidden_size}d → {enc_config.num_hidden_layers} layers")
    print(f"  Decoder (Gemma3):     {dec_config.hidden_size}d → {dec_config.num_hidden_layers} layers")

    if enc_config.hidden_size != dec_config.hidden_size:
        print(f"  ⚠️  Dimension mismatch detected!")
        print(f"      Mammoth's model_opts will handle cross-attention dimension bridging")
        print(f"      Encoder {enc_config.hidden_size}d → Decoder expects {dec_config.hidden_size}d context")
    else:
        print(f"  ✓ Dimensions match - direct cross-attention")

    # Build encoder (ModernBERT architecture)
    print(f"\n[Building Encoder]")
    encoder = Encoder(
        dim=enc_config.hidden_size,  # 768 for ModernBERT-base
        depth=enc_config.num_hidden_layers,  # 22 for ModernBERT-base
        heads=enc_config.num_attention_heads,
        ff_mult=enc_ff_mult,
        rotary_pos_emb=True,  # ModernBERT uses RoPE
        rotary_pos_emb_base=enc_global_rope_theta,  # Use global theta as base
        ff_glu=True,  # ModernBERT uses GeGLU
        ff_no_bias=True,  # ModernBERT is bias-free
        attn_dropout=getattr(enc_config, "attention_dropout", 0.0),
        # Sliding window attention with dual RoPE
        sliding_window=enc_sliding_window,
        global_attn_every_n_layers=enc_global_attn_every_n_layers,
        global_rope_theta=enc_global_rope_theta,
        local_rope_theta=enc_local_rope_theta,
    )

    encoder_model = TransformerWrapper(
        num_tokens=enc_config.vocab_size,
        max_seq_len=enc_config.max_position_embeddings,
        attn_layers=encoder,
        emb_dropout=0.0,
        post_emb_norm=True,  # ModernBERT has post-embedding norm
        return_only_embed=True,  # Encoder returns embeddings only
    )

    print(f"  ✓ Encoder created: ModernBERT architecture ({enc_config.num_hidden_layers} layers)")

    # Build decoder (Gemma3 architecture)
    print(f"\n[Building Decoder]")
    decoder = Decoder(
        dim=dec_config.hidden_size,  # Variable based on Gemma3 model size
        depth=dec_config.num_hidden_layers,
        heads=dec_config.num_attention_heads,
        attn_dim_head=dec_config.head_dim,  # Explicit head dim (critical for Gemma3)
        ff_mult=dec_ff_mult,
        rotary_pos_emb=True,
        rotary_pos_emb_base=dec_global_rope_theta,  # Gemma3's global RoPE theta
        ff_glu=True,  # Gemma3 uses gated MLP
        ff_no_bias=True,  # Gemma3 is bias-free
        attn_qk_norm=True,  # Gemma3 has Q/K normalization
        attn_qk_norm_dim_scale=True,  # Learnable QK norm scales
        attn_one_kv_head=True,  # MQA (one KV head)
        attn_dropout=dec_config.attention_dropout,
        sandwich_norm=True,  # Gemma3 has 4 norms per layer
        use_rmsnorm=True,  # Gemma3 uses RMSNorm
        cross_attend=True,  # Enable cross-attention for encoder-decoder
        # Sliding window attention with dual RoPE (Gemma3 alternating attention)
        sliding_window=dec_sliding_window,
        global_attn_every_n_layers=dec_global_attn_every_n_layers,
        global_rope_theta=dec_global_rope_theta,
        local_rope_theta=dec_local_rope_theta,
    )

    decoder_model = TransformerWrapper(
        num_tokens=dec_config.vocab_size,
        max_seq_len=dec_config.max_position_embeddings,
        attn_layers=decoder,
        emb_dropout=0.0,
        post_emb_norm=False,  # Gemma3 has no post-emb norm
    )

    print(f"  ✓ Decoder created: Gemma3 architecture ({dec_config.num_hidden_layers} layers)")
    print(f"  ✓ Cross-attention enabled for encoder-decoder architecture")

    return encoder_model, decoder_model, enc_config, dec_config


def load_encoder_weights(encoder_model_path, encoder_model, enc_config):
    """
    Load ModernBERT encoder weights into x-transformers encoder model

    Adapted from BERT2mammoth.py with modifications for hybrid model
    """
    print("\n" + "=" * 70)
    print("Loading ModernBERT Encoder Weights")
    print("=" * 70)

    print(f"  [1/3] Loading ModernBERT model...")
    hf_model = AutoModel.from_pretrained(
        encoder_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    )

    print(f"  [2/3] Extracting state dict...")
    hf_state_dict = hf_model.state_dict()
    print(f"  ✓ Loaded {len(hf_state_dict)} encoder parameters")

    print(f"  [3/3] Mapping weights to x-transformers encoder...")

    # Create weight mapping (from BERT2mammoth.py)
    mapping = create_modernbert_encoder_mapping(enc_config.num_hidden_layers)
    x_state_dict = OrderedDict()

    hidden_size = enc_config.hidden_size

    # Handle missing keys (special case: first layer pre-norm in ModernBERT)
    identity_inits = []

    for hf_key, x_key in mapping.items():
        if hf_key not in hf_state_dict:
            if hf_key == "layers.0.attn_norm.weight":
                # ModernBERT layer 0 has no pre-norm, initialize as identity
                x_state_dict[x_key] = torch.ones(hidden_size, dtype=torch.float32)
                identity_inits.append(hf_key)
            continue

        # Check if x_key is a tuple (for split QKV mapping)
        if isinstance(x_key, tuple):
            # Split fused Wqkv into separate Q, K, V
            fused_qkv = hf_state_dict[hf_key]
            q_weight, k_weight, v_weight = torch.chunk(fused_qkv, 3, dim=0)

            x_state_dict[x_key[0]] = q_weight  # to_q.weight
            x_state_dict[x_key[1]] = k_weight  # to_k.weight
            x_state_dict[x_key[2]] = v_weight  # to_v.weight
        else:
            x_state_dict[x_key] = hf_state_dict[hf_key]

    # Load encoder weights only (strict=False allows missing decoder weights)
    load_result = encoder_model.load_state_dict(x_state_dict, strict=False)

    print(f"  ✓ Encoder weights loaded: {len(x_state_dict)} parameters")
    if identity_inits:
        print(f"  ℹ️  Identity-initialized {len(identity_inits)} missing norms (layer 0)")

    # Log model size
    log_model_size(encoder_model, "Encoder")

    return encoder_model


def load_decoder_weights(decoder_model_path, decoder_model, dec_config):
    """
    Load Gemma3 decoder weights into x-transformers decoder model

    Adapted from gemma2mammoth.py with modifications for hybrid model
    """
    print("\n" + "=" * 70)
    print("Loading Gemma3 Decoder Weights")
    print("=" * 70)

    print(f"  [1/3] Loading Gemma3 model...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        decoder_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False
    )

    print(f"  [2/3] Extracting state dict...")
    hf_state_dict = hf_model.state_dict()
    print(f"  ✓ Loaded {len(hf_state_dict)} decoder parameters")

    print(f"  [3/3] Mapping weights to x-transformers decoder...")

    # Create weight mapping (from gemma2mammoth.py)
    mapping = create_gemma3_decoder_mapping(dec_config.num_hidden_layers)
    x_state_dict = OrderedDict()

    # Track HF keys processed
    hf_keys_processed = set()

    # First pass: collect gate and up projections
    gate_weights = {}
    up_weights = {}

    for hf_key, x_info in mapping.items():
        if hf_key not in hf_state_dict:
            continue

        if isinstance(x_info, tuple):
            x_key, part_type = x_info
            hf_weight = hf_state_dict[hf_key]

            if part_type == "gate":
                gate_weights[x_key] = hf_weight
                hf_keys_processed.add(hf_key)
            elif part_type == "up":
                up_weights[x_key] = hf_weight
                hf_keys_processed.add(hf_key)
        else:
            # Direct mapping - check if QK norm weight needs reshaping
            if "q_norm.weight" in hf_key or "k_norm.weight" in hf_key:
                hf_weight = hf_state_dict[hf_key]
                num_heads = dec_config.num_attention_heads
                kv_heads = 1  # MQA
                dim_head = dec_config.head_dim

                # Reshape for x-transformers 3D format
                reshaped_weight = hf_weight.unsqueeze(0).unsqueeze(0).expand(
                    num_heads if "q_norm" in hf_key else kv_heads,
                    1,
                    dim_head
                )
                x_state_dict[x_info] = reshaped_weight
                hf_keys_processed.add(hf_key)
            else:
                x_state_dict[x_info] = hf_state_dict[hf_key]
                hf_keys_processed.add(hf_key)

    # Second pass: concatenate gate + up projections for GLU
    for x_key in gate_weights.keys():
        if x_key in up_weights:
            gate_weight = gate_weights[x_key]
            up_weight = up_weights[x_key]
            # Concatenate [up; gate] for GLU
            fused_weight = torch.cat([up_weight, gate_weight], dim=0)
            x_state_dict[x_key] = fused_weight

    # Load decoder weights (strict=False allows missing encoder weights)
    load_result = decoder_model.load_state_dict(x_state_dict, strict=False)

    print(f"  ✓ Decoder weights loaded: {len(x_state_dict)} parameters")
    print(f"  ✓ HF weight coverage: {len(hf_keys_processed)}/{len(hf_state_dict)} parameters")

    unmapped = set(hf_state_dict.keys()) - hf_keys_processed
    if unmapped:
        print(f"  ⚠️  {len(unmapped)} HF parameters not mapped (expected for decoder-only extras)")

    # Log model size
    log_model_size(decoder_model, "Decoder")

    return decoder_model


def create_modernbert_encoder_mapping(num_layers):
    """
    Create mapping from HuggingFace ModernBERT to standalone TransformerWrapper encoder
    (adapted from BERT2mammoth.py, removing "encoder." prefix for standalone model)
    """
    mapping = {}

    # Token embeddings
    mapping["embeddings.tok_embeddings.weight"] = "token_emb.emb.weight"

    # Embedding normalization
    mapping["embeddings.norm.weight"] = "post_emb_norm.gamma"

    # Encoder layers
    for i in range(num_layers):
        attn_idx = i * 2
        ff_idx = attn_idx + 1

        # Pre-LN before attention
        mapping[f"layers.{i}.attn_norm.weight"] = (
            f"attn_layers.layers.{attn_idx}.0.0.gamma"
        )

        # Fused Wqkv - split into Q, K, V
        mapping[f"layers.{i}.attn.Wqkv.weight"] = (
            f"attn_layers.layers.{attn_idx}.1.to_q.weight",
            f"attn_layers.layers.{attn_idx}.1.to_k.weight",
            f"attn_layers.layers.{attn_idx}.1.to_v.weight"
        )

        # Attention output
        mapping[f"layers.{i}.attn.Wo.weight"] = (
            f"attn_layers.layers.{attn_idx}.1.to_out.weight"
        )

        # Pre-LN before MLP
        mapping[f"layers.{i}.mlp_norm.weight"] = (
            f"attn_layers.layers.{ff_idx}.0.0.gamma"
        )

        # MLP with GLU
        mapping[f"layers.{i}.mlp.Wi.weight"] = (
            f"attn_layers.layers.{ff_idx}.1.ff.0.proj.weight"
        )
        mapping[f"layers.{i}.mlp.Wo.weight"] = (
            f"attn_layers.layers.{ff_idx}.1.ff.2.weight"
        )

    # Final layer normalization
    mapping["final_norm.weight"] = "attn_layers.final_norm.gamma"

    return mapping


def create_gemma3_decoder_mapping(num_layers):
    """
    Create mapping from HuggingFace Gemma3 to standalone TransformerWrapper decoder
    (adapted from gemma2mammoth.py, removing "decoder." prefix for standalone model)
    """
    mapping = {}

    # Embeddings
    mapping["model.embed_tokens.weight"] = "token_emb.emb.weight"

    # LM head
    mapping["lm_head.weight"] = "to_logits.weight"

    # Final norm
    mapping["model.norm.weight"] = "attn_layers.final_norm.g"

    # Decoder layers (with cross-attention - indices: self_attn, cross_attn, ff)
    for layer_idx in range(num_layers):
        # In decoder-only + cross_attend=True mode: [self_attn, cross_attn, ff] triplets
        # self_attn_idx = layer_idx * 3, cross_attn_idx = layer_idx * 3 + 1, ff_idx = layer_idx * 3 + 2
        self_attn_idx = layer_idx * 3
        cross_attn_idx = layer_idx * 3 + 1  # Cross-attention (will be skipped by Gemma3)
        ff_idx = layer_idx * 3 + 2

        hf_prefix = f"model.layers.{layer_idx}"
        xt_attn_prefix = f"attn_layers.layers.{self_attn_idx}"
        xt_ff_prefix = f"attn_layers.layers.{ff_idx}"

        # === Self-Attention Block ===
        mapping[f"{hf_prefix}.input_layernorm.weight"] = f"{xt_attn_prefix}.0.0.g"
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{xt_attn_prefix}.1.to_q.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{xt_attn_prefix}.1.to_k.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{xt_attn_prefix}.1.to_v.weight"
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{xt_attn_prefix}.1.to_out.weight"

        # Q/K normalization
        mapping[f"{hf_prefix}.self_attn.q_norm.weight"] = f"{xt_attn_prefix}.1.qk_norm_q_scale"
        mapping[f"{hf_prefix}.self_attn.k_norm.weight"] = f"{xt_attn_prefix}.1.qk_norm_k_scale"

        # Post-attention norm (sandwich_norm)
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{xt_attn_prefix}.0.1.g"

        # === Feedforward Block ===
        mapping[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = f"{xt_ff_prefix}.0.0.g"

        # MLP - Gated structure (will be concatenated)
        mapping[f"{hf_prefix}.mlp.gate_proj.weight"] = (
            f"{xt_ff_prefix}.1.ff.0.proj.weight",
            "gate"
        )
        mapping[f"{hf_prefix}.mlp.up_proj.weight"] = (
            f"{xt_ff_prefix}.1.ff.0.proj.weight",
            "up"
        )
        mapping[f"{hf_prefix}.mlp.down_proj.weight"] = f"{xt_ff_prefix}.1.ff.2.weight"

        # Post-feedforward norm (sandwich_norm)
        mapping[f"{hf_prefix}.post_feedforward_layernorm.weight"] = f"{xt_ff_prefix}.0.1.g"

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




def rename_encoder_special_tokens_to_mammoth(tokenizer_path, output_path=None):
    """
    Rename HuggingFace special tokens to MAMMOTH conventions in-place

    This avoids adding new tokens and wasting embeddings. Instead, we directly
    rename the existing HF tokens to match MAMMOTH's naming:
    - [CLS] → <s> (BOS)
    - [SEP] → </s> (EOS)
    - [PAD] → <pad>
    - [MASK] → <mask>
    - [UNK] → <unk>

    Args:
        tokenizer_path: Path to the HuggingFace tokenizer.json file
        output_path: Optional path to save modified tokenizer (if None, modifies in-place)

    Returns:
        Path to the modified tokenizer file
    """
    import json
    import shutil
    from pathlib import Path

    print(f"Renaming HF special tokens to MAMMOTH conventions...")
    print(f"  Input: {tokenizer_path}")

    # Load tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Define renaming mapping
    token_renames = {
        '[CLS]': '<s>',      # BOS
        '[SEP]': '</s>',     # EOS
        '[PAD]': '<pad>',    # PAD (lowercase)
        '[MASK]': '<mask>',  # MASK (lowercase)
        '[UNK]': '<unk>',    # UNK (lowercase)
    }

    # Track renamed tokens
    renamed_count = 0
    rename_log = []

    # 1. Rename in vocabulary
    vocab = tokenizer_data['model']['vocab']
    for old_token, new_token in token_renames.items():
        if old_token in vocab:
            token_id = vocab[old_token]
            del vocab[old_token]
            vocab[new_token] = token_id
            renamed_count += 1
            rename_log.append(f"  ✓ {old_token:12s} → {new_token:12s} (ID {token_id})")

    # 2. Rename in added_tokens section
    if 'added_tokens' in tokenizer_data:
        for token_entry in tokenizer_data['added_tokens']:
            if token_entry['content'] in token_renames:
                old_content = token_entry['content']
                token_entry['content'] = token_renames[old_content]
                rename_log.append(f"  ✓ Updated added_tokens: {old_content} → {token_entry['content']}")

    # 3. Rename in post_processor special_tokens
    if 'post_processor' in tokenizer_data and 'special_tokens' in tokenizer_data['post_processor']:
        special_tokens = tokenizer_data['post_processor']['special_tokens']
        new_special_tokens = {}

        for token_name, token_info in special_tokens.items():
            if token_name in token_renames:
                new_name = token_renames[token_name]
                # Update the token info
                token_info['id'] = new_name
                token_info['tokens'] = [new_name]
                new_special_tokens[new_name] = token_info
                rename_log.append(f"  ✓ Updated post_processor special_tokens: {token_name} → {new_name}")
            else:
                new_special_tokens[token_name] = token_info

        tokenizer_data['post_processor']['special_tokens'] = new_special_tokens

    # 4. Update post_processor template (single and pair)
    if 'post_processor' in tokenizer_data:
        post_proc = tokenizer_data['post_processor']

        # Update 'single' template
        if 'single' in post_proc:
            for item in post_proc['single']:
                if 'SpecialToken' in item:
                    old_id = item['SpecialToken']['id']
                    if old_id in token_renames:
                        item['SpecialToken']['id'] = token_renames[old_id]
                        rename_log.append(f"  ✓ Updated single template: {old_id} → {token_renames[old_id]}")

        # Update 'pair' template
        if 'pair' in post_proc:
            for item in post_proc['pair']:
                if 'SpecialToken' in item:
                    old_id = item['SpecialToken']['id']
                    if old_id in token_renames:
                        item['SpecialToken']['id'] = token_renames[old_id]
                        rename_log.append(f"  ✓ Updated pair template: {old_id} → {token_renames[old_id]}")

    # Save modified tokenizer
    if output_path is None:
        output_path = tokenizer_path
        # Create backup
        backup_path = str(Path(tokenizer_path).with_suffix('.json.backup'))
        if not Path(backup_path).exists():
            shutil.copy2(tokenizer_path, backup_path)
            print(f"  Created backup: {backup_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Renamed {renamed_count} special tokens:")
    for log_entry in rename_log[:5]:  # Show first 5 renames
        print(log_entry)

    print(f"✓ Modified tokenizer saved to: {output_path}")
    return output_path


def rename_decoder_special_tokens_to_mammoth(tokenizer_path, output_path=None, vocab_size_limit=None):
    """
    Rename HuggingFace special tokens to MAMMOTH conventions in-place.
    This avoids adding new tokens and wasting embeddings.

    Comprehensive renaming that handles:
    - Vocabulary mapping
    - added_tokens section
    - post_processor special_tokens
    - post_processor templates (single and pair)

    Args:
        tokenizer_path: Path to the HuggingFace tokenizer.json file
        output_path: Optional path to save modified tokenizer (if None, modifies in-place)
        vocab_size_limit: Optional max vocab size. Tokens with IDs >= this will be removed.
                         This is useful for removing multimodal tokens (e.g., <image_soft_token>)
                         that exceed the text model's embedding size.

    Returns:
        Path to the modified tokenizer file
    """
    import json
    import shutil
    from pathlib import Path

    print(f"Renaming HF special tokens to MAMMOTH conventions...")
    print(f"  Input: {tokenizer_path}")
    if vocab_size_limit:
        print(f"  Vocab size limit: {vocab_size_limit}")

    # Load tokenizer
    with open(tokenizer_path, 'r', encoding='utf-8') as f:
        tokenizer_data = json.load(f)

    # Define renaming mapping
    token_renames = {
        '<bos>': '<s>',      # BOS
        '<eos>': '</s>',     # EOS
        '<pad>': '<pad>',    # PAD (lowercase)
        '<mask>': '<mask>',  # MASK (lowercase)
        '<unk>': '<unk>',    # UNK (lowercase)
    }

    # Track renamed tokens
    renamed_count = 0
    rename_log = []

    # 1. Rename in vocabulary
    vocab = tokenizer_data['model']['vocab']

    # Pre-step: Remove tokens exceeding vocab_size_limit (e.g., multimodal tokens)
    removed_tokens = []
    if vocab_size_limit:
        tokens_to_remove = [token for token, token_id in vocab.items() if token_id >= vocab_size_limit]
        for token in tokens_to_remove:
            token_id = vocab[token]
            del vocab[token]
            removed_tokens.append((token, token_id))

        if removed_tokens:
            print(f"  ℹ️  Removed {len(removed_tokens)} token(s) exceeding vocab_size_limit ({vocab_size_limit}):")
            for token, token_id in removed_tokens[:3]:
                print(f"    - {token} (ID {token_id})")
            if len(removed_tokens) > 3:
                print(f"    ... and {len(removed_tokens) - 3} more")

    # Pre-step: Handle conflicts - rename existing target tokens to avoid collisions
    # In Gemma3, both <bos> and <s> exist, as well as <eos> and </s>
    # We need to rename the existing <s> and </s> first to avoid conflicts
    conflict_renames = {
        '<s>': '<s_original>',      # Rename existing <s> to avoid conflict with <bos> → <s>
        '</s>': '</s_original>',    # Rename existing </s> to avoid conflict with <eos> → </s>
    }

    for existing_token, temp_name in conflict_renames.items():
        if existing_token in vocab:
            # Only rename if this token is not a target of our main renaming
            # (i.e., no token in token_renames maps to it as source)
            if existing_token not in token_renames:
                token_id = vocab[existing_token]
                del vocab[existing_token]
                vocab[temp_name] = token_id
                rename_log.append(f"  ℹ️  {existing_token:12s} → {temp_name:12s} (ID {token_id}) [conflict resolution]")

    # Main renaming: Map HF tokens to MAMMOTH conventions
    for old_token, new_token in token_renames.items():
        if old_token in vocab:
            token_id = vocab[old_token]
            del vocab[old_token]
            vocab[new_token] = token_id
            renamed_count += 1
            rename_log.append(f"  ✓ {old_token:12s} → {new_token:12s} (ID {token_id})")

    # 2. Filter and rename in added_tokens section
    if 'added_tokens' in tokenizer_data:
        # First, remove tokens exceeding vocab_size_limit
        if vocab_size_limit:
            original_count = len(tokenizer_data['added_tokens'])
            tokenizer_data['added_tokens'] = [
                token for token in tokenizer_data['added_tokens']
                if token['id'] < vocab_size_limit
            ]
            removed_count = original_count - len(tokenizer_data['added_tokens'])
            if removed_count > 0:
                print(f"  ℹ️  Removed {removed_count} token(s) from added_tokens exceeding limit")

        # Then, handle conflicts in added_tokens
        for token_entry in tokenizer_data['added_tokens']:
            if token_entry['content'] in conflict_renames:
                if token_entry['content'] not in token_renames:
                    old_content = token_entry['content']
                    token_entry['content'] = conflict_renames[old_content]
                    rename_log.append(f"  ℹ️  Updated added_tokens (conflict): {old_content} → {token_entry['content']}")

        # Then, apply main renaming
        for token_entry in tokenizer_data['added_tokens']:
            if token_entry['content'] in token_renames:
                old_content = token_entry['content']
                token_entry['content'] = token_renames[old_content]
                rename_log.append(f"  ✓ Updated added_tokens: {old_content} → {token_entry['content']}")

    # 3. Rename in post_processor special_tokens
    if 'post_processor' in tokenizer_data and 'special_tokens' in tokenizer_data['post_processor']:
        special_tokens = tokenizer_data['post_processor']['special_tokens']
        new_special_tokens = {}

        # First pass: handle conflicts
        for token_name, token_info in special_tokens.items():
            if token_name in conflict_renames and token_name not in token_renames:
                new_name = conflict_renames[token_name]
                # Update the token info
                token_info['id'] = new_name
                token_info['tokens'] = [new_name]
                new_special_tokens[new_name] = token_info
                rename_log.append(f"  ℹ️  Updated post_processor special_tokens (conflict): {token_name} → {new_name}")
            else:
                new_special_tokens[token_name] = token_info

        # Second pass: apply main renaming
        special_tokens = new_special_tokens.copy()
        new_special_tokens = {}

        for token_name, token_info in special_tokens.items():
            if token_name in token_renames:
                new_name = token_renames[token_name]
                # Update the token info
                token_info['id'] = new_name
                token_info['tokens'] = [new_name]
                new_special_tokens[new_name] = token_info
                rename_log.append(f"  ✓ Updated post_processor special_tokens: {token_name} → {new_name}")
            else:
                new_special_tokens[token_name] = token_info

        tokenizer_data['post_processor']['special_tokens'] = new_special_tokens

    # 4. Update post_processor template (single and pair)
    if 'post_processor' in tokenizer_data:
        post_proc = tokenizer_data['post_processor']

        # Update 'single' template
        if 'single' in post_proc:
            # First pass: handle conflicts
            for item in post_proc['single']:
                if 'SpecialToken' in item:
                    old_id = item['SpecialToken']['id']
                    if old_id in conflict_renames and old_id not in token_renames:
                        item['SpecialToken']['id'] = conflict_renames[old_id]
                        rename_log.append(f"  ℹ️  Updated single template (conflict): {old_id} → {conflict_renames[old_id]}")

            # Second pass: apply main renaming
            for item in post_proc['single']:
                if 'SpecialToken' in item:
                    old_id = item['SpecialToken']['id']
                    if old_id in token_renames:
                        item['SpecialToken']['id'] = token_renames[old_id]
                        rename_log.append(f"  ✓ Updated single template: {old_id} → {token_renames[old_id]}")

        # Update 'pair' template
        if 'pair' in post_proc:
            # First pass: handle conflicts
            for item in post_proc['pair']:
                if 'SpecialToken' in item:
                    old_id = item['SpecialToken']['id']
                    if old_id in conflict_renames and old_id not in token_renames:
                        item['SpecialToken']['id'] = conflict_renames[old_id]
                        rename_log.append(f"  ℹ️  Updated pair template (conflict): {old_id} → {conflict_renames[old_id]}")

            # Second pass: apply main renaming
            for item in post_proc['pair']:
                if 'SpecialToken' in item:
                    old_id = item['SpecialToken']['id']
                    if old_id in token_renames:
                        item['SpecialToken']['id'] = token_renames[old_id]
                        rename_log.append(f"  ✓ Updated pair template: {old_id} → {token_renames[old_id]}")

    # Save modified tokenizer
    if output_path is None:
        output_path = tokenizer_path
        # Create backup
        backup_path = str(Path(tokenizer_path).with_suffix('.json.backup'))
        if not Path(backup_path).exists():
            shutil.copy2(tokenizer_path, backup_path)
            print(f"  Created backup: {backup_path}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    print(f"✓ Renamed {renamed_count} special tokens:")
    for log_entry in rename_log[:5]:  # Show first 5 renames
        print(log_entry)

    print(f"✓ Modified tokenizer saved to: {output_path}")
    return output_path


def create_vocabs_dict_from_hf_tokenizer(src_tokenizer_path, tgt_tokenizer_path, src_lang="en", tgt_lang="ar"):
    """
    Create vocabs_dict for Mammoth model

    For hybrid model, we typically use:
    - Source (encoder): ModernBERT tokenizer
    - Target (decoder): Gemma3 tokenizer
    """
    from tokenizers import Tokenizer

    src_vocab = HFTokenizerVocab(
        tokenizer_path=src_tokenizer_path,
        tag=f"src_{src_lang}",
    )

    tgt_vocab = HFTokenizerVocab(
        tokenizer_path=tgt_tokenizer_path,
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


def create_model_opts_from_hybrid_model(encoder_model, decoder_model, enc_config, dec_config):
    """
    Create model_opts for Mammoth from hybrid encoder + decoder models

    Critical: Uses enc_/dec_ prefix system to specify different dimensions
    """
    model_opts = Namespace()

    # Basic settings
    model_opts.model_dtype = "bf16"

    # Architecture - different encoder/decoder dimensions
    model_opts.enc_layers = [enc_config.num_hidden_layers]  # 22 for ModernBERT-base
    model_opts.dec_layers = [dec_config.num_hidden_layers]  # Variable for Gemma3

    model_opts.enc_model_dim = enc_config.hidden_size  # 768 for ModernBERT
    model_opts.dec_model_dim = dec_config.hidden_size  # Variable for Gemma3

    # Calculate ff_mult
    enc_intermediate_size = getattr(enc_config, "intermediate_size", enc_config.hidden_size * 4)
    enc_ff_mult = enc_intermediate_size / enc_config.hidden_size

    dec_intermediate_size = dec_config.intermediate_size
    dec_ff_mult = dec_intermediate_size / dec_config.hidden_size

    # Extract sliding window and dual RoPE parameters
    enc_sliding_window = getattr(enc_config, "local_attention", -1)
    enc_global_attn_every_n_layers = getattr(enc_config, "global_attn_every_n_layers", 3)
    enc_global_rope_theta = getattr(enc_config, "global_rope_theta", 160000.0)
    enc_local_rope_theta = getattr(enc_config, "local_rope_theta", 10000.0)

    dec_sliding_window = getattr(dec_config, 'sliding_window', -1)
    dec_global_attn_every_n_layers = getattr(dec_config, 'global_attn_every_n_layers', 3)
    dec_global_rope_theta = getattr(dec_config, 'rope_theta', 1000000.0)
    dec_local_rope_theta = getattr(dec_config, 'rope_local_base_freq', 10000.0)

    # Set top-level model_opts attributes for sliding window and dual RoPE
    model_opts.enc_sliding_window = enc_sliding_window
    model_opts.enc_global_attn_every_n_layers = enc_global_attn_every_n_layers
    model_opts.enc_global_rope_theta = enc_global_rope_theta
    model_opts.enc_local_rope_theta = enc_local_rope_theta

    model_opts.dec_sliding_window = dec_sliding_window
    model_opts.dec_global_attn_every_n_layers = dec_global_attn_every_n_layers
    model_opts.dec_global_rope_theta = dec_global_rope_theta
    model_opts.dec_local_rope_theta = dec_local_rope_theta

    # x_transformers_opts with enc_/dec_ prefixes
    model_opts.x_transformers_opts = {
        # Shared options
        "attn_flash": True,

        # Encoder-specific (ModernBERT)
        "enc_heads": enc_config.num_attention_heads,
        "enc_attn_dim_head": enc_config.hidden_size // enc_config.num_attention_heads,
        "enc_attn_dropout": getattr(enc_config, "attention_dropout", 0.0),
        "enc_ff_mult": enc_ff_mult,
        "enc_ff_dropout": getattr(enc_config, "mlp_dropout", 0.0),
        "enc_ff_glu": True,  # ModernBERT uses GeGLU
        "enc_ff_no_bias": True,
        "enc_rotary_pos_emb": True,
        "enc_rotary_pos_emb_base": enc_global_rope_theta,  # Use global theta as base
        "enc_post_emb_norm": True,
        "enc_max_seq_len": enc_config.max_position_embeddings,

        # Decoder-specific (Gemma3)
        "dec_heads": dec_config.num_attention_heads,
        "dec_attn_dim_head": dec_config.head_dim,
        "dec_attn_dropout": dec_config.attention_dropout,
        "dec_attn_qk_norm": True,
        "dec_attn_qk_norm_dim_scale": True,
        "dec_attn_kv_heads": dec_config.num_key_value_heads,
        "dec_ff_mult": dec_ff_mult,
        "dec_ff_dropout": 0.0,
        "dec_ff_glu": True,
        "dec_ff_no_bias": True,
        "dec_rotary_pos_emb": True,
        "dec_rotary_pos_emb_base": dec_global_rope_theta,  # Use global theta as base
        "dec_use_rmsnorm": True,
        "dec_sandwich_norm": True,
        "dec_post_emb_norm": False,
        "dec_max_seq_len": dec_config.max_position_embeddings,
        "dec_scaled_embeddings": True,  # Gemma3 uses scaled embeddings

        # Cross-attention bridge
        "dec_cross_attn_dim_context": model_opts.enc_model_dim,
    }

    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.normformer = False
    model_opts.dropout = [0.0]
    model_opts.attention_dropout = [dec_config.attention_dropout]

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


def create_task_queue_manager(vocabs_dict, num_encoder_layers, num_decoder_layers, src_lang="en", tgt_lang="ar"):
    """
    Create TaskQueueManager for hybrid model

    Architecture:
    - 1 shared encoder (ModernBERT)
    - 1 task-specific decoder (Gemma3)
    """
    opts = Namespace()
    opts.tasks = {}

    task_name = f"task_{src_lang}_{tgt_lang}"
    opts.tasks[task_name] = {
        "src_tgt": f"{src_lang}-{tgt_lang}",
        "weight": 1.0,
        "introduce_at_training_step": 0,
        "node_gpu": "0:0",
        "enc_sharing_group": ["en"],  # Shared ModernBERT encoder
        "dec_sharing_group": [tgt_lang],  # Task-specific Gemma3 decoder
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
        use_attention_bridge=False, new_group_func=lambda ranks: None
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


def create_hybrid_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, task_name):
    """
    Create combined mapping for hybrid model:
    - Encoder: ModernBERT weights
    - Decoder: Gemma3 weights
    """
    mapping = {}

    # ======== ENCODER MAPPING (ModernBERT) ========
    mapping["encoder.token_emb.emb.weight"] = f"encoder.{task_name}.token_emb.emb.weight"
    mapping["encoder.post_emb_norm.gamma"] = f"encoder.{task_name}.post_emb_norm.gamma"

    # Note: RoPE inv_freq is NOT mapped - it's auto-computed by Mammoth from the theta values
    # With dual RoPE (global and local theta), Mammoth will generate inv_freq dynamically

    for layer_idx in range(num_encoder_layers):
        attn_idx = layer_idx * 2
        ff_idx = attn_idx + 1

        xt_attn_base = f"encoder.attn_layers.layers.{attn_idx}"
        mammoth_attn_base = f"encoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{attn_idx}"

        # Attention
        mapping[f"{xt_attn_base}.0.0.gamma"] = f"{mammoth_attn_base}.0.0.gamma"
        mapping[f"{xt_attn_base}.1.to_q.weight"] = f"{mammoth_attn_base}.1.to_q.weight"
        mapping[f"{xt_attn_base}.1.to_k.weight"] = f"{mammoth_attn_base}.1.to_k.weight"
        mapping[f"{xt_attn_base}.1.to_v.weight"] = f"{mammoth_attn_base}.1.to_v.weight"
        mapping[f"{xt_attn_base}.1.to_out.weight"] = f"{mammoth_attn_base}.1.to_out.weight"

        # Feedforward
        xt_ff_base = f"encoder.attn_layers.layers.{ff_idx}"
        mammoth_ff_base = f"encoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{ff_idx}"

        mapping[f"{xt_ff_base}.0.0.gamma"] = f"{mammoth_ff_base}.0.0.gamma"
        mapping[f"{xt_ff_base}.1.ff.0.proj.weight"] = f"{mammoth_ff_base}.1.ff.0.proj.weight"
        mapping[f"{xt_ff_base}.1.ff.2.weight"] = f"{mammoth_ff_base}.1.ff.2.weight"

    mapping["encoder.attn_layers.final_norm.gamma"] = (
        f"encoder.{task_name}.attn_layers.attention_layers_stack.0.final_norm.gamma"
    )

    # ======== DECODER MAPPING (Gemma3) ========
    mapping["decoder.token_emb.emb.weight"] = f"decoder.{task_name}.token_emb.emb.weight"

    # Note: RoPE inv_freq is NOT mapped - it's auto-computed by Mammoth from the theta values
    # With dual RoPE (global and local theta), Mammoth will generate inv_freq dynamically

    for layer_idx in range(num_decoder_layers):
        # IMPORTANT: Since we created the standalone decoder with cross_attend=True,
        # x-transformers creates [self_attn, cross_attn, ff] triplets, not pairs!
        # So we use triplet indexing for both source (xt) and target (mammoth)
        xt_self_attn_idx = layer_idx * 3      # Self-attention
        xt_cross_attn_idx = layer_idx * 3 + 1  # Cross-attention (skip - no Gemma3 weights)
        xt_ff_idx = layer_idx * 3 + 2         # Feedforward

        # Mammoth encoder-decoder: also [self_attn, cross_attn, ff] triplets
        mammoth_self_attn_idx = layer_idx * 3
        mammoth_cross_attn_idx = layer_idx * 3 + 1  # Will be randomly initialized
        mammoth_ff_idx = layer_idx * 3 + 2

        xt_attn_base = f"decoder.attn_layers.layers.{xt_self_attn_idx}"
        mammoth_attn_base = f"decoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{mammoth_self_attn_idx}"

        # Self-attention (map from standalone decoder to mammoth)
        mapping[f"{xt_attn_base}.0.0.g"] = f"{mammoth_attn_base}.0.0.g"
        mapping[f"{xt_attn_base}.1.to_q.weight"] = f"{mammoth_attn_base}.1.to_q.weight"
        mapping[f"{xt_attn_base}.1.to_k.weight"] = f"{mammoth_attn_base}.1.to_k.weight"
        mapping[f"{xt_attn_base}.1.to_v.weight"] = f"{mammoth_attn_base}.1.to_v.weight"
        mapping[f"{xt_attn_base}.1.to_out.weight"] = f"{mammoth_attn_base}.1.to_out.weight"
        mapping[f"{xt_attn_base}.1.qk_norm_q_scale"] = f"{mammoth_attn_base}.1.qk_norm_q_scale"
        mapping[f"{xt_attn_base}.1.qk_norm_k_scale"] = f"{mammoth_attn_base}.1.qk_norm_k_scale"
        mapping[f"{xt_attn_base}.0.1.g"] = f"{mammoth_attn_base}.0.1.g"  # Post-attn norm

        # Cross-attention: Skip! Gemma3 doesn't have cross-attention weights
        # The standalone decoder has placeholder cross-attn layers (randomly initialized)
        # We don't map these - Mammoth's cross-attn will be randomly initialized too

        # Feedforward
        xt_ff_base = f"decoder.attn_layers.layers.{xt_ff_idx}"
        mammoth_ff_base = f"decoder.{task_name}.attn_layers.attention_layers_stack.0.layers.{mammoth_ff_idx}"

        mapping[f"{xt_ff_base}.0.0.g"] = f"{mammoth_ff_base}.0.0.g"
        mapping[f"{xt_ff_base}.1.ff.0.proj.weight"] = f"{mammoth_ff_base}.1.ff.0.proj.weight"
        mapping[f"{xt_ff_base}.1.ff.2.weight"] = f"{mammoth_ff_base}.1.ff.2.weight"
        mapping[f"{xt_ff_base}.0.1.g"] = f"{mammoth_ff_base}.0.1.g"  # Post-FF norm

    mapping["decoder.attn_layers.final_norm.g"] = (
        f"decoder.{task_name}.attn_layers.attention_layers_stack.0.final_norm.g"
    )

    return mapping


def map_hybrid_weights_to_mammoth(encoder_model, decoder_model, mammoth_model, num_encoder_layers, num_decoder_layers, task_name):
    """
    Map hybrid encoder + decoder weights to Mammoth model

    Handles both encoder (ModernBERT) and decoder (Gemma3) weights
    """
    print("\n" + "=" * 70)
    print("Mapping Hybrid Weights to Mammoth")
    print("=" * 70)

    enc_sd = encoder_model.state_dict()
    dec_sd = decoder_model.state_dict()
    mammoth_sd = mammoth_model.state_dict()

    print(f"  Encoder model: {len(enc_sd)} parameters")
    print(f"  Decoder model: {len(dec_sd)} parameters")
    print(f"  Mammoth model: {len(mammoth_sd)} parameters")

    mapping = create_hybrid_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, task_name)

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


def convert_hybrid_to_mammoth(
    encoder_model_path,
    decoder_model_path,
    save_path,
    src_tokenizer_path=None,
    tgt_tokenizer_path=None,
    src_lang="en",
    tgt_lang="ar"
):
    """
    Convert ModernBERT (encoder) + Gemma3 (decoder) to Mammoth hybrid model

    Args:
        encoder_model_path: Path to HuggingFace ModernBERT model
        decoder_model_path: Path to HuggingFace Gemma3 model
        save_path: Path to save converted Mammoth model
        src_tokenizer_path: Path to source tokenizer (if None, uses encoder tokenizer)
        tgt_tokenizer_path: Path to target tokenizer (if None, uses decoder tokenizer)
        src_lang: Source language code
        tgt_lang: Target language code
    """
    print("=" * 70)
    print("ModernBERT + Gemma3 → Mammoth Hybrid Converter")
    print("=" * 70)
    print(f"Encoder: {encoder_model_path}")
    print(f"Decoder: {decoder_model_path}")
    print(f"Save path: {save_path}")
    print("=" * 70)

    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    # Stage 0: Prepare tokenizers
    print("\n[Stage 0] Preparing tokenizers")

    if src_tokenizer_path is None:
        print("  Creating source tokenizer from ModernBERT...")
        src_tokenizer_path = os.path.join(save_dir, "src_tokenizer.json")
        encoder_tokenizer = os.path.join(encoder_model_path, "tokenizer.json")
        rename_encoder_special_tokens_to_mammoth(encoder_tokenizer, src_tokenizer_path)

    if tgt_tokenizer_path is None:
        print("  Creating target tokenizer from Gemma3...")
        tgt_tokenizer_path = os.path.join(save_dir, "tgt_tokenizer.json")
        decoder_tokenizer = os.path.join(decoder_model_path, "tokenizer.json")

        # Get Gemma3 vocab size for filtering
        dec_config_temp = AutoConfig.from_pretrained(decoder_model_path, local_files_only=False)
        if hasattr(dec_config_temp, 'text_config'):
            dec_config_temp = dec_config_temp.text_config
        vocab_size = dec_config_temp.vocab_size

        rename_decoder_special_tokens_to_mammoth(decoder_tokenizer, tgt_tokenizer_path, vocab_size_limit=vocab_size)

    # Stage 1: Create hybrid encoder + decoder models
    print("\n[Stage 1] Creating hybrid encoder + decoder models")
    encoder_model, decoder_model, enc_config, dec_config = create_hybrid_xtransformer_model(
        encoder_model_path, decoder_model_path
    )

    # Stage 2: Load encoder weights
    print("\n[Stage 2] Loading ModernBERT encoder weights")
    encoder_model = load_encoder_weights(encoder_model_path, encoder_model, enc_config)

    # Stage 3: Load decoder weights
    print("\n[Stage 3] Loading Gemma3 decoder weights")
    decoder_model = load_decoder_weights(decoder_model_path, decoder_model, dec_config)

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

    vocabs_dict = create_vocabs_dict_from_hf_tokenizer(
        src_tokenizer_path, tgt_tokenizer_path, src_lang, tgt_lang
    )

    model_opts = create_model_opts_from_hybrid_model(encoder_model, decoder_model, enc_config, dec_config)
    task_queue_manager = create_task_queue_manager(
        vocabs_dict, enc_config.num_hidden_layers, dec_config.num_hidden_layers,
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
    log_model_size(mammoth_model, "Mammoth Hybrid Model")

    # Stage 5: Map weights
    print("\n[Stage 5] Mapping weights to Mammoth")

    task_name = f"task_{src_lang}_{tgt_lang}"
    success = map_hybrid_weights_to_mammoth(
        encoder_model, decoder_model, mammoth_model,
        enc_config.num_hidden_layers,
        dec_config.num_hidden_layers,
        task_name
    )

    if not success:
        raise RuntimeError("Weight mapping failed!")

    # Save Mammoth keys
    mammoth_keys_path = os.path.join(save_dir, "mammoth_hybrid_keys.txt")
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
    print(f"  Encoder: ModernBERT ({enc_config.num_hidden_layers} layers, {enc_config.hidden_size}d)")
    print(f"  Decoder: Gemma3 ({dec_config.num_hidden_layers} layers, {dec_config.hidden_size}d)")
    if enc_config.hidden_size != dec_config.hidden_size:
        print(f"  Cross-attention: Dimension bridge ({enc_config.hidden_size}d → {dec_config.hidden_size}d)")
    print("=" * 70)

    return mammoth_model


# =============================================================================
# SECTION 5: Command line interface
# =============================================================================


def main():
    """Main entry point for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert ModernBERT (encoder) + Gemma3 (decoder) to Mammoth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Output:
  - Hybrid encoder-decoder model
  - Encoder: ModernBERT (pretrained weights)
  - Decoder: Gemma3 (pretrained weights)
  - Automatic dimension bridging if encoder_dim != decoder_dim
        """,
    )

    parser.add_argument(
        "encoder_model_path",
        help="HuggingFace ModernBERT model name or local path",
    )

    parser.add_argument(
        "decoder_model_path",
        help="HuggingFace Gemma3 model name or local path",
    )

    parser.add_argument(
        "save_path",
        help="Path to save converted model (e.g., './models/hybrid.pt')"
    )

    parser.add_argument(
        "--src-tokenizer",
        help="Path to source tokenizer.json (if not provided, uses encoder tokenizer)"
    )

    parser.add_argument(
        "--tgt-tokenizer",
        help="Path to target tokenizer.json (if not provided, uses decoder tokenizer)"
    )

    parser.add_argument(
        "--src-lang",
        default="en",
        help="Source language code (default: en)"
    )

    parser.add_argument(
        "--tgt-lang",
        default="ar",
        help="Target language code (default: ar)"
    )

    args = parser.parse_args()

    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    try:
        convert_hybrid_to_mammoth(
            encoder_model_path=args.encoder_model_path,
            decoder_model_path=args.decoder_model_path,
            save_path=args.save_path,
            src_tokenizer_path=args.src_tokenizer,
            tgt_tokenizer_path=args.tgt_tokenizer,
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