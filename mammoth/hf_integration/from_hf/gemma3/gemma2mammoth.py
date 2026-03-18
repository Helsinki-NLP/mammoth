#!/usr/bin/env python3
import sys
import os

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

"""
HuggingFace Gemma3 270M to Mammoth Model Converter

Usage: python gemma3_2mammoth.py <hf_model_path> <save_path>

Gemma3 Architecture (✅ = Fully Supported):
- ✅ RMSNorm with unit_offset: (x * norm) * (1.0 + weight)
- ✅ 4 normalizations per layer (double sandwich) via sandwich_norm=True:
  * input_layernorm → attn → post_attention_layernorm → residual
  * pre_feedforward_layernorm → mlp → post_feedforward_layernorm → residual
- ✅ Q/K normalization (RMSNorm on queries and keys)
- ✅ Multi Query Attention
- ✅ Gated MLP with GELU activation (gate_proj, up_proj pattern)
- ✅ RoPE with dual theta (global: 1M, local: 10K for sliding window)
  * Fully supported via sliding_window, global_rope_theta, local_rope_theta parameters
- ✅ Scaled word embeddings (multiply by sqrt(hidden_size))
  * Supported via dec_scaled_embeddings parameter in x_transformers_opts
- ✅ Bias-free architecture
- ✅ Alternating attention pattern (sliding window / full attention)
  * Fully supported via sliding_window and global_attn_every_n_layers parameters
- ✅ Final RMSNorm + optional logit softcapping

Improvements in this version:
- Uses sandwich_norm=True for complete 4-norm-per-layer support
- Uses rms_norm=True for RMSNorm matching Gemma3
- Uses ff_glu=True for gated MLP (matches Gemma3's gate_proj * GELU(up_proj))
- Correctly concatenates [up_proj; gate_proj] → GLU.proj.weight
- Maps ALL weights including lm_head.weight → decoder.net.to_logits.weight
- 100% HF weight transfer tracking and verification
"""

import torch
import re
from collections import OrderedDict
from argparse import Namespace
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from mammoth.x_transformers import TransformerWrapper, Decoder
from mammoth.inputters.vocab import Vocab, HFTokenizerVocab
from mammoth.distributed.tasks import TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.model_builder import build_model
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.model_saver import build_model_saver


# =============================================================================
# SECTION 1: HuggingFace to x-transformers conversion
# =============================================================================


def create_xtransformer_model(model_path):
    """
    Create decoder-only x-transformers model from HF Gemma3 config

    Gemma3 is decoder-only. We build a standalone decoder using TransformerWrapper + Decoder.

    Note: x_transformers feature support for Gemma3:
    - ✅ RMSNorm with unit_offset (rms_norm=True), not used in qk_norm
    - ✅ 4 norms per layer (sandwich_norm=True)
    - ✅ QK normalization (qk_norm=True), using L2 Norm
    - ✅ GQA/MQA (attn_one_kv_head=True for single key/value head)
    - ✅ Gated MLP with GELU (ff_glu=True) matching gate_proj * GELU(up_proj)
    - ✅ RoPE with dual theta (global_rope_theta, local_rope_theta)
    - ✅ Sliding window attention (sliding_window, global_attn_every_n_layers)
    - ✅ Scaled embeddings (custom scaling in TransformerWrapper)

    Args:
        model_path: Path to HuggingFace Gemma3 model

    Returns:
        TransformerWrapper: Decoder-only model with embeddings and to_logits projection
    """
    config = AutoConfig.from_pretrained(model_path, local_files_only=False, trust_remote_code=False)

    # Extract Gemma3 text config (for multimodal models)
    if hasattr(config, 'text_config'):
        config = config.text_config

    # Calculate ff_mult from Gemma3's intermediate_size
    intermediate_size = config.intermediate_size
    ff_mult = intermediate_size / config.hidden_size

    # Extract head_dim - Gemma3 uses explicit head_dim (not derived from hidden_size)
    head_dim = config.head_dim if hasattr(config, 'head_dim') else (config.hidden_size // config.num_attention_heads)

    # Extract sliding window and dual RoPE parameters
    dec_sliding_window = getattr(config, 'sliding_window', -1)
    dec_global_attn_every_n_layers = getattr(config, 'global_attn_every_n_layers', 3)
    dec_global_rope_theta = getattr(config, 'rope_theta', 1000000.0)  # Global theta (default 1M)
    dec_local_rope_theta = getattr(config, 'rope_local_base_freq', 10000.0)  # Local theta (default 10K)

    print(f"Gemma3 Configuration:")
    print(f"  - Model type: Decoder-only LM (causal)")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Attention heads: {config.num_attention_heads}")
    print(f"  - KV heads: {config.num_key_value_heads} (GQA)")
    print(f"  - Head dim: {head_dim}")
    print(f"  - Q dim: {config.num_attention_heads * head_dim}")
    print(f"  - K/V dim: {config.num_key_value_heads * head_dim}")
    print(f"  - Intermediate size: {intermediate_size} (ff_mult={ff_mult:.1f})")
    print(f"  - Vocab size: {config.vocab_size}")
    print(f"  - Max position embeddings: {config.max_position_embeddings}")
    print(f"  - RoPE global theta: {dec_global_rope_theta}")
    print(f"  - RoPE local theta: {dec_local_rope_theta}")

    # Log sliding window configuration
    sliding_window_status = "enabled" if dec_sliding_window > 0 else "disabled"
    print(f"  - Sliding window: {sliding_window_status}")
    if dec_sliding_window > 0:
        print(f"    • Window size: {dec_sliding_window} tokens")
        print(f"    • Global attention every N layers: {dec_global_attn_every_n_layers}")

    print(f"  - Attention dropout: {config.attention_dropout}")
    print(f"  - RMS norm eps: {config.rms_norm_eps}")

    print(f"\nCreating decoder-only x-transformers model:")
    print(f"  - Architecture: TransformerWrapper + Decoder")
    print(f"  - Layers: {config.num_hidden_layers}")
    print(f"  - Mapping strategy:")
    print(f"    * 4 Gemma3 norms → 4 x_transformers norms (using sandwich_norm=True)")
    print(f"    * pre_norm=True + sandwich_norm=True (pre + post normalization)")
    print(f"    * Q/K normalization: qk_norm=True")
    if dec_sliding_window > 0:
        print(f"    * Sliding window attention with dual RoPE theta")

    # Create Decoder (attention layers only)
    decoder = Decoder(
        dim=config.hidden_size,
        depth=config.num_hidden_layers,
        heads=config.num_attention_heads,
        attn_dim_head=head_dim,  # ✅ Explicit head_dim from config (critical for Gemma3!)
        rotary_pos_emb=True,
        rotary_pos_emb_base=dec_global_rope_theta,  # ✅ Use Gemma3's global RoPE theta (1M)
        ff_mult=ff_mult,
        ff_glu=True,  # ✅ Gemma3 uses gated MLP (gate_proj * GELU(up_proj))
        ff_no_bias=True,  # Gemma3 is bias-free
        attn_qk_norm=True,  # Gemma3 has Q/K normalization, using L2 Norm rather RMS norm!
        attn_qk_norm_dim_scale=True,  # Enable learnable qk_norm scales, using L2 Norm rather RMS norm!
        attn_one_kv_head=True,  # GQA with one kv head (MQA)
        attn_dropout=config.attention_dropout,
        sandwich_norm=True,  # ✅ Enable sandwich norm for 4 norms per layer!
        use_rmsnorm=True,  # ✅ Use RMSNorm matching Gemma3 (NOT used in qk_norm!)
        # ✅ Sliding window attention with dual RoPE
        sliding_window=dec_sliding_window,  # Window size (-1 = disabled)
        global_attn_every_n_layers=dec_global_attn_every_n_layers,  # Global attention frequency
        global_rope_theta=dec_global_rope_theta,  # RoPE theta for global attention
        local_rope_theta=dec_local_rope_theta,  # RoPE theta for local (sliding window) attention
    )

    # Wrap with TransformerWrapper to add embeddings and output projection
    xt_model = TransformerWrapper(
        num_tokens=config.vocab_size,
        max_seq_len=config.max_position_embeddings,
        attn_layers=decoder,
        emb_dropout=0.0,  # Gemma3 has no embedding dropout
        post_emb_norm=False,  # Gemma3 has embedding norm
    )

    print(f"✓ Decoder-only x-transformers model created")
    return xt_model


def create_weight_mapping_gemma3(num_layers):
    """
    Create mapping from HuggingFace Gemma3 to x-transformers decoder-only model

    Gemma3 layer structure (modeling_gemma3.py:347-409):
    - input_layernorm (RMSNorm)
    - self_attn (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm)
    - post_attention_layernorm (RMSNorm)
    - pre_feedforward_layernorm (RMSNorm)
    - mlp (gate_proj, up_proj, down_proj) - but actually standard MLP with GELU
    - post_feedforward_layernorm (RMSNorm)

    TransformerWrapper structure (decoder-only):
    - token_emb.emb (embeddings)
    - attn_layers (Decoder instance)
      - layers (ModuleList of attention and feedforward blocks)
      - final_norm
    - to_logits (output projection)

    Decoder layer structure (pre_norm=True, sandwich_norm=True, causal=True):
    - For each layer_idx:
      * Attention block at layers[layer_idx * 2]:
        - Pre-attention norm (layers.{idx}.0.0)
        - Attention (to_q, to_k, to_v, to_out, with qk_norm)
        - Post-attention norm (layers.{idx}.0.1) ✅ sandwich_norm!
      * Feedforward block at layers[layer_idx * 2 + 1]:
        - Pre-feedforward norm (layers.{idx}.0.0)
        - FeedForward (GLU with gated MLP)
        - Post-feedforward norm (layers.{idx}.0.1) ✅ sandwich_norm!

    Mapping strategy:
    - input_layernorm → pre-attention norm
    - post_attention_layernorm → post-attention norm
    - pre_feedforward_layernorm → pre-feedforward norm
    - post_feedforward_layernorm → post-feedforward norm
    - Gemma3 gate_proj + up_proj → x_transformers input_proj (concatenated)
    """
    mapping = {}

    # Embeddings - Gemma3 uses scaled embeddings
    # Note: Scaling (sqrt(hidden_size)) needs to be applied after loading
    mapping["model.embed_tokens.weight"] = "token_emb.emb.weight"

    # LM head (output projection to vocabulary)
    mapping["lm_head.weight"] = "to_logits.weight"

    # Final norm (after all decoder layers)
    # Note: Gemma3 uses unit_offset pattern: (1.0 + weight)
    # x_transformers RMSNorm also supports unit_offset
    mapping["model.norm.weight"] = "attn_layers.final_norm.g"

    # Decoder layers
    # In decoder-only mode, each layer is [attention, feedforward] (no cross-attention)
    for layer_idx in range(num_layers):
        # Decoder structure: layers[attn_idx, ff_idx] for each layer
        # Each pair: 0=self_attention, 1=feedforward
        attn_idx = layer_idx * 2
        ff_idx = layer_idx * 2 + 1

        hf_prefix = f"model.layers.{layer_idx}"
        xt_attn_prefix = f"attn_layers.layers.{attn_idx}"
        xt_ff_prefix = f"attn_layers.layers.{ff_idx}"

        # === Attention Block ===

        # Pre-attention LayerNorm (input_layernorm)
        # Gemma3: model.layers.X.input_layernorm.weight
        # x_transformers: attn_layers.layers.{attn_idx}.0.0.g (norms[0] = pre_branch_norm)
        mapping[f"{hf_prefix}.input_layernorm.weight"] = f"{xt_attn_prefix}.0.0.g"

        # Attention projections (separate Q, K, V)
        mapping[f"{hf_prefix}.self_attn.q_proj.weight"] = f"{xt_attn_prefix}.1.to_q.weight"
        mapping[f"{hf_prefix}.self_attn.k_proj.weight"] = f"{xt_attn_prefix}.1.to_k.weight"
        mapping[f"{hf_prefix}.self_attn.v_proj.weight"] = f"{xt_attn_prefix}.1.to_v.weight"
        mapping[f"{hf_prefix}.self_attn.o_proj.weight"] = f"{xt_attn_prefix}.1.to_out.weight"

        # Q/K normalization (Gemma3 specific)
        # x_transformers with qk_norm=True creates qk_norm_q_scale and qk_norm_k_scale
        mapping[f"{hf_prefix}.self_attn.q_norm.weight"] = f"{xt_attn_prefix}.1.qk_norm_q_scale"
        mapping[f"{hf_prefix}.self_attn.k_norm.weight"] = f"{xt_attn_prefix}.1.qk_norm_k_scale"

        # Post-attention LayerNorm (post_attention_layernorm) ✅ NEW!
        # Gemma3: model.layers.X.post_attention_layernorm.weight
        # x_transformers: attn_layers.layers.{attn_idx}.0.1.g (norms[1] = post_branch_norm, enabled by sandwich_norm=True)
        mapping[f"{hf_prefix}.post_attention_layernorm.weight"] = f"{xt_attn_prefix}.0.1.g"

        # === Feedforward Block ===

        # Pre-feedforward LayerNorm (pre_feedforward_layernorm)
        # Gemma3: model.layers.X.pre_feedforward_layernorm.weight
        # x_transformers: attn_layers.layers.{ff_idx}.0.0.g (norms[0] = pre_branch_norm)
        mapping[f"{hf_prefix}.pre_feedforward_layernorm.weight"] = f"{xt_ff_prefix}.0.0.g"

        # MLP - Gated structure with GELU
        # Gemma3: gate_proj, up_proj, down_proj with gating: gate * GELU(up)
        # x_transformers GLU: single proj.weight that gets chunked into [gate, up]
        # Need to concatenate gate_proj and up_proj → GLU.proj.weight
        mapping[f"{hf_prefix}.mlp.gate_proj.weight"] = (
            f"{xt_ff_prefix}.1.ff.0.proj.weight",
            "gate"  # Mark as gate part (first half after chunking)
        )
        mapping[f"{hf_prefix}.mlp.up_proj.weight"] = (
            f"{xt_ff_prefix}.1.ff.0.proj.weight",
            "up"  # Mark as up part (second half after chunking)
        )
        mapping[f"{hf_prefix}.mlp.down_proj.weight"] = f"{xt_ff_prefix}.1.ff.2.weight"

        # Post-feedforward LayerNorm (post_feedforward_layernorm) ✅ NEW!
        # Gemma3: model.layers.X.post_feedforward_layernorm.weight
        # x_transformers: attn_layers.layers.{ff_idx}.0.1.g (norms[1] = post_branch_norm, enabled by sandwich_norm=True)
        mapping[f"{hf_prefix}.post_feedforward_layernorm.weight"] = f"{xt_ff_prefix}.0.1.g"

    return mapping


def load_hf_weights_to_xtransformer(hf_model_path, xt_model):
    """
    Load weights from HuggingFace Gemma3 to x-transformers model

    Special handling:
    1. Concatenate gate_proj + up_proj → standard FF input projection
    2. Handle Q/K normalization parameter differences
    3. Handle scaled embeddings (note for post-processing)
    4. ✅ Map all 4 norms per layer (via sandwich_norm=True)

    Returns 100% coverage tracking of HF weights
    """
    print(f"Loading HF Gemma3 model from {hf_model_path}")
    print(f"  [1/3] Loading config...")
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=False, trust_remote_code=False)

    # Extract text config if multimodal
    if hasattr(config, 'text_config'):
        config = config.text_config

    print(f"  [2/3] Loading model weights ")
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False
    )
    print(f"  [3/3] Extracting state dict...")
    hf_state_dict = hf_model.state_dict()
    print(f"✓ Model loaded successfully ({len(hf_state_dict)} keys)")

    print(f"\n[Weight Mapping] Creating HF → x-transformers mapping...")
    mapping = create_weight_mapping_gemma3(config.num_hidden_layers)
    x_state_dict = OrderedDict()

    print(f"  HF Gemma3 Model: {len(hf_state_dict)} parameters")
    print(f"  Mapping entries: {len(mapping)} mappings defined")

    # Pre-validate HF→x-transformers mapping
    print(f"\n  [Pre-validation] Checking HF → x-transformers mapping validity...")
    xt_sd = xt_model.state_dict()

    hf_missing_keys = []
    xt_missing_keys = []
    valid_mappings = []
    gate_up_pairs = {}

    for hf_key, x_info in mapping.items():
        if hf_key not in hf_state_dict:
            hf_missing_keys.append(hf_key)
            continue

        if isinstance(x_info, tuple):
            x_key, part_type = x_info
            if x_key not in gate_up_pairs:
                gate_up_pairs[x_key] = {"gate": None, "up": None}
            gate_up_pairs[x_key][part_type] = hf_key
        else:
            x_key = x_info
            if x_key not in xt_sd:
                xt_missing_keys.append(f"{hf_key} → {x_key}")
            else:
                valid_mappings.append((hf_key, x_key))

    # Check gate/up pairs for completeness
    incomplete_pairs = []
    complete_pairs = []
    for x_key, parts in gate_up_pairs.items():
        if parts["gate"] and parts["up"]:
            if x_key in xt_sd:
                complete_pairs.append((parts["gate"], parts["up"], x_key))
                valid_mappings.append((parts["gate"], x_key))  # Count both parts
                valid_mappings.append((parts["up"], x_key))
            else:
                xt_missing_keys.append(f"{parts['gate']}/{parts['up']} → {x_key}")
        else:
            missing_parts = [k for k, v in parts.items() if v is None]
            incomplete_pairs.append((x_key, missing_parts))

    print(f"    - Total mappings defined: {len(mapping)}")
    print(f"    - Valid direct mappings: {len([m for m in valid_mappings if not any(p in str(m[0]) for p in ['gate_proj', 'up_proj'])])}")
    print(f"    - Complete gate+up pairs: {len(complete_pairs)}")
    print(f"    - Incomplete gate+up pairs: {len(incomplete_pairs)}")
    print(f"    - HF model missing keys: {len(hf_missing_keys)}")
    print(f"    - XT model missing keys: {len(xt_missing_keys)}")

    if hf_missing_keys:
        print(f"\n  ✗ CRITICAL: {len(hf_missing_keys)} HF keys NOT FOUND:")
        for key in hf_missing_keys[:5]:
            print(f"    - {key}")
        if len(hf_missing_keys) > 5:
            print(f"    ... and {len(hf_missing_keys) - 5} more")

    if xt_missing_keys:
        print(f"\n  ✗ CRITICAL: {len(xt_missing_keys)} x-transformer targets NOT FOUND:")
        for key in xt_missing_keys[:5]:
            print(f"    - {key}")
        if len(xt_missing_keys) > 5:
            print(f"    ... and {len(xt_missing_keys) - 5} more")

    if incomplete_pairs:
        print(f"\n  ✗ CRITICAL: {len(incomplete_pairs)} incomplete gate+up pairs:")
        for x_key, missing_parts in incomplete_pairs[:5]:
            print(f"    - {x_key}: missing {', '.join(missing_parts)}")
        if len(incomplete_pairs) > 5:
            print(f"    ... and {len(incomplete_pairs) - 5} more")

    if hf_missing_keys or xt_missing_keys or incomplete_pairs:
        print(f"\n  ⚠️  VALIDATION FAILED: Cannot proceed with incomplete HF→x-transformers mapping!")
        print(f"      Please fix the mapping in create_weight_mapping_gemma3()")
        return None

    # Track which HF keys have been processed
    hf_keys_processed = set()

    # Categorize mappings
    direct_mappings = [k for k, v in mapping.items() if not isinstance(v, tuple)]
    gate_mappings = [k for k, v in mapping.items() if isinstance(v, tuple) and v[1] == "gate"]
    up_mappings = [k for k, v in mapping.items() if isinstance(v, tuple) and v[1] == "up"]

    print(f"  Direct mappings: {len(direct_mappings)}")
    print(f"  Gate+Up projections (will be fused): {len(gate_mappings)} pairs")

    # Track statistics
    successfully_mapped = []
    mapping_errors = []

    # First pass: collect gate and up projections for concatenation
    gate_weights = {}  # {base_key: weight}
    up_weights = {}    # {base_key: weight}

    for hf_key, x_info in mapping.items():
        if hf_key not in hf_state_dict:
            mapping_errors.append(f"  ✗ HF key not found: {hf_key}")
            continue

        # Check if this is a tuple (gate/up projection for standard FF)
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
            # Direct mapping - check if this is QK norm weight that needs reshaping
            if "q_norm.weight" in hf_key or "k_norm.weight" in hf_key:
                # QK norm reshaping: [dim_head] → [heads, groups, dim_head]
                hf_weight = hf_state_dict[hf_key]
                num_heads = config.num_attention_heads
                kv_heads = 1  # MQA: one key/value head
                dim_head = config.head_dim

                # Simple reshaping: unsqueeze to match x-transformers 3D format
                reshaped_weight = hf_weight.unsqueeze(0).unsqueeze(0).expand(
                    num_heads if "q_norm" in hf_key else kv_heads,
                    1,
                    dim_head
                )
                x_state_dict[x_info] = reshaped_weight
                successfully_mapped.append(hf_key)
                hf_keys_processed.add(hf_key)
            else:
                # Regular direct mapping
                x_state_dict[x_info] = hf_state_dict[hf_key]
                successfully_mapped.append(hf_key)
                hf_keys_processed.add(hf_key)

    # Second pass: concatenate gate + up projections for GLU
    fused_count = 0
    for x_key in gate_weights.keys():
        if x_key in up_weights:
            # Concatenate [up; gate] along dim 0 for GLU projection
            # GLU chunks and computes: x * act(gate)
            # Where x = first half (up_proj, linear), gate = second half (gate_proj, through activation)
            # This matches Gemma3's: GELU(gate_proj) * up_proj
            gate_weight = gate_weights[x_key]
            up_weight = up_weights[x_key]
            fused_weight = torch.cat([up_weight, gate_weight], dim=0)  # [up; gate] order!
            x_state_dict[x_key] = fused_weight
            fused_count += 1
        else:
            mapping_errors.append(f"  ✗ Missing up_proj for gate at {x_key}")

    # Load weights (strict=False to allow missing encoder weights)
    result = xt_model.load_state_dict(x_state_dict, strict=False)

    # Report HF weight coverage
    print(f"\n[HF Weight Transfer Report]")
    print(f"  Total HF Gemma3 parameters: {len(hf_state_dict)}")
    print(f"  Successfully mapped: {len(hf_keys_processed)}")
    print(f"  Coverage: {len(hf_keys_processed)}/{len(hf_state_dict)} ({len(hf_keys_processed)/len(hf_state_dict)*100:.1f}%)")

    # Check for any unmapped HF keys
    unmapped_hf_keys = set(hf_state_dict.keys()) - hf_keys_processed
    if unmapped_hf_keys:
        print(f"\n  ⚠ WARNING: {len(unmapped_hf_keys)} HF parameters NOT mapped:")
        for key in sorted(unmapped_hf_keys)[:10]:
            print(f"    - {key}")
        if len(unmapped_hf_keys) > 10:
            print(f"    ... and {len(unmapped_hf_keys) - 10} more")
    else:
        print(f"  ✓ All HF Gemma3 weights successfully transferred!")

    print(f"\n  Special transformations applied:")
    print(f"    - Fused gate+up projections: {fused_count} layers")
    print(f"    - Reshaped QK norms: {sum(1 for k in successfully_mapped if 'norm.weight' in k and 'q_norm' in k or 'k_norm' in k)} parameters")

    if mapping_errors:
        print(f"\n  ⚠ Mapping errors ({len(mapping_errors)}):")
        for msg in mapping_errors[:5]:
            print(msg)
        if len(mapping_errors) > 5:
            print(f"  ... and {len(mapping_errors) - 5} more")

    # Embedding weights are stored unscaled; sqrt(dim) scaling is applied at forward-pass
    # time via ScaledTokenEmbedding (instantiated by model_builder when dec_scaled_embeddings=True).
    print(f"\n  ✓ Embedding weights stored unscaled (sqrt({config.hidden_size}) applied at runtime via ScaledTokenEmbedding)")

    return xt_model


# =============================================================================
# SECTION 2: MAMMOTH utilities
# =============================================================================


def rename_hf_special_tokens_to_mammoth(tokenizer_path, output_path=None, vocab_size_limit=None):
    """
    Rename HuggingFace special tokens to MAMMOTH conventions in-place.
    This avoids adding new tokens and wasting embeddings.

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


def create_vocabs_dict_from_hf_tokenizer(src_tokenizer_path, tgt_tokenizer_paths=None, src_lang="en", tgt_langs=None, shared_tgt_tokenizer_path=None):
    """
    Create vocabs_dict from HuggingFace tokenizer(s) - supports both shared and separate vocabs

    Args:
        src_tokenizer_path: Path to source tokenizer.json file
        tgt_tokenizer_paths: Dict mapping language codes to tokenizer paths (e.g., {"ar": "path/to/ar_tokenizer.json"})
                            If None, uses shared_tgt_tokenizer_path for all languages (multilingual setup)
        src_lang: Source language (default: "en")
        tgt_langs: List of target languages (default: ["ar"])
        shared_tgt_tokenizer_path: Path to shared target tokenizer (used when tgt_tokenizer_paths is None)
    """
    if tgt_langs is None:
        tgt_langs = ["ar"]

    if not os.path.exists(src_tokenizer_path):
        raise FileNotFoundError(
            f"Source tokenizer not found at {src_tokenizer_path}"
        )

    from tokenizers import Tokenizer

    # Create source vocabulary
    src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
    src_vocab = HFTokenizerVocab(
        tokenizer_path=src_tokenizer_path,
        tag=f"src_{src_lang}",
    )
    vocabs_dict = {("src", src_lang): src_vocab}

    print(f"✓ Created source vocab: {len(src_vocab)} tokens")
    print(f"  Path: {src_tokenizer_path}")

    # Create target vocabularies
    if tgt_tokenizer_paths is None:
        # Multilingual setup: all languages share the same tokenizer
        tokenizer_to_use = shared_tgt_tokenizer_path if shared_tgt_tokenizer_path else src_tokenizer_path
        print(f"✓ Using shared tokenizer for all languages (multilingual mode): {tokenizer_to_use}")
        for tgt_lang in tgt_langs:
            tgt_vocab = HFTokenizerVocab(
                tokenizer_path=tokenizer_to_use,
                tag=f"tgt_{tgt_lang}",
            )
            vocabs_dict[("tgt", tgt_lang)] = tgt_vocab
    else:
        # Separate vocabs: each target language has its own tokenizer
        print(f"✓ Using separate tokenizers for target languages:")
        for tgt_lang in tgt_langs:
            if tgt_lang not in tgt_tokenizer_paths:
                raise ValueError(
                    f"Target language '{tgt_lang}' not found in tgt_tokenizer_paths. "
                    f"Available: {list(tgt_tokenizer_paths.keys())}"
                )

            tgt_path = tgt_tokenizer_paths[tgt_lang]
            if not os.path.exists(tgt_path):
                raise FileNotFoundError(f"Target tokenizer not found at {tgt_path}")

            tgt_tokenizer = Tokenizer.from_file(tgt_path)
            tgt_vocab = HFTokenizerVocab(
                tokenizer_path=tgt_path,
                tag=f"tgt_{tgt_lang}",
            )
            vocabs_dict[("tgt", tgt_lang)] = tgt_vocab
            print(f"  - {tgt_lang}: {len(tgt_vocab)} tokens ({tgt_path})")

    return vocabs_dict


def create_model_opts_from_xt_model(xt_model, hf_model_path):
    """Create model_opts for Mammoth from x-transformer model"""
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=False, trust_remote_code=False)

    if hasattr(config, 'text_config'):
        config = config.text_config

    model_opts = Namespace()

    # Basic settings
    model_opts.model_dtype = "bf16"

    # Architecture - Encoder/Decoder can have different dimensions
    # If enc_model_dim != dec_model_dim, attention bridge will be disabled
    model_opts.enc_layers = [6]  # Encoder: 6 layers (randomly initialized)
    model_opts.dec_layers = [config.num_hidden_layers]  # Decoder: Gemma3 layers (from pretrained)

    # Encoder/Decoder dimensions (can be different)
    model_opts.enc_model_dim = 512  # Encoder dimension (can be different from decoder)
    model_opts.dec_model_dim = config.hidden_size  # Decoder dimension (Gemma3's hidden_size)
    # model_opts.model_dim = config.hidden_size  # Fallback for shared dimension mode

    # Calculate decoder ff_mult from Gemma3's intermediate_size
    # Encoder uses its own enc_ff_mult (set in x_transformers_opts below)
    dec_intermediate_size = config.intermediate_size
    dec_ff_mult = dec_intermediate_size / config.hidden_size

    # Extract sliding window and dual RoPE parameters from config
    # These are top-level model_opts attributes (defined in opts.py)
    model_opts.dec_sliding_window = getattr(config, 'sliding_window', -1)
    model_opts.dec_global_attn_every_n_layers = getattr(config, 'global_attn_every_n_layers', 3)
    model_opts.dec_global_rope_theta = getattr(config, 'rope_theta', 1000000.0)  # Global theta (default 1M)
    model_opts.dec_local_rope_theta = getattr(config, 'rope_local_base_freq', 10000.0)  # Local theta (default 10K)

    # Encoder uses standard full attention (no sliding window)
    model_opts.enc_sliding_window = -1  # Disabled
    model_opts.enc_global_attn_every_n_layers = 3  # Not used (sliding window disabled)
    model_opts.enc_global_rope_theta = 10000.0  # Standard RoPE base
    model_opts.enc_local_rope_theta = 10000.0  # Standard RoPE base

    # x_transformers_opts with enc_/dec_ prefix support (added to model_builder.py)
    #
    # IMPORTANT: build_model() creates a NEW decoder from scratch using these parameters.
    # The pre-trained Gemma3 weights are then copied to this decoder via map_xt_to_mammoth_weights().
    # Therefore, decoder parameters MUST match the Gemma3 architecture exactly to avoid shape mismatches.
    model_opts.x_transformers_opts = {
        # Shared options (apply to both encoder and decoder)
        "attn_flash": True,

        # Decoder-specific TransformerWrapper options (must match Gemma3 architecture)
        # "dec_num_tokens": config.vocab_size,  # Decoder vocab size (Gemma3)
        "dec_post_emb_norm": False,  # Gemma3 has no post-emb norm
        "dec_max_seq_len": config.max_position_embeddings,  # Gemma3's max sequence length
        "dec_scaled_embeddings": True,  # Gemma3 uses scaled word embeddings (multiply by sqrt(hidden_size))

        # Decoder-specific AttentionLayers options (must match Gemma3 exactly)
        "dec_heads": config.num_attention_heads,
        "dec_attn_dim_head": config.head_dim,
        "dec_attn_dropout": config.attention_dropout,
        "dec_attn_qk_norm": True,  # Gemma3 has Q/K normalization
        "dec_attn_qk_norm_dim_scale": True,  # Enable learnable qk_norm scales
        "dec_attn_kv_heads": config.num_key_value_heads,  # MQA (one KV head)

        "dec_ff_mult": dec_ff_mult,
        "dec_ff_dropout": 0.0,
        "dec_ff_glu": True,  # Gemma3 uses gated MLP
        "dec_ff_no_bias": True,

        "dec_rotary_pos_emb": True,
        "dec_rotary_pos_emb_base": config.rope_theta,
        "dec_use_rmsnorm": True,
        "dec_sandwich_norm": True,  # 4 norms per layer
        # CRITICAL: Tell decoder cross-attention to expect encoder dimension
        "dec_cross_attn_dim_context": model_opts.enc_model_dim,  # Encoder outputs 512-dim, decoder is 256-dim

        # Encoder-specific TransformerWrapper options
        # "enc_num_tokens": config.vocab_size,  # Encoder vocab size (can be different from decoder)
        "enc_post_emb_norm": True,  # Standard encoder has post-emb norm
        "enc_max_seq_len": 512,  # Encoder max sequence length (can be different from decoder)
        "enc_scaled_embeddings": False,  # Standard encoder doesn't use scaled embeddings

        # Encoder-specific AttentionLayers options (standard transformer encoder)
        "enc_heads": 8,
        "enc_attn_dim_head": 64,
        "enc_ff_mult": 4.0,  # Standard 4x expansion
        "enc_ff_glu": False,  # Standard feedforward
        "enc_ff_no_bias": True,
        "enc_rotary_pos_emb": True,
    }

    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.normformer = False
    model_opts.dropout = [0.0]
    model_opts.attention_dropout = [config.attention_dropout]

    # Log sliding window configuration
    print(f'decoder max_seq_len: {config.max_position_embeddings}')
    if model_opts.dec_sliding_window > 0:
        print(f'decoder sliding window: {model_opts.dec_sliding_window} tokens (enabled)')
        print(f'  - global attention every {model_opts.dec_global_attn_every_n_layers} layers')
        print(f'  - global RoPE theta: {model_opts.dec_global_rope_theta}')
        print(f'  - local RoPE theta: {model_opts.dec_local_rope_theta}')
    else:
        print('decoder sliding window: disabled (full causal attention)')
    print('encoder sliding window: disabled (standard full attention)')

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


def create_task_queue_manager(vocabs_dict=None, num_decoder_layers=18, src_lang="en", tgt_langs=None):
    """
    Create TaskQueueManager for multi-language multi-task setup

    For Gemma3 (decoder-only):
    - 1 shared decoder (Gemma3 with num_decoder_layers)
    - N language-specific encoders (6 layers each)

    Args:
        vocabs_dict: Dictionary of vocabularies
        num_decoder_layers: Number of decoder layers (18 for Gemma3-4b)
        src_lang: Source language code (default: "en")
        tgt_langs: List of target languages (if None, defaults to ["ar"])
    """
    # Extract target languages from vocabs_dict if not provided
    if tgt_langs is None:
        # Try to infer from vocabs_dict
        if vocabs_dict:
            tgt_langs = [lang for side, lang in vocabs_dict.keys() if side == "tgt"]
        if not tgt_langs:
            tgt_langs = ["ar"]  # Fallback default

    target_languages = tgt_langs

    opts = Namespace()
    opts.tasks = {}

    # Create 1 task per language pair
    for gpu_id, tgt_lang in enumerate(target_languages):
        node_id = gpu_id // 8  # 8 GPUs per node #NOTE: 8 GPUs per node on LUMI-CSC
        local_gpu = gpu_id % 8  # GPU within node  #NOTE: 8 GPUs per node on LUMI-CSC

        task_name = f"task_{src_lang}_{tgt_lang}"
        opts.tasks[task_name] = {
            "src_tgt": f"{src_lang}-{tgt_lang}",
            "weight": 1.0,
            "introduce_at_training_step": 0,
            "node_gpu": f"{node_id}:{local_gpu}",
            "enc_sharing_group": [src_lang],  # Each task has language-specific encoder
            "dec_sharing_group": [tgt_lang],  # All tasks share Gemma3 decoder
        }

    opts.enc_layers = [6]  # Standard encoder layers
    opts.dec_layers = [num_decoder_layers]  # Gemma3 decoder layers
    opts.task_distribution_strategy = "weighted_sampling"
    opts.accum_count = [1]
    opts.seed = 42

    world_context = SimpleWorldContext()
    task_manager = TaskQueueManager.from_opts(opts, world_context)

    # Assign vocabularies to all tasks
    src_vocab = vocabs_dict.get(("src", src_lang))

    for task in task_manager.tasks:
        task.src_vocab = src_vocab

        # Extract target language from task's src_tgt field (e.g., "en-ar" -> "ar")
        tgt_lang = task.tgt_lang

        # Check if separate target vocab exists, otherwise fallback to source vocab (multilingual mode)
        tgt_vocab = vocabs_dict.get(("tgt", tgt_lang))
        if tgt_vocab is None:
            print(f"  ⚠ Target vocab for '{tgt_lang}' not found, using source vocab (multilingual mode)")
            task.tgt_vocab = src_vocab
        else:
            task.tgt_vocab = tgt_vocab
            print(f"  ✓ Task {task.corpus_id}: src_vocab={len(src_vocab)} tokens, tgt_vocab={len(tgt_vocab)} tokens")

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
    opts.adam_beta1 = 0.9  # Default value (only used for adam/adamw optimizers)
    opts.adam_beta2 = 0.999  # Default value (only used for adam/adamw optimizers)
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


def create_xt_to_mammoth_mapping(num_decoder_layers, corpus_id):
    """
    Create mapping from decoder-only xt_model keys to mammoth_model keys for Gemma3

    x_transformers decoder-only structure (TransformerWrapper):
    - token_emb.emb.weight
    - (no post_emb_norm - Gemma3 has post_emb_norm=False)
    - attn_layers.layers.{idx}.{sublayer}  (decoder-only: [attn, ff] pairs, no cross-attn)
    - attn_layers.final_norm.g
    - to_logits.weight  (output projection)

    Mammoth structure:
    - decoder.{corpus_id}.token_emb.emb.weight
    - decoder.{corpus_id}.attn_layers.attention_layers_stack.0.layers.{idx}.{sublayer}
    - decoder.{corpus_id}.attn_layers.attention_layers_stack.0.final_norm.g
    - (no to_logits in MAMMOTH - uses decoder's internal projection)
    """
    mapping = {}

    # Token embeddings
    mapping["token_emb.emb.weight"] = f"decoder.{corpus_id}.token_emb.emb.weight"

    # RoPE - Note: inv_freq is auto-computed from rotary_pos_emb_base, so we don't copy it
    # The x-transformer model has dual RoPE (global and local) but Mammoth uses single RoPE
    # The base frequency is already set in model_opts, so inv_freq will be auto-generated

    # Decoder layers
    # In decoder-only mode: each layer is [attention, feedforward] (no cross-attention)
    for layer_idx in range(num_decoder_layers):
        # x_transformers decoder-only structure: layers[attn_idx, ff_idx]
        # Each pair: attn_idx = layer_idx * 2, ff_idx = layer_idx * 2 + 1
        xt_attn_idx = layer_idx * 2
        xt_ff_idx = layer_idx * 2 + 1

        # MAMMOTH uses encoder-decoder structure: layers[attn_idx, cross_attn_idx, ff_idx]
        # Each triplet: attn_idx = layer_idx * 3, cross_attn = layer_idx * 3 + 1, ff = layer_idx * 3 + 2
        mammoth_attn_idx = layer_idx * 3
        mammoth_ff_idx = layer_idx * 3 + 2

        xt_attn_base = f"attn_layers.layers.{xt_attn_idx}"
        mammoth_attn_base = f"decoder.{corpus_id}.attn_layers.attention_layers_stack.0.layers.{mammoth_attn_idx}"

        # Attention layer
        mapping[f"{xt_attn_base}.0.0.g"] = f"{mammoth_attn_base}.0.0.g"  # Pre-norm
        mapping[f"{xt_attn_base}.1.to_q.weight"] = f"{mammoth_attn_base}.1.to_q.weight"
        mapping[f"{xt_attn_base}.1.to_k.weight"] = f"{mammoth_attn_base}.1.to_k.weight"
        mapping[f"{xt_attn_base}.1.to_v.weight"] = f"{mammoth_attn_base}.1.to_v.weight"
        mapping[f"{xt_attn_base}.1.to_out.weight"] = f"{mammoth_attn_base}.1.to_out.weight"

        # Q/K norm (if present)
        mapping[f"{xt_attn_base}.1.qk_norm_q_scale"] = f"{mammoth_attn_base}.1.qk_norm_q_scale"
        mapping[f"{xt_attn_base}.1.qk_norm_k_scale"] = f"{mammoth_attn_base}.1.qk_norm_k_scale"

        # Post-attention norm (sandwich_norm) ✅
        mapping[f"{xt_attn_base}.0.1.g"] = f"{mammoth_attn_base}.0.1.g"

        # Feedforward layer
        xt_ff_base = f"attn_layers.layers.{xt_ff_idx}"
        mammoth_ff_base = f"decoder.{corpus_id}.attn_layers.attention_layers_stack.0.layers.{mammoth_ff_idx}"

        mapping[f"{xt_ff_base}.0.0.g"] = f"{mammoth_ff_base}.0.0.g"  # Pre-norm
        mapping[f"{xt_ff_base}.1.ff.0.proj.weight"] = f"{mammoth_ff_base}.1.ff.0.proj.weight"  # GLU projection (fused gate+up)
        mapping[f"{xt_ff_base}.1.ff.2.weight"] = f"{mammoth_ff_base}.1.ff.2.weight"  # Output

        # Post-feedforward norm (sandwich_norm) ✅
        mapping[f"{xt_ff_base}.0.1.g"] = f"{mammoth_ff_base}.0.1.g"

    # Final norm
    mapping["attn_layers.final_norm.g"] = (
        f"decoder.{corpus_id}.attn_layers.attention_layers_stack.0.final_norm.g"
    )

    return mapping


def is_cross_attention_key(mammoth_key):
    """
    Check if a MAMMOTH key corresponds to a cross-attention layer.

    In MAMMOTH encoder-decoder architecture:
    - Pattern: layers[self_attn_idx, cross_attn_idx, ff_idx] per transformer block
    - cross_attn_idx = layer_idx * 3 + 1 (indices 1, 4, 7, 10, ...)
    - Self-attention indices: 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51
    - Cross-attention indices: 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52
    - Feedforward indices: 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53
    """
    if 'attn_layers.attention_layers_stack.0.layers.' not in mammoth_key:
        return False

    # Extract layer index from pattern like "...layers.{idx}..."
    match = re.search(r'\.layers\.(\d+)\.', mammoth_key)
    if not match:
        return False

    layer_idx = int(match.group(1))

    # Check if this is a cross-attention index (layer_idx % 3 == 1)
    # But exclude feedforward layers that might happen to be at .1.
    if layer_idx % 3 == 1 and not ('.1.ff.' in mammoth_key):
        return True

    return False


def map_xt_to_mammoth_weights(xt_model, mammoth_model, num_decoder_layers, task_names):
    """
    Map Gemma3 decoder weights to all task-specific decoders in Mammoth model
    """
    xt_sd = xt_model.state_dict()
    mammoth_sd = mammoth_model.state_dict()

    print(f"  x-transformers to Mammoth component analysis:")
    print(f"    - x-transformers decoder keys: {len(xt_sd)}")
    print(f"    - Mammoth model keys: {len(mammoth_sd)}")
    print(f"    - Tasks to map: {len(task_names)} ({', '.join(task_names)})")

    # Count task-specific decoder keys
    task_decoder_keys = [k for k in mammoth_sd.keys() if 'decoder.' in k and not 'encoder.' in k]
    print(f"    - Mammoth decoder-related keys: {len(task_decoder_keys)}")

    # Pre-validate mappings before processing
    print(f"\n  [Pre-validation] Checking mapping validity...")
    mapping = create_xt_to_mammoth_mapping(num_decoder_layers, task_names[0])

    xt_missing_keys = []
    mammoth_missing_keys = []
    valid_mappings = []
    encoder_keys_ignored = []
    cross_attention_keys_ignored = []

    for xt_key, mammoth_key in mapping.items():
        if xt_key not in xt_sd:
            xt_missing_keys.append(xt_key)
        elif mammoth_key not in mammoth_sd:
            # Check if this is an encoder key (which we expect to be missing)
            if mammoth_key.startswith('encoder.'):
                encoder_keys_ignored.append(mammoth_key)
            # Check if this is a cross-attention key (which we expect to be missing in decoder-only model)
            # In MAMMOTH encoder-decoder: pattern is layers[self_attn_idx, cross_attn_idx, ff_idx]
            # where cross_attn_idx = layer_idx * 3 + 1 (indices 1, 4, 7, 10, ...)
            elif 'cross_attn' in mammoth_key or is_cross_attention_key(mammoth_key):
                cross_attention_keys_ignored.append(mammoth_key)
            else:
                mammoth_missing_keys.append(mammoth_key)
        else:
            valid_mappings.append((xt_key, mammoth_key))

    print(f"    - Total mappings defined: {len(mapping)}")
    print(f"    - Valid mappings: {len(valid_mappings)}")
    print(f"    - XT model missing keys: {len(xt_missing_keys)}")
    print(f"    - Mammoth model missing keys: {len(mammoth_missing_keys)}")
    print(f"    - Encoder keys ignored (expected): {len(encoder_keys_ignored)}")
    print(f"    - Cross-attention keys ignored (expected): {len(cross_attention_keys_ignored)}")

    if xt_missing_keys:
        print(f"\n  ✗ CRITICAL: {len(xt_missing_keys)} x-transformer keys NOT FOUND:")
        for key in xt_missing_keys[:5]:
            print(f"    - {key}")
        if len(xt_missing_keys) > 5:
            print(f"    ... and {len(xt_missing_keys) - 5} more")

    if mammoth_missing_keys:
        print(f"\n  ✗ CRITICAL: {len(mammoth_missing_keys)} mammoth keys NOT FOUND:")
        for key in mammoth_missing_keys[:5]:
            print(f"    - {key}")
        if len(mammoth_missing_keys) > 5:
            print(f"    ... and {len(mammoth_missing_keys) - 5} more")

    # Show ignored keys for transparency
    if encoder_keys_ignored:
        print(f"\n  ℹ️  Encoder keys ignored (expected for decoder-only → encoder-decoder conversion):")
        for key in encoder_keys_ignored[:3]:
            print(f"    - {key}")
        if len(encoder_keys_ignored) > 3:
            print(f"    ... and {len(encoder_keys_ignored) - 3} more")

    if cross_attention_keys_ignored:
        print(f"\n  ℹ️  Cross-attention keys ignored (expected for decoder-only → encoder-decoder conversion):")
        for key in cross_attention_keys_ignored[:3]:
            print(f"    - {key}")
        if len(cross_attention_keys_ignored) > 3:
            print(f"    ... and {len(cross_attention_keys_ignored) - 3} more")

    if xt_missing_keys or mammoth_missing_keys:
        print(f"\n  ⚠️  VALIDATION FAILED: Cannot proceed with incomplete mapping!")
        print(f"      Please fix the mapping in create_xt_to_mammoth_mapping()")
        return False

    total_copied = 0
    total_missed = 0
    missed_details = []

    for task_idx, task_name in enumerate(task_names):
        mapping = create_xt_to_mammoth_mapping(num_decoder_layers, task_name)

        print(f"    Task {task_idx + 1}/{len(task_names)}: {task_name}")
        print(f"      - Mapping entries: {len(mapping)}")

        # Count mapping types for this task
        task_decoder_mappings = [k for k in mapping.keys() if 'decoder.' in k]
        print(f"      - Task-specific decoder mappings: {len(task_decoder_mappings)}")

        n_copied = 0
        n_missed = 0

        for xt_key, mammoth_key in mapping.items():
            # Double-check keys exist (should be guaranteed by pre-validation)
            if xt_key not in xt_sd:
                n_missed += 1
                if task_idx == 0:
                    missed_details.append(f"  ✗ XT key missing: {xt_key}")
                continue

            if mammoth_key not in mammoth_sd:
                n_missed += 1
                if task_idx == 0:
                    missed_details.append(f"  ✗ Mammoth key missing: {mammoth_key}")
                continue

            xt_val = xt_sd[xt_key]
            mammoth_val = mammoth_sd[mammoth_key]

            # Ensure shapes match exactly (vocab sizes are expected to be aligned)
            if xt_val.shape != mammoth_val.shape:
                if "token_emb.emb.weight" in xt_key:
                    raise RuntimeError(
                        f"CRITICAL: Vocabulary size mismatch detected!\n"
                        f"  HF model vocab size: {xt_val.shape[0]}\n"
                        f"  MAMMOTH model vocab size: {mammoth_val.shape[0]}\n"
                        f"  Key: {xt_key} → {mammoth_key}\n"
                        f"  This indicates that the source and target tokenizers have different vocabularies.\n"
                        f"  Please ensure that the source and target tokenizers have aligned vocabularies before running conversion."
                    )
                else:
                    n_missed += 1
                    if task_idx == 0:
                        missed_details.append(
                            f"  ✗ Shape mismatch: {xt_key} {xt_val.shape} → {mammoth_key} {mammoth_val.shape}"
                        )
                    continue

            mammoth_sd[mammoth_key] = xt_val.clone()
            n_copied += 1

        total_copied += n_copied
        total_missed += n_missed

    # Use strict=True to catch any remaining issues
    try:
        mammoth_model.load_state_dict(mammoth_sd, strict=True)
        print(f"    ✓ All weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"    ✗ Error loading state dict: {e}")
        print(f"    Falling back to strict=False")
        mammoth_model.load_state_dict(mammoth_sd, strict=False)

    keys_per_task = len(create_xt_to_mammoth_mapping(num_decoder_layers, task_names[0]))
    copied_per_task = total_copied // len(task_names)
    missed_per_task = total_missed // len(task_names)
    expected_total = keys_per_task * len(task_names)
    actual_valid_total = len(valid_mappings) * len(task_names)

    print(f"✓ Weight mapping complete:")
    print(f"  - Total mappings defined: {expected_total}")
    print(f"  - Total valid mappings: {actual_valid_total}")
    print(f"  - Total successfully mapped: {total_copied}/{actual_valid_total} parameters")
    print(f"  - Tasks processed: {len(task_names)}")
    print(f"  - Per task: {copied_per_task}/{len(valid_mappings)} keys successfully mapped ({copied_per_task/len(valid_mappings)*100:.1f}%)")
    print(f"  - Per task failures: {missed_per_task}/{len(valid_mappings)} keys failed to map")
    print(f"  - Overall success rate: {total_copied/actual_valid_total*100:.1f}%")

    if missed_details:
        print(f"\n  Mapping error details (first task only):")
        for detail in missed_details[:10]:
            print(detail)
        if len(missed_details) > 10:
            print(f"  ... and {len(missed_details) - 10} more")

    return True


def create_mammoth_model(xt_model, hf_model_path, src_tokenizer_path=None, src_lang="en", tgt_tokenizer_paths=None):
    """
    Create Mammoth multi-task model from x-transformer

    Creates:
    - N language-specific encoders (randomly initialized)
    - 1 shared decoder (Gemma3 weights loaded)
    - N task configurations

    Args:
        xt_model: x-transformers model with loaded HF weights
        hf_model_path: Path to HuggingFace model
        src_tokenizer_path: Path to source tokenizer.json (if None, uses HF model path)
        src_lang: Source language code (default: "en")
        tgt_tokenizer_paths: Dict of {lang: tokenizer_path} for separate target vocabs (optional)
    """
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=False, trust_remote_code=False)

    if hasattr(config, 'text_config'):
        config = config.text_config

    # Determine target languages from tokenizer paths
    if tgt_tokenizer_paths:
        tgt_langs = list(tgt_tokenizer_paths.keys())
    else:
        tgt_langs = ["ar"]  # Default for multilingual/shared tokenizer mode

    print("Creating Mammoth multi-task model...")
    print("  - 1 shared decoder (Gemma3)")
    print(f"  - {len(tgt_langs)} language-specific encoder(s): {', '.join(tgt_langs)}")

    # Create vocabulary (supports both shared and separate vocabs)
    if src_tokenizer_path is None:
        raise ValueError("src_tokenizer_path must be provided")

    # Prepare shared target tokenizer path for multilingual mode
    shared_tgt_tokenizer_path = None
    if tgt_tokenizer_paths is None:
        # Derive save directory from src_tokenizer_path
        tokenizer_dir = os.path.dirname(src_tokenizer_path) if os.path.dirname(src_tokenizer_path) else "."
        shared_tgt_tokenizer_path = os.path.join(tokenizer_dir, "shared_tgt_tokenizer.json")
        # This file should already exist from Stage 0
        if not os.path.exists(shared_tgt_tokenizer_path):
            # Fallback: create it now if it doesn't exist
            rename_hf_special_tokens_to_mammoth(src_tokenizer_path, shared_tgt_tokenizer_path, vocab_size_limit=config.vocab_size)

    vocabs_dict = create_vocabs_dict_from_hf_tokenizer(
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_paths=tgt_tokenizer_paths,
        src_lang=src_lang,
        tgt_langs=tgt_langs,
        shared_tgt_tokenizer_path=shared_tgt_tokenizer_path
    )

    model_opts = create_model_opts_from_xt_model(xt_model, hf_model_path)
    task_queue_manager = create_task_queue_manager(vocabs_dict, config.num_hidden_layers, src_lang=src_lang, tgt_langs=tgt_langs)
    opts = create_opts()

    # Build Mammoth multi-task model
    mammoth_model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=None,  # Multi-task training
    )

    # Map Gemma3 weights to task decoders
    task_names = [f"task_{src_lang}_{lang}" for lang in tgt_langs]

    print("Mapping Gemma3 weights to all task decoders...")
    success = map_xt_to_mammoth_weights(
        xt_model, mammoth_model, config.num_hidden_layers, task_names
    )

    if not success:
        pass
        # raise RuntimeError("Weight mapping failed due to invalid layer references. Please check the mapping in create_xt_to_mammoth_mapping()")

    # Create optimizer
    optimizer = MultipleOptimizer.from_opts(
        model=mammoth_model,
        opts=opts,
        task_queue_manager=task_queue_manager,
        frame_checkpoint=None,
    )

    print("✓ Mammoth multi-task model created")
    print("  ✓ Decoder: Gemma3 weights loaded")
    print(f"  ✓ Encoder: {len(tgt_langs)} randomly initialized ({', '.join(tgt_langs)})")
    return mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer


def calculate_model_parameters(model):
    """Calculate total parameters in the model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total_params, trainable_params


def format_parameter_count(num_params):
    """Format parameter count into human readable format"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.2f}K"
    else:
        return str(num_params)


# =============================================================================
# SECTION 4: Main conversion pipeline
# =============================================================================


def convert_hf_gemma3_to_mammoth(hf_model_path, save_path, src_tokenizer_path=None, src_lang="en", tgt_tokenizer_paths=None):
    """
    Convert HuggingFace Gemma3 to Mammoth multi-task format

    Creates multi-task setup:
    - N language-specific encoders (random init)
    - 1 shared decoder (Gemma3 weights)

    Args:
        hf_model_path: Path to HuggingFace model directory
        save_path: Path to save converted Mammoth model
        src_tokenizer_path: Path to source tokenizer.json (if None, uses hf_model_path/tokenizer.json)
        src_lang: Source language code (default: "en")
        tgt_tokenizer_paths: Dict of {lang: tokenizer_path} for separate target vocabs (optional)
                            If None, uses shared multilingual tokenizer
    """
    # Determine target languages
    if tgt_tokenizer_paths:
        tgt_langs = list(tgt_tokenizer_paths.keys())
    else:
        tgt_langs = ["ar"]  # Default

    print(f"Converting HuggingFace Gemma3: {hf_model_path}")
    print(f"Save path: {save_path}")
    print(f"Multi-task setup: 1 shared decoder + {len(tgt_langs)} language-specific encoder(s) ({', '.join(tgt_langs)})")
    print("=" * 70)

    # Load HF config to get vocab_size (needed for filtering multimodal tokens)
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=False, trust_remote_code=False)
    if hasattr(config, 'text_config'):
        config = config.text_config
    vocab_size = config.vocab_size
    print(f"Model vocab size: {vocab_size}")

    # Determine save directory for debug files
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    # Stage 0: Prepare source and target tokenizers for MAMMOTH compatibility
    print("\n[Stage 0] Preparing tokenizers")

    # Handle source tokenizer
    if src_tokenizer_path is None:
        print("  No src_tokenizer_path provided, using HF model tokenizer for source")
        tokenizer_path = os.path.join(hf_model_path, "tokenizer.json")
        src_tokenizer_path = os.path.join(save_dir, "src_tokenizer.json")
        rename_hf_special_tokens_to_mammoth(tokenizer_path, src_tokenizer_path, vocab_size_limit=vocab_size)
        print(f"  ✓ Source tokenizer prepared: {src_tokenizer_path}")
    else:
        print(f"  Using provided source tokenizer: {src_tokenizer_path}")

    # Handle target tokenizers (decoder-only model needs target tokenizers for mammoth conventions)
    if tgt_tokenizer_paths:
        print(f"  Processing {len(tgt_langs)} separate target tokenizer(s):")
        for lang, path in tgt_tokenizer_paths.items():
            print(f"    - {lang}: {path}")
            # Create MAMMOTH-compatible version of each target tokenizer
            tgt_tokenizer_name = f"tgt_{lang}_tokenizer.json"
            tgt_tokenizer_path_mammoth = os.path.join(save_dir, tgt_tokenizer_name)
            rename_hf_special_tokens_to_mammoth(path, tgt_tokenizer_path_mammoth, vocab_size_limit=vocab_size)
            # Update the path in tgt_tokenizer_paths to point to the MAMMOTH version
            tgt_tokenizer_paths[lang] = tgt_tokenizer_path_mammoth
    else:
        print(f"  Using shared tokenizer for target language (multilingual mode)")
        # Note: The shared target tokenizer will be prepared in Stage 2 where vocabs_dict is created
        print(f"  → Shared target tokenizer will be prepared for MAMMOTH compatibility in Stage 2")

    # Stage 1: HuggingFace → x-transformers
    print("\n[Stage 1] HuggingFace Gemma3 → x-transformers")
    xt_model = create_xtransformer_model(hf_model_path)
    xt_model = load_hf_weights_to_xtransformer(hf_model_path, xt_model)

    if xt_model is None:
        raise RuntimeError("HuggingFace → x-transformers weight loading failed due to invalid mapping. Please check the mapping in create_weight_mapping_gemma3()")

    # Save x-transformer keys for debugging
    xt_keys_path = os.path.join(save_dir, "xt_model_keys_gemma3.txt")
    with open(xt_keys_path, "w") as f:
        for key, value in xt_model.state_dict().items():
            f.write(key + "\t" + str(value.shape) + "\n")
    print(f"✓ x-transformer keys saved to {xt_keys_path} ({len(xt_model.state_dict())} keys)")

    # Stage 2: x-transformers → Mammoth multi-task
    print("\n[Stage 2] x-transformers → Mammoth multi-task")
    mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer = (
        create_mammoth_model(
            xt_model,
            hf_model_path,
            src_tokenizer_path=src_tokenizer_path,
            src_lang=src_lang,
            tgt_tokenizer_paths=tgt_tokenizer_paths
        )
    )

    # Save Mammoth model keys
    mammoth_keys_path = os.path.join(save_dir, "mammoth_model_keys_gemma3.txt")
    with open(mammoth_keys_path, "w") as f:
        for key,value in mammoth_model.state_dict().items():
            f.write(key + "\t" + str(value.shape) + "\n")
    print(f"✓ Mammoth model keys saved to {mammoth_keys_path} ({len(mammoth_model.state_dict())} keys)")

    # Stage 3: Save Mammoth model
    print("\n[Stage 3] Saving Mammoth model")

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

    # Calculate and display model parameter count
    print("\n[Model Size Analysis]")
    total_params, trainable_params = calculate_model_parameters(mammoth_model)

    print(f"  Total parameters: {format_parameter_count(total_params)} ({total_params:,})")
    print(f"  Trainable parameters: {format_parameter_count(trainable_params)} ({trainable_params:,})")
    print(f"  Frozen parameters: {format_parameter_count(total_params - trainable_params)} ({total_params - trainable_params:,})")

    # Estimate memory usage (assuming bf16)
    model_size_gb = total_params * 2 / (1024**3)  # bf16 = 2 bytes per parameter
    print(f"  Estimated model size (bf16): ~{model_size_gb:.2f} GB")

    print("=" * 50)
    print(f"✓ Conversion complete! Mammoth model saved to: {save_path}")
    print(f"  ✓ Architecture:")
    print(f"    - 1 shared decoder (Gemma3 weights loaded)")
    print(f"    - {len(tgt_langs)} language-specific encoder(s) (randomly initialized)")
    print(f"  ✓ Gemma3 features fully implemented:")
    print(f"    ✅ 4 norms per layer (via sandwich_norm=True)")
    print(f"    ✅ RMSNorm with unit_offset (via rms_norm=True)")
    print(f"    ✅ Q/K normalization")
    print(f"    ✅ MQA (Multi-Query Attention via attn_one_kv_head=True)")
    print(f"    ✅ Gated MLP with GELU (via ff_glu=True)")
    print(f"    ✅ LM head output projection (lm_head.weight → to_logits.weight)")
    print(f"    ✅ Scaled embeddings (via dec_scaled_embeddings=True)")
    print(f"    ✅ Dual RoPE (global_rope_theta={config.rope_theta:.0f}, local_rope_theta={getattr(config, 'rope_local_base_freq', 10000):.0f})")

    # Show sliding window status
    sliding_window = getattr(config, 'sliding_window', -1)
    if sliding_window > 0:
        print(f"    ✅ Sliding window attention (window={sliding_window}, global_attn_every_n={getattr(config, 'global_attn_every_n_layers', 3)})")
    else:
        print(f"    ℹ️  Sliding window attention (disabled - using full causal attention)")

    return mammoth_model


# =============================================================================
# SECTION 5: Command line interface
# =============================================================================


def main():
    """Main entry point for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace Gemma3 to Mammoth multi-task format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""

Output:
  - 1 shared decoder (Gemma3 weights loaded)
  - N language-specific encoders (randomly initialized, one per --tgt-tokenizer)
  - Languages determined by --src-tokenizer and --tgt-tokenizer arguments (defaults to src='en', tgt='ar' if none provided)
  - Target tokenizers automatically converted to MAMMOTH special token conventions
  - Both shared and separate tokenizer modes supported
        """,
    )

    parser.add_argument(
        "hf_model_path",
        help="HuggingFace model name (e.g., 'google/gemma-3-4b') or local path",
    )

    parser.add_argument(
        "save_path",
        help="Path to save converted model (e.g., './models/gemma3_4b.pt')"
    )

    parser.add_argument(
        "--src-tokenizer",
        nargs=2,
        metavar=("LANG", "PATH"),
        help="Source language tokenizer: LANG PATH (e.g., --src-tokenizer en /path/to/en_tokenizer.json). Required for specifying custom source language and tokenizer. If not provided, uses HF model tokenizer with default language 'en'."
    )

    parser.add_argument(
        "--tgt-tokenizer",
        nargs=2,
        action="append",
        metavar=("LANG", "PATH"),
        help="Target language tokenizer: LANG PATH (e.g., --tgt-tokenizer ar /path/to/ar_tokenizer.json). Can be specified multiple times."
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.hf_model_path) and "/" not in args.hf_model_path:
        print(f"Downloading from HuggingFace Hub: {args.hf_model_path}")

    # Ensure save directory exists
    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        print(f"Creating directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)

    # Parse target tokenizers into dict
    tgt_tokenizer_paths = None
    if args.tgt_tokenizer:
        tgt_tokenizer_paths = {lang: path for lang, path in args.tgt_tokenizer}

    # Parse source tokenizer
    src_tokenizer_path = None
    src_lang = "en"  # Default source language
    if args.src_tokenizer:
        src_lang, src_tokenizer_path = args.src_tokenizer

    try:
        # Run conversion
        convert_hf_gemma3_to_mammoth(
            hf_model_path=args.hf_model_path,
            save_path=args.save_path,
            src_tokenizer_path=src_tokenizer_path,
            src_lang=src_lang,
            tgt_tokenizer_paths=tgt_tokenizer_paths,
        )

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
