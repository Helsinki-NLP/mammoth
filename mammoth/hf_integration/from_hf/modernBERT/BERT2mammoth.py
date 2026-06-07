#!/usr/bin/env python3
import sys
import os

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

"""
HuggingFace ModernBERT to Mammoth Model Converter
Usage: python BERT2mammoth.py <hf_model_path> <save_path> [--src-lang en] [--tgt-lang es]

The minimum conversion solution will only map the weights instead of implementing the same techniques that ModernBert use.
The converted ModernBERT will be fully served by Mammoth (x-transformers).

ModernBERT-specific features:
- Pre-Layer Normalization (Pre-LN)
- GeGLU MLP (GELU-gated GLU with fused gate+value projection)
- Bias-free architecture
- Sliding window attention with configurable global attention pattern
- RoPE positional embeddings with per-layer theta (different bases for global/local attention)
- Fused Wqkv projection (mapped directly with fused_qkv=True for efficiency) (Not implemented yet)
- ModernBERT-style unpadding for efficient computation with variable-length sequences (Not implemented yet)
"""

import os
import torch
from collections import OrderedDict
from argparse import Namespace
from transformers import AutoConfig, AutoModel, AutoTokenizer
from mammoth.x_transformers import XTransformer
from mammoth.inputters.vocab import Vocab, HFTokenizerVocab, DEFAULT_SPECIALS
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
    Create XTransformer model from HF ModernBERT config with full feature support

    ModernBERT is encoder-only. For MAMMOTH (seq2seq framework):
    - Encoder: Load ModernBERT weights
    - Decoder: Standard 6-layer transformer (randomly initialized)

    Args:
        model_path: Path to HuggingFace model
    """
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)

    # Calculate ff_mult from ModernBERT's intermediate_size
    # ModernBERT uses intermediate_size=1152 (1.5x hidden_size)
    # With GLU, x-transformers will create weights of [768, 1152*2] = [768, 2304]
    intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
    ff_mult = intermediate_size / config.hidden_size

    # Extract sliding window attention parameters from HF config
    # If not present, default to -1 (disabled)
    enc_sliding_window = getattr(config, "local_attention", -1)
    enc_global_attn_every_n_layers = getattr(config, "global_attn_every_n_layers", 3)
    enc_global_rope_theta = getattr(config, "global_rope_theta", 160000.0)
    enc_local_rope_theta = getattr(config, "local_rope_theta", 10000.0)

    # Log configuration
    sliding_window_status = "enabled" if enc_sliding_window > 0 else "disabled"
    print(f"  - Encoder: ModernBERT (bias-free, Pre-LN, GLU, {config.num_hidden_layers} layers)")
    print(f"    • Sliding window: {sliding_window_status}")
    if enc_sliding_window > 0:
        print(f"      - Window size: {enc_sliding_window} tokens")
        print(f"      - Global attention every N layers: {enc_global_attn_every_n_layers}")
        print(f"      - Global RoPE theta: {enc_global_rope_theta}")
        print(f"      - Local RoPE theta: {enc_local_rope_theta}")
    print(f"  - Decoder: Standard transformer (6 layers, full causal attention, randomly initialized)")
    print(f"  - FF multiplier: {ff_mult} (intermediate_size={intermediate_size})")

    xt_model = XTransformer(
        dim=config.hidden_size,
        # Encoder: ModernBERT weights will be loaded here
        enc_num_tokens=config.vocab_size,
        enc_max_seq_len=config.max_position_embeddings,
        enc_rotary_pos_emb=True,  # ModernBERT uses RoPE
        enc_post_emb_norm=True,
        enc_depth=config.num_hidden_layers,
        enc_heads=config.num_attention_heads,
        enc_ff_mult=ff_mult,  # Match ModernBERT's intermediate size
        enc_emb_dropout=0.0,
        enc_ff_glu=True,  # ModernBERT uses GLU
        enc_ff_no_bias=True,  # ModernBERT is bias-free
        # Sliding window attention configuration
        enc_sliding_window=enc_sliding_window,
        enc_global_attn_every_n_layers=enc_global_attn_every_n_layers,
        enc_global_rope_theta=enc_global_rope_theta,
        enc_local_rope_theta=enc_local_rope_theta,

        # Decoder: Bias-free transformer (randomly initialized)
        # Uses standard full causal attention (no sliding window)
        dec_num_tokens=config.vocab_size,
        dec_max_seq_len=config.max_position_embeddings,
        dec_rotary_pos_emb=True,  # Match encoder (uses RoPE)
        dec_post_emb_norm=True,
        dec_depth=6,  # 6-layer decoder (per train.yaml)
        dec_heads=config.num_attention_heads,
        dec_emb_dropout=0.1,
        dec_ff_glu=True,  # Match encoder GLU
        dec_ff_no_bias=True,  # Match encoder (bias-free)
    )
    return xt_model


def create_weight_mapping_modernbert(num_layers):
    """
    Create mapping from HuggingFace ModernBERT to x-transformers
    """
    mapping = {}

    # Token embeddings
    mapping["embeddings.tok_embeddings.weight"] = "encoder.token_emb.emb.weight"
    # Note: ModernBERT uses RoPE, no separate positional embeddings

    # Embedding normalization (bias-free)
    mapping["embeddings.norm.weight"] = "encoder.post_emb_norm.gamma"

    # Encoder layers
    for i in range(num_layers):
        attn_idx = i * 2
        ff_idx = attn_idx + 1

        # Pre-LN before attention
        mapping[f"layers.{i}.attn_norm.weight"] = (
            f"encoder.attn_layers.layers.{attn_idx}.0.0.gamma"
        )

        # Fused Wqkv - will be split into separate Q, K, V during loading
        mapping[f"layers.{i}.attn.Wqkv.weight"] = (
            f"encoder.attn_layers.layers.{attn_idx}.1.to_q.weight",
            f"encoder.attn_layers.layers.{attn_idx}.1.to_k.weight",
            f"encoder.attn_layers.layers.{attn_idx}.1.to_v.weight"
        )

        # Attention output
        mapping[f"layers.{i}.attn.Wo.weight"] = (
            f"encoder.attn_layers.layers.{attn_idx}.1.to_out.weight"
        )

        # Pre-LN before MLP
        mapping[f"layers.{i}.mlp_norm.weight"] = (
            f"encoder.attn_layers.layers.{ff_idx}.0.0.gamma"
        )

        # MLP with GLU (fused gate+value projection)
        mapping[f"layers.{i}.mlp.Wi.weight"] = (
            f"encoder.attn_layers.layers.{ff_idx}.1.ff.0.proj.weight"
        )
        mapping[f"layers.{i}.mlp.Wo.weight"] = (
            f"encoder.attn_layers.layers.{ff_idx}.1.ff.2.weight"
        )

    # Final layer normalization
    mapping["final_norm.weight"] = "encoder.attn_layers.final_norm.gamma"

    return mapping



def load_hf_weights_to_xtransformer(hf_model_path, xt_model):
    """
    Load weights from HuggingFace ModernBERT to x-transformers model

    """
    print(f"Loading HF ModernBERT model from {hf_model_path}", flush=True)
    print(f"  [1/3] Loading config...", flush=True)
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True, trust_remote_code=False)
    print(f"  [2/3] Loading model weights (this may take a while)...", flush=True)
    hf_model = AutoModel.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        use_safetensors=True  # Prefer safetensors format (faster loading)
    )
    print(f"  [3/3] Extracting state dict...", flush=True)
    hf_state_dict = hf_model.state_dict()
    print(f"✓ Model loaded successfully", flush=True)

    mapping = create_weight_mapping_modernbert(config.num_hidden_layers)
    x_state_dict = OrderedDict()

    # Calculate attention dimensions
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads

    # Handle missing keys (special case: first layer pre-norm in ModernBERT)
    missing_keys = []
    identity_inits = []

    for hf_key, x_key in mapping.items():
        if hf_key not in hf_state_dict:
            if hf_key == "layers.0.attn_norm.weight":
                # Special case: ModernBERT layer 0 has no pre-norm, initialize as identity
                missing_keys.append(f"Missing: {hf_key} → {x_key}")
                identity_inits.append((x_key, "layers.0.attn_norm.weight"))
            continue

        # Check if x_key is a tuple (for split QKV mapping)
        if isinstance(x_key, tuple):
            # Split fused Wqkv into separate Q, K, V
            # ModernBERT has fused Wqkv of shape [hidden_size * 3, hidden_size]
            fused_qkv = hf_state_dict[hf_key]
            chunk_size = hidden_size

            # Split into Q, K, V chunks
            q_weight, k_weight, v_weight = torch.chunk(fused_qkv, 3, dim=0)

            # Assign to separate keys
            x_state_dict[x_key[0]] = q_weight  # to_q.weight
            x_state_dict[x_key[1]] = k_weight  # to_k.weight
            x_state_dict[x_key[2]] = v_weight  # to_v.weight
        else:
            # Direct weight mapping
            x_state_dict[x_key] = hf_state_dict[hf_key]

    # Initialize missing first-layer pre-norm as identity
    for x_key, hf_key in identity_inits:
        if x_key == "encoder.attn_layers.layers.0.0.0.gamma":
            # Initialize LayerNorm weight (gamma) as ones (identity)
            x_state_dict[x_key] = torch.ones(hidden_size, dtype=torch.float32)
            missing_keys.append(f"✓ Initialized identity: {hf_key} → {x_key}")

    # Special case: Check if first layer pre-norm needs identity initialization
    first_layer_prenorm_key = "encoder.attn_layers.layers.0.0.0.gamma"
    if first_layer_prenorm_key in x_state_dict and "layers.0.attn_norm.weight" not in hf_state_dict:
        # ModernBERT layer 0 has no pre-norm, but x-transformers expects one
        x_state_dict[first_layer_prenorm_key] = torch.ones(hidden_size, dtype=torch.float32)
        missing_keys.append(f"✓ Initialized identity: layers.0.attn_norm.weight → {first_layer_prenorm_key}")
        identity_inits.append((first_layer_prenorm_key, "layers.0.attn_norm.weight"))

    # Load weights (strict=False to allow missing decoder weights)
    xt_model.load_state_dict(x_state_dict, strict=False)
    print("✓ HF ModernBERT weights loaded to x-transformers model")
    print(f"  Loaded {len(x_state_dict)} parameters")

    # Report missing keys and identity initializations
    if missing_keys:
        print(f"  Handled {len(identity_inits)} missing keys:")
        for msg in missing_keys:
            print(f"    {msg}")

    return xt_model


# =============================================================================
# SECTION 2:  MAMMOTH utilities
# =============================================================================


def rename_hf_special_tokens_to_mammoth(tokenizer_path, output_path=None):
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


def create_vocabs_dict_from_hf_tokenizer(src_tokenizer_path, tgt_tokenizer_paths=None, src_lang="en", tgt_langs=None):
    """
    Create vocabs_dict from HuggingFace tokenizer(s) - supports both shared and separate vocabs

    Args:
        src_tokenizer_path: Path to source tokenizer.json file
        tgt_tokenizer_paths: Dict mapping language codes to tokenizer paths (e.g., {"ar": "path/to/ar_tokenizer.json"})
                            If None, uses src_tokenizer_path for all languages (multilingual setup)
        src_lang: Source language (default: "en")
        tgt_langs: List of target languages (default: ["ar"])
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
        print(f"✓ Using shared tokenizer for all languages (multilingual mode)")
        for tgt_lang in tgt_langs:
            tgt_vocab = HFTokenizerVocab(
                tokenizer_path=src_tokenizer_path,
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
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True, trust_remote_code=False)
    model_opts = Namespace()

    # Basic settings
    model_opts.model_dtype = "bf16"
    # Architecture
    model_opts.model_dim = config.hidden_size
    model_opts.enc_layers = [config.num_hidden_layers]
    model_opts.dec_layers = [6]  # 6-layer decoder (per train.yaml)

    # ModernBERT intermediate size calculation
    # Note: With GLU, x-transformers doubles the ff_mult internally (one for gate, one for value)
    # So if ModernBERT has intermediate_size=1152, we need ff_mult=1.5, not 3.0
    intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
    ff_mult = intermediate_size / config.hidden_size

    # Sliding window attention configuration (from HF config)
    # These are top-level model_opts attributes (defined in opts.py)
    model_opts.enc_sliding_window = getattr(config, "local_attention", -1)
    model_opts.enc_global_attn_every_n_layers = getattr(config, "global_attn_every_n_layers", 3)
    model_opts.enc_global_rope_theta = getattr(config, "global_rope_theta", 160000.0)
    model_opts.enc_local_rope_theta = getattr(config, "local_rope_theta", 10000.0)

    # Decoder uses standard full causal attention (no sliding window)
    model_opts.dec_sliding_window = -1  # Disabled
    model_opts.dec_global_attn_every_n_layers = 3  # Not used (sliding window disabled)
    model_opts.dec_global_rope_theta = 10000.0  # Standard RoPE base
    model_opts.dec_local_rope_theta = 10000.0  # Standard RoPE base

    model_opts.x_transformers_opts = {
        "heads": config.num_attention_heads,
        "attn_dropout": getattr(config, "attention_dropout", 0.0),
        "attn_flash": True,
        "ff_mult": ff_mult,
        "ff_dropout": getattr(config, "mlp_dropout", 0.0),
        "ff_glu": True,  # ModernBERT uses GLU
        "ff_no_bias": True,  # ModernBERT is bias-free
        # NOTE: pre_norm_has_final_norm is automatically managed by MAMMOTH and cannot be set explicitly
        "pre_norm": True,  # ModernBERT uses Pre-LN (AttentionLayers parameter)
        "post_emb_norm": True,  # TransformerWrapper parameter
        "rotary_pos_emb": True,  # AttentionLayers parameter
    }

    model_opts.max_length = getattr(config, "max_position_embeddings", 1024)
    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.dropout = [0.0]
    model_opts.attention_dropout = [getattr(config, "attention_dropout", 0.0)]

    print(f'model max_length: {model_opts.max_length}')

    # Log sliding window configuration
    if model_opts.enc_sliding_window > 0:
        print(f'encoder sliding window: {model_opts.enc_sliding_window} tokens (enabled)')
        print(f'  - global attention every {model_opts.enc_global_attn_every_n_layers} layers')
        print(f'  - global RoPE theta: {model_opts.enc_global_rope_theta}')
        print(f'  - local RoPE theta: {model_opts.enc_local_rope_theta}')
    else:
        print('encoder sliding window: disabled (full attention)')
    print('decoder sliding window: disabled (standard full causal attention)')

    return model_opts


class SimpleWorldContext(WorldContext):
    def __init__(self):
        super().__init__(context=1, n_nodes=1, gpus_per_node=1)
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


def create_task_queue_manager(vocabs_dict=None, num_layers=22, tgt_langs=None):
    """
    Create TaskQueueManager for multi-language multi-task setup

    Matches train.yaml configuration:
    - 1 shared encoder (ModernBERT with {num_layers} layers)
    - N language-specific decoders (6 layers each)

    Args:
        vocabs_dict: Dictionary of vocabularies
        num_layers: Number of encoder layers
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

    # Create 1 task - one per language pair
    for gpu_id, tgt_lang in enumerate(target_languages):
        node_id = gpu_id // 2  # 2 GPUs per node
        local_gpu = gpu_id % 2  # GPU within node (0 or 1)

        task_name = f"task_en_{tgt_lang}"
        opts.tasks[task_name] = {
            "src_tgt": f"en-{tgt_lang}",
            "weight": 1.0,
            "introduce_at_training_step": 0,
            "node_gpu": f"{node_id}:{local_gpu}",
            "enc_sharing_group": ["en"],  # All tasks share ModernBERT encoder
            "dec_sharing_group": [tgt_lang],  # Each task has language-specific decoder
        }

    opts.enc_layers = [num_layers]  # ModernBERT encoder layers
    opts.dec_layers = [6]  # Decoder layers (per train.yaml)
    opts.task_distribution_strategy = "weighted_sampling"
    opts.accum_count = [1]
    opts.seed = 42

    world_context = SimpleWorldContext()
    task_manager = TaskQueueManager.from_opts(opts, world_context)

    # Assign vocabularies to all tasks
    src_vocab = vocabs_dict.get(("src", "en"))

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



def create_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, corpus_id="modernbert_translation"):
    """
    Create mapping from xt_model keys to mammoth_model keys for ModernBERT
    """
    mapping = {}

    # Token embeddings (no positional embeddings - ModernBERT uses RoPE)
    mapping["encoder.token_emb.emb.weight"] = (
        f"encoder.{corpus_id}.token_emb.emb.weight"
    )

    # Post-embedding norm (bias-free - gamma parameter)
    mapping["encoder.post_emb_norm.gamma"] = (
        f"encoder.{corpus_id}.post_emb_norm.gamma"
    )


    # Encoder layers
    for layer_idx in range(num_encoder_layers):
        attn_idx = layer_idx * 2
        xt_attn_base = f"encoder.attn_layers.layers.{attn_idx}"
        mammoth_attn_base = f"encoder.{corpus_id}.attn_layers.attention_layers_stack.0.layers.{attn_idx}"

        # Pre-attention LayerNorm (bias-free - gamma parameter)
        # x-transformers: layers.0.0.0.gamma
        # Mammoth:        layers.0.0.0.gamma (same!)
        mapping[f"{xt_attn_base}.0.0.gamma"] = f"{mammoth_attn_base}.0.0.gamma"

        # Attention projections (bias-free, separate Q, K, V)
        mapping[f"{xt_attn_base}.1.to_q.weight"] = f"{mammoth_attn_base}.1.to_q.weight"
        mapping[f"{xt_attn_base}.1.to_k.weight"] = f"{mammoth_attn_base}.1.to_k.weight"
        mapping[f"{xt_attn_base}.1.to_v.weight"] = f"{mammoth_attn_base}.1.to_v.weight"
        mapping[f"{xt_attn_base}.1.to_out.weight"] = f"{mammoth_attn_base}.1.to_out.weight"

        # Feedforward layer
        ff_idx = layer_idx * 2 + 1
        xt_ff_base = f"encoder.attn_layers.layers.{ff_idx}"
        mammoth_ff_base = f"encoder.{corpus_id}.attn_layers.attention_layers_stack.0.layers.{ff_idx}"

        # Pre-FF LayerNorm (bias-free - gamma parameter)
        mapping[f"{xt_ff_base}.0.0.gamma"] = f"{mammoth_ff_base}.0.0.gamma"

        # GLU feedforward (fused gate+value projection, then output)
        # Both use the same structure:
        # ff.0.proj.weight is the fused [gate; value] projection
        # ff.2.weight is the output projection
        mapping[f"{xt_ff_base}.1.ff.0.proj.weight"] = f"{mammoth_ff_base}.1.ff.0.proj.weight"
        mapping[f"{xt_ff_base}.1.ff.2.weight"] = f"{mammoth_ff_base}.1.ff.2.weight"

    # Final encoder norm (bias-free - gamma parameter)
    mapping["encoder.attn_layers.final_norm.gamma"] = (
        f"encoder.{corpus_id}.attn_layers.attention_layers_stack.0.final_norm.gamma"
    )

    # Note: Decoder mappings are not needed since decoders are randomly initialized
    # The shared encoder is what we're transferring from ModernBERT

    return mapping


def map_xt_to_mammoth_weights(
    xt_model, mammoth_model, num_encoder_layers, num_decoder_layers, task_names
):
    """
    Map ModernBERT encoder weights to all task-specific encoders in Mammoth model

    Since enc_sharing_group: ["shared"] creates 1 separate encoder instances
    with shared parameters, we copy the same ModernBERT weights to all of them.
    During training, MAMMOTH's gradient synchronization keeps them aligned.

    Args:
        xt_model: x-transformers model with ModernBERT weights
        mammoth_model: Mammoth multi-task model
        num_encoder_layers: Number of encoder layers (22 for ModernBERT-base)
        num_decoder_layers: Number of decoder layers (6)
        task_names: List of task names (e.g., ["task_en_ar", "task_en_bn", ...])
    """
    xt_sd = xt_model.state_dict()
    mammoth_sd = mammoth_model.state_dict()

    total_copied = 0
    total_missed = 0
    missed_details = []

    # Copy encoder weights to ALL task-specific encoders
    for task_idx, task_name in enumerate(task_names):
        mapping = create_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, task_name)

        n_copied = 0
        n_missed = 0

        for xt_key, mammoth_key in mapping.items():
            if xt_key not in xt_sd:
                n_missed += 1
                if task_idx == 0:  # Only collect details for first task to avoid duplication
                    missed_details.append(f"  ✗ XT key missing: {xt_key}")
                continue

            if mammoth_key not in mammoth_sd:
                n_missed += 1
                if task_idx == 0:
                    missed_details.append(f"  ✗ Mammoth key missing: {mammoth_key}")
                continue

            xt_val = xt_sd[xt_key]
            mammoth_val = mammoth_sd[mammoth_key]

            # Handle vocabulary size mismatch (token embeddings)
            # Mammoth may add special tokens, increasing vocab size
            if xt_val.shape != mammoth_val.shape and "token_emb.emb.weight" in xt_key:
                if xt_val.shape[0] < mammoth_val.shape[0] and xt_val.shape[1] == mammoth_val.shape[1]:
                    # Pad with zeros for new special tokens
                    import torch
                    padding_size = mammoth_val.shape[0] - xt_val.shape[0]
                    padding = torch.zeros(padding_size, xt_val.shape[1], dtype=xt_val.dtype)
                    xt_val = torch.cat([xt_val, padding], dim=0)
                    if task_idx == 0:
                        missed_details.append(
                            f"  ⚠ Padded vocab: {xt_key} {xt_sd[xt_key].shape} → {xt_val.shape} (added {padding_size} tokens)"
                        )
                else:
                    n_missed += 1
                    if task_idx == 0:
                        missed_details.append(
                            f"  ✗ Shape mismatch: {xt_key} {xt_val.shape} → {mammoth_key} {mammoth_val.shape}"
                        )
                    continue
            elif xt_val.shape != mammoth_val.shape:
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

    mammoth_model.load_state_dict(mammoth_sd, strict=False)

    # Calculate per-task statistics
    keys_per_task = len(create_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, task_names[0]))
    copied_per_task = total_copied // len(task_names)
    missed_per_task = total_missed // len(task_names)

    print(f"✓ Weight mapping complete:")
    print(f"  - Copied ModernBERT encoder to {len(task_names)} task encoders")
    print(f"  - Per task: {copied_per_task}/{keys_per_task} keys successfully mapped")
    print(f"  - Per task: {missed_per_task}/{keys_per_task} keys failed to map")
    print(f"  - Total: {total_copied} keys copied, {total_missed} keys missed")

    if missed_details:
        print(f"\n  Missing key details (first task only):")
        for detail in missed_details[:10]:  # Show first 10
            print(detail)
        if len(missed_details) > 10:
            print(f"  ... and {len(missed_details) - 10} more missing keys")

    print(f"\n  Note: Decoders remain randomly initialized")



def create_mammoth_model(xt_model, hf_model_path, src_tokenizer_path=None, tgt_tokenizer_paths=None):
    """
    Create Mammoth multi-task model from x-transformer

    Creates:
    - 1 shared encoder (ModernBERT weights loaded)
    - N language-specific decoders (randomly initialized)
    - N task configurations

    Args:
        xt_model: x-transformers model with loaded HF weights
        hf_model_path: Path to HuggingFace model
        src_tokenizer_path: Path to source tokenizer.json (if None, uses HF model path)
        tgt_tokenizer_paths: Dict of {lang: tokenizer_path} for separate target vocabs (optional)
    """
    config = AutoConfig.from_pretrained(hf_model_path, local_files_only=True, trust_remote_code=False)

    # Determine target languages from tokenizer paths
    if tgt_tokenizer_paths:
        tgt_langs = list(tgt_tokenizer_paths.keys())
    else:
        tgt_langs = ["ar"]  # Default for multilingual/shared tokenizer mode

    print("Creating Mammoth multi-task model...")
    print("  - 1 shared encoder (ModernBERT)")
    print(f"  - {len(tgt_langs)} language-specific decoder(s): {', '.join(tgt_langs)}")

    # Create vocabulary (supports both shared and separate vocabs)
    if src_tokenizer_path is None:
        raise ValueError("src_tokenizer_path must be provided")

    vocabs_dict = create_vocabs_dict_from_hf_tokenizer(
        src_tokenizer_path=src_tokenizer_path,
        tgt_tokenizer_paths=tgt_tokenizer_paths,
        src_lang="en",
        tgt_langs=tgt_langs
    )

    model_opts = create_model_opts_from_xt_model(xt_model, hf_model_path)
    task_queue_manager = create_task_queue_manager(vocabs_dict, config.num_hidden_layers, tgt_langs=tgt_langs)
    opts = create_opts()

    # Build Mammoth multi-task model
    mammoth_model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=None,  # Multi-task training
    )

    # Map ModernBERT weights to task encoders
    task_names = [f"task_en_{lang}" for lang in tgt_langs]

    print("Mapping ModernBERT weights to all task encoders...")
    map_xt_to_mammoth_weights(
        xt_model, mammoth_model, config.num_hidden_layers, 6, task_names
    )

    # Create optimizer
    optimizer = MultipleOptimizer.from_opts(
        model=mammoth_model,
        opts=opts,
        task_queue_manager=task_queue_manager,
        frame_checkpoint=None,
    )

    print("✓ Mammoth multi-task model created")
    print("  ✓ Encoder: ModernBERT weights loaded")
    print(f"  ✓ Decoder: {len(tgt_langs)} randomly initialized ({', '.join(tgt_langs)})")
    return mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer


# =============================================================================
# SECTION 4: Main conversion pipeline
# =============================================================================


def convert_hf_to_mammoth(hf_model_path, save_path, src_tokenizer_path=None, tgt_tokenizer_paths=None):
    """
    Convert HuggingFace ModernBERT to Mammoth multi-task format

    Creates multi-task setup:
    - 1 shared encoder (ModernBERT weights)
    - N language-specific decoders (random init)

    Args:
        hf_model_path: Path to HuggingFace model directory
        save_path: Path to save converted Mammoth model
        src_tokenizer_path: Path to source tokenizer.json (if None, uses hf_model_path/tokenizer.json)
        tgt_tokenizer_paths: Dict of {lang: tokenizer_path} for separate target vocabs (optional)
                            If None, uses shared multilingual tokenizer
    """
    # Determine target languages
    if tgt_tokenizer_paths:
        tgt_langs = list(tgt_tokenizer_paths.keys())
    else:
        tgt_langs = ["ar"]  # Default

    print(f"Converting HuggingFace ModernBERT: {hf_model_path}")
    print(f"Save path: {save_path}")
    print(f"Multi-task setup: 1 shared encoder + {len(tgt_langs)} language-specific decoder(s) ({', '.join(tgt_langs)})")
    print("=" * 70)

    # Determine save directory for debug files
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    # Stage 0: Prepare tokenizer
    print("\n[Stage 0] Preparing tokenizer")

    # If no src_tokenizer_path provided, rename HF tokenizer to MAMMOTH conventions
    if src_tokenizer_path is None:
        print("  No src_tokenizer_path provided, using HF model tokenizer")
        tokenizer_path = os.path.join(hf_model_path, "tokenizer.json")
        src_tokenizer_path = os.path.join(save_dir, "tokenizer.json")
        rename_hf_special_tokens_to_mammoth(tokenizer_path, src_tokenizer_path)
    else:
        print(f"  Using provided source tokenizer: {src_tokenizer_path}")

    # Check if separate target tokenizers are provided
    if tgt_tokenizer_paths:
        print(f"  Using separate target tokenizers for {len(tgt_langs)} language(s):")
        for lang, path in tgt_tokenizer_paths.items():
            print(f"    - {lang}: {path}")
    else:
        print(f"  Using shared tokenizer for all languages (multilingual mode)")

    # Stage 1: HuggingFace → x-transformers
    print("\n[Stage 1] HuggingFace ModernBERT → x-transformers")
    xt_model = create_xtransformer_model(hf_model_path)
    xt_model = load_hf_weights_to_xtransformer(hf_model_path, xt_model)

    # Save x-transformer keys for debugging
    xt_keys_path = os.path.join(save_dir, "xt_model_keys_modernbert.txt")
    xt_state_dict = xt_model.state_dict()
    with open(xt_keys_path, "w") as f:
        for key in xt_state_dict.keys():
            f.write(key + "\t" + str(xt_state_dict[key].shape) + "\n")
    print(f"✓ x-transformer keys saved to {xt_keys_path}")

    # Stage 2: x-transformers → Mammoth multi-task
    print("\n[Stage 2] x-transformers → Mammoth (1-task)")
    mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer = (
        create_mammoth_model(
            xt_model,
            hf_model_path,
            src_tokenizer_path=src_tokenizer_path,
            tgt_tokenizer_paths=tgt_tokenizer_paths
        )
    )

    # Save Mammoth model keys
    mammoth_keys_path = os.path.join(save_dir, "mammoth_model_keys_modernbert.txt")
    mammoth_state_dict = mammoth_model.state_dict()
    with open(mammoth_keys_path, "w") as f:
        for key in mammoth_state_dict.keys():
            f.write(key + "\t" + str(mammoth_state_dict[key].shape) + "\n")
    print(f"✓ Mammoth model keys saved to {mammoth_keys_path}")


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

    print("=" * 50)
    print(f"✓ Conversion complete! Mammoth model saved to: {save_path}")
    return mammoth_model


# =============================================================================
# SECTION 5: Command line interface
# =============================================================================


def main():
    """Main entry point for command line usage"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace ModernBERT to Mammoth multi-task format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multilingual setup (shared tokenizer)
  python hfModernBERT2mammoth.py answerdotai/ModernBERT-base ./models/modernbert_base.pt

  # Separate source/target tokenizers (for translation)
  python hfModernBERT2mammoth.py answerdotai/ModernBERT-base ./models/modernbert_base.pt \\
    --src-tokenizer /path/to/en_tokenizer.json \\
    --tgt-tokenizer ar /path/to/ar_tokenizer.json

Output:
  - 1 shared encoder (ModernBERT weights loaded)
  - N language-specific decoders (randomly initialized, one per --tgt-tokenizer)
  - Languages determined by --tgt-tokenizer arguments (defaults to 'ar' if none provided)
        """,
    )

    parser.add_argument(
        "hf_model_path",
        help="HuggingFace model name (e.g., 'answerdotai/ModernBERT-base') or local path",
    )

    parser.add_argument(
        "save_path",
        help="Path to save converted model (e.g., './models/modernbert_base.pt')"
    )

    parser.add_argument(
        "--src-tokenizer",
        help="Path to source tokenizer.json (if not provided, uses HF model tokenizer)"
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

    try:
        # Run conversion
        convert_hf_to_mammoth(
            hf_model_path=args.hf_model_path,
            save_path=args.save_path,
            src_tokenizer_path=args.src_tokenizer,
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