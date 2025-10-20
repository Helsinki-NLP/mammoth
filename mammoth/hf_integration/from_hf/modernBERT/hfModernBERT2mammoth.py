#!/usr/bin/env python3
"""
HuggingFace ModernBERT to Mammoth Model Converter
Usage: python hfModernBERT2mammoth.py <hf_model_path> <save_path> [--src-lang en] [--tgt-lang es]

ModernBERT-specific features:
- Fused Wqkv projection (needs splitting into separate Q, K, V)
- Pre-Layer Normalization (Pre-LN)
- GeGLU MLP (GELU-gated GLU)
- Bias-free architecture
- RoPE positional embeddings; different RoPE bases and dimensions for local and global attention
- ModernBERT-style unpadding for efficient computation with variable-length sequences
"""

import os
import torch
from collections import OrderedDict
from argparse import Namespace
from transformers import AutoConfig, AutoModel, AutoTokenizer
from mammoth.x_transformers import XTransformer
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.distributed.tasks import TaskQueueManager
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.model_builder import build_model
from mammoth.utils.optimizers import MultipleOptimizer
from mammoth.utils.model_saver import build_model_saver
from mammoth.models.architecture_config import get_model_architecture_config


# =============================================================================
# SECTION 1: HuggingFace to x-transformers conversion
# =============================================================================


def create_xtransformer_model(model_path, model_type='modernbert'):
    """
    Create XTransformer model from HF ModernBERT config

    ModernBERT is encoder-only. For MAMMOTH (seq2seq framework):
    - Encoder: Load ModernBERT weights (bias-free, Pre-LN, GLU, local/global attention)
    - Decoder: Standard 6-layer transformer (randomly initialized)

    Args:
        model_path: Path to HuggingFace model
        model_type: Architecture type (default: 'modernbert')
    """
    config = AutoConfig.from_pretrained(model_path)

    # Calculate ff_mult from ModernBERT's intermediate_size
    # ModernBERT uses intermediate_size=1152 (1.5x hidden_size)
    # With GLU, x-transformers will create weights of [768, 1152*2] = [768, 2304]
    intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
    ff_mult = intermediate_size / config.hidden_size

    # Extract sliding window attention configuration
    sliding_window = getattr(config, "local_attention", -1)  # -1 means no sliding window
    global_attn_every_n_layers = getattr(config, "global_attn_every_n_layers", -1)

    # Extract RoPE theta configuration for global/local attention layers
    global_rope_theta = getattr(config, "global_rope_theta", 160000.0)
    local_rope_theta = getattr(config, "local_rope_theta", 10000.0)

    print(f"Using architecture: {model_type}")
    print(f"  - Encoder: ModernBERT (bias-free, Pre-LN, GLU, {config.num_hidden_layers} layers)")
    print(f"  - Decoder: Standard transformer (6 layers, randomly initialized)")
    print(f"  - FF multiplier: {ff_mult} (intermediate_size={intermediate_size})")
    print(f"  - RoPE theta: global={global_rope_theta}, local={local_rope_theta}")

    # Print attention configuration
    if sliding_window > 0:
        print(f"  - Sliding window: {sliding_window} (±{sliding_window//2} tokens)")
        if global_attn_every_n_layers > 0:
            global_layers = [i for i in range(config.num_hidden_layers) if i % global_attn_every_n_layers == 0]
            local_layers = [i for i in range(config.num_hidden_layers) if i % global_attn_every_n_layers != 0]
            print(f"  - Global attention every {global_attn_every_n_layers} layers")
            print(f"    Global layers: {global_layers}")
            print(f"    Local layers:  {local_layers}")
        else:
            print(f"  - All layers use sliding window attention")
    else:
        print(f"  - Full attention (no sliding window)")


    xt_model = XTransformer(
        dim=config.hidden_size,
        # Encoder: ModernBERT weights will be loaded here
        enc_num_tokens=config.vocab_size,
        enc_max_seq_len=config.max_position_embeddings,
        enc_rotary_pos_emb=True,  # ModernBERT uses RoPE
        enc_global_rope_theta=global_rope_theta,  # RoPE theta for global attention layers
        enc_local_rope_theta=local_rope_theta,    # RoPE theta for local attention layers
        enc_post_emb_norm=True,
        enc_depth=config.num_hidden_layers,
        enc_heads=config.num_attention_heads,
        enc_ff_mult=ff_mult,  # Match ModernBERT's intermediate size
        enc_emb_dropout=0.0,
        enc_attn_flash=True,
        enc_ff_glu=True,  # ModernBERT uses GLU
        enc_ff_no_bias=True,  # ModernBERT is bias-free
        enc_sliding_window=sliding_window,  # NEW: Sliding window support
        enc_global_attn_every_n_layers=global_attn_every_n_layers,  # NEW: Global attention pattern


        # Decoder: Bias-free transformer (randomly initialized)
        # Will be replicated 16x for multi-task training
        dec_num_tokens=config.vocab_size,
        dec_max_seq_len=config.max_position_embeddings,
        dec_rotary_pos_emb=True,  # Match encoder (uses RoPE)
        # Decoder uses standard causal attention (no global/local pattern)
        # Set both thetas to same value - standard RoPE base of 10000
        dec_global_rope_theta=10000.0,  # Standard RoPE for decoder
        dec_local_rope_theta=10000.0,   # Same value (no pattern)
        dec_post_emb_norm=True,
        dec_depth=6,  # 6-layer decoder (per train.yaml)
        dec_heads=config.num_attention_heads,
        dec_emb_dropout=0.1,
        dec_attn_flash=True,
        dec_ff_glu=True,  # Match encoder GLU
        dec_ff_no_bias=True,  # Match encoder (bias-free)
    )
    return xt_model


def create_weight_mapping_modernbert(num_layers):
    """
    Create mapping from HuggingFace ModernBERT to x-transformers

    ModernBERT is bias-free, so we only map .weight parameters.
    Key challenge: Split fused Wqkv into separate to_q, to_k, to_v
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

        # Fused Wqkv - special handling (will split during loading)
        mapping[f"layers.{i}.attn.Wqkv.weight"] = "FUSED_QKV"

        # Attention output
        mapping[f"layers.{i}.attn.Wo.weight"] = (
            f"encoder.attn_layers.layers.{attn_idx}.1.to_out.weight"
        )

        # Pre-LN before MLP
        mapping[f"layers.{i}.mlp_norm.weight"] = (
            f"encoder.attn_layers.layers.{ff_idx}.0.0.gamma"
        )

        # MLP with GLU
        mapping[f"layers.{i}.mlp.Wi.weight"] = (
            f"encoder.attn_layers.layers.{ff_idx}.1.ff.0.weight"
        )
        mapping[f"layers.{i}.mlp.Wo.weight"] = (
            f"encoder.attn_layers.layers.{ff_idx}.1.ff.3.weight"
        )

    # Final layer normalization
    mapping["final_norm.weight"] = "encoder.attn_layers.final_norm.gamma"

    return mapping


def split_fused_qkv(fused_weight, num_heads, head_dim):
    """
    Split ModernBERT's fused Wqkv into separate Q, K, V projections

    Args:
        fused_weight: Shape [3 * hidden_size, hidden_size]
        num_heads: Number of attention heads
        head_dim: Dimension per head

    Returns:
        q_weight, k_weight, v_weight (no bias - ModernBERT is bias-free)
    """
    hidden_size = num_heads * head_dim

    # Fused weight is stacked as [Q; K; V] vertically
    q_weight = fused_weight[:hidden_size, :]
    k_weight = fused_weight[hidden_size:2*hidden_size, :]
    v_weight = fused_weight[2*hidden_size:, :]

    return q_weight, k_weight, v_weight


def load_hf_weights_to_xtransformer(hf_model_path, xt_model):
    """Load weights from HuggingFace ModernBERT to x-transformers model"""
    print(f"Loading HF ModernBERT model from {hf_model_path}")
    config = AutoConfig.from_pretrained(hf_model_path)
    hf_model = AutoModel.from_pretrained(hf_model_path)
    hf_state_dict = hf_model.state_dict()

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

        if x_key == "FUSED_QKV":
            # Special handling for fused Wqkv - split into Q, K, V
            layer_idx = int(hf_key.split('.')[1])
            attn_idx = layer_idx * 2

            fused_weight = hf_state_dict[hf_key]
            q_w, k_w, v_w = split_fused_qkv(fused_weight, num_heads, head_dim)

            # Map to x-transformers (bias-free)
            x_state_dict[f"encoder.attn_layers.layers.{attn_idx}.1.to_q.weight"] = q_w
            x_state_dict[f"encoder.attn_layers.layers.{attn_idx}.1.to_k.weight"] = k_w
            x_state_dict[f"encoder.attn_layers.layers.{attn_idx}.1.to_v.weight"] = v_w
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
# SECTION 2: Reuse MAMMOTH utilities from BART converter
# =============================================================================

def create_vocabs_dict_from_hf_tokenizer(hf_model_path, src_lang="en", tgt_langs=None, save_dir="."):
    """
    Create vocabs_dict from HuggingFace tokenizer for multilingual setup

    Args:
        hf_model_path: Path to HuggingFace model
        src_lang: Source language (default: "en")
        tgt_langs: List of target languages (default: 16 languages for multi-task)
        save_dir: Directory to save vocabulary files (default: current directory)
    """
    if tgt_langs is None:
        # Default 16 target languages from train.yaml
        tgt_langs = [
            "hi", "bn", "gu", "kn",  # Indo-Aryan
            "be", "bs", "hr", "eo",  # Slavic + Esperanto
            "ar", "he", "fa", "sw",  # Semitic + Iranian + Bantu
            "eu", "fi", "gl", "is",  # Mixed
        ]

    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    vocab_items = [
        tok for tok, _ in sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    ]

    specials = list(DEFAULT_SPECIALS)

    # Create source vocabulary
    src_vocab = Vocab(
        path=None,
        items=vocab_items,
        tag=f"src_{src_lang}",
        size=len(tokenizer),
        specials=specials,
    )

    # Create vocabs_dict with source and all target languages
    vocabs_dict = {("src", src_lang): src_vocab}

    # Add all target language vocabularies (sharing the same multilingual vocab)
    for tgt_lang in tgt_langs:
        tgt_vocab = Vocab(
            path=None,
            items=vocab_items,
            tag=f"tgt_{tgt_lang}",
            size=len(tokenizer),
            specials=specials,
        )
        vocabs_dict[("tgt", tgt_lang)] = tgt_vocab

    # Print vocabularies to txt files in save directory
    src_vocab_file = os.path.join(save_dir, f"src_vocab_{src_lang}.txt")
    with open(src_vocab_file, "w", encoding="utf-8") as f:
        for token in vocab_items:
            f.write(token + "\n")

    print(f"✓ Created vocabularies: {len(src_vocab)} tokens")
    print(f"✓ Saved src vocab to: {src_vocab_file}")
    print(f"✓ Target languages: {', '.join(tgt_langs)}")
    return vocabs_dict


def create_model_opts_from_xt_model(xt_model, hf_model_path, model_type='modernbert'):
    """Create model_opts for Mammoth from x-transformer model"""
    config = AutoConfig.from_pretrained(hf_model_path)
    model_opts = Namespace()

    # Basic settings
    model_opts.model_type = model_type
    model_opts.model_dtype = "fp32" # ModernBERT is using fp32
    model_opts.pos_ffn_activation_fn = "gelu"

    # Architecture
    model_opts.model_dim = config.hidden_size
    model_opts.enc_layers = [config.num_hidden_layers]
    model_opts.dec_layers = [config.num_hidden_layers]

    # ModernBERT intermediate size calculation
    # Note: With GLU, x-transformers doubles the ff_mult internally (one for gate, one for value)
    # So if ModernBERT has intermediate_size=1152, we need ff_mult=1.5, not 3.0
    intermediate_size = getattr(config, "intermediate_size", config.hidden_size * 4)
    ff_mult = intermediate_size / config.hidden_size

    # Extract sliding window attention configuration
    sliding_window = getattr(config, "local_attention", -1)  # -1 means no sliding window
    global_attn_every_n_layers = getattr(config, "global_attn_every_n_layers", -1)

    # Store sliding window configuration for later use
    model_opts.sliding_window = sliding_window
    model_opts.global_attn_every_n_layers = global_attn_every_n_layers

    # Extract RoPE theta configuration for global/local attention layers
    global_rope_theta = getattr(config, "global_rope_theta", 160000.0)
    local_rope_theta = getattr(config, "local_rope_theta", 10000.0)

    model_opts.x_transformers_opts = {
        "heads": config.num_attention_heads,
        "attn_dropout": getattr(config, "attention_dropout", 0.0),
        "attn_flash": True,
        "attn_use_unpadding": True,  # ModernBERT-style unpadding for efficient computation with variable-length sequences
        "ff_mult": ff_mult,
        "ff_dropout": getattr(config, "mlp_dropout", 0.0),
        "ff_glu": True,  # ModernBERT uses GLU
        "ff_no_bias": True,  # ModernBERT is bias-free
        "pre_norm": True,  # ModernBERT uses Pre-LN (AttentionLayers parameter)
        "post_emb_norm": True,  # TransformerWrapper parameter
        "rotary_pos_emb": True,  # AttentionLayers parameter
        "global_rope_theta": global_rope_theta,  # RoPE theta for global attention layers
        "local_rope_theta": local_rope_theta,    # RoPE theta for local attention layers
        "sliding_window": sliding_window,  # NEW: Sliding window support
        "global_attn_every_n_layers": global_attn_every_n_layers,  # NEW: Global attention pattern
    }

    model_opts.max_length = getattr(config, "max_position_embeddings", 1024)
    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.normformer = False
    model_opts.self_attn_type = "scaled-dot"
    model_opts.dropout = [0.0]
    model_opts.attention_dropout = [getattr(config, "attention_dropout", 0.0)]

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


def create_task_queue_manager(vocabs_dict=None, num_layers=22):
    """
    Create TaskQueueManager for 16-language multi-task setup

    Matches train.yaml configuration:
    - 1 shared encoder (ModernBERT with {num_layers} layers)
    - 16 language-specific decoders (6 layers each)
    - 16 GPUs across 8 nodes (2 GPUs per node)
    """
    # 16 target languages from train.yaml
    target_languages = [
        "hi", "bn", "gu", "kn",  # Indo-Aryan (Node 0-1, GPUs 0-3)
        "be", "bs", "hr", "eo",  # Slavic + Esperanto (Node 2-3, GPUs 4-7)
        "ar", "he", "fa", "sw",  # Semitic + Iranian + Bantu (Node 4-5, GPUs 8-11)
        "eu", "fi", "gl", "is",  # Mixed (Node 6-7, GPUs 12-15)
    ]

    opts = Namespace()
    opts.tasks = {}

    # Create 16 tasks - one per language pair
    for gpu_id, tgt_lang in enumerate(target_languages):
        node_id = gpu_id // 2  # 2 GPUs per node
        local_gpu = gpu_id % 2  # GPU within node (0 or 1)

        task_name = f"task_en_{tgt_lang}"
        opts.tasks[task_name] = {
            "src_tgt": f"en-{tgt_lang}",
            "weight": 1.0,
            "introduce_at_training_step": 0,
            "node_gpu": f"{node_id}:{local_gpu}",
            "enc_sharing_group": ["shared"],  # All tasks share ModernBERT encoder
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
        # All languages share the same multilingual vocabulary
        task.tgt_vocab = src_vocab  # ModernBERT uses single multilingual vocab

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
    opts.model_dtype = "fp32"
    opts.optim = "adafactor"
    opts.learning_rate = 0.001
    opts.adam_beta1 = 0.9  # Required even for adafactor (fallback)
    opts.adam_beta2 = 0.999  # Required even for adafactor (fallback)
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
# SECTION 3: x-transformers to Mammoth conversion (reuse from BART)
# =============================================================================

# Note: The x-transformers → MAMMOTH mapping is identical to BART
# since both use the same x-transformers backend
# Reusing the functions from hfBART2mammoth.py


def create_xt_to_mammoth_mapping(num_encoder_layers, num_decoder_layers, corpus_id="modernbert_translation"):
    """
    Create mapping from xt_model keys to mammoth_model keys for ModernBERT

    Both x-transformers and Mammoth use bias-free architecture for ModernBERT:
    - Uses RoPE (no positional embeddings)
    - Bias-free LayerNorm (uses 'gamma' parameter)
    - GLU feedforward with fused gate+value projection (ff.0.proj.weight, ff.2.weight)
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

        # Attention projections (bias-free)
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

    Since enc_sharing_group: ["shared"] creates 16 separate encoder instances
    with shared parameters, we copy the same ModernBERT weights to all of them.
    During training, MAMMOTH's gradient synchronization keeps them aligned.

    Args:
        xt_model: x-transformers model with ModernBERT weights
        mammoth_model: Mammoth multi-task model
        num_encoder_layers: Number of encoder layers (22 for ModernBERT-base)
        num_decoder_layers: Number of decoder layers (6)
        task_names: List of task names (e.g., ["task_en_hi", "task_en_bn", ...])
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


def verify_attention_patterns(mammoth_model, hf_config):
    """
    Verify that each layer has the correct attention type (global vs local)

    Args:
        mammoth_model: MAMMOTH multi-task model
        hf_config: HuggingFace ModernBERT config
    """
    print("\nVerifying attention pattern configuration...")

    num_layers = hf_config.num_hidden_layers
    sliding_window = getattr(hf_config, "local_attention", -1)
    global_attn_every_n_layers = getattr(hf_config, "global_attn_every_n_layers", -1)

    if sliding_window <= 0:
        print(f"  ✓ Full attention (no sliding window configured)")
        return

    print(f"  Expected pattern:")
    print(f"    - Sliding window: {sliding_window} (±{sliding_window//2} tokens)")
    if global_attn_every_n_layers > 0:
        print(f"    - Global attention every {global_attn_every_n_layers} layers")
        expected_global_layers = [i for i in range(num_layers) if i % global_attn_every_n_layers == 0]
        expected_local_layers = [i for i in range(num_layers) if i % global_attn_every_n_layers != 0]
        print(f"    - Global layers: {expected_global_layers}")
        print(f"    - Local layers:  {expected_local_layers}")
    else:
        print(f"    - All layers use sliding window attention")
        expected_global_layers = []
        expected_local_layers = list(range(num_layers))

    # Check the first task's encoder (all tasks share the same encoder)
    first_task_name = f"task_en_hi"  # First task from target languages

    try:
        # Navigate to the encoder attention layers using the correct MAMMOTH structure
        # MAMMOTH uses StackXcoder which is a ModuleDict containing task-specific TransformerWrappers
        transformer_wrapper = mammoth_model.encoder[first_task_name]
        attention_layers = transformer_wrapper.attn_layers.attention_layers_stack[0].layers

        print(f"\n  Actual attention configuration:")

        correct_count = 0
        total_count = 0

        for layer_idx in range(num_layers):
            # Every other layer is attention (even indices: 0, 2, 4, ...)
            attn_idx = layer_idx * 2

            if attn_idx >= len(attention_layers):
                print(f"    Layer {layer_idx:2d}: ⚠️  Index {attn_idx} out of range")
                continue

            attention_module = attention_layers[attn_idx][1]  # [0] is LayerNorm, [1] is Attention

            # Get the window size from the attention module
            window_size = getattr(attention_module.attend, 'window_size', None)

            # Determine if this should be global or local
            expected_global = (global_attn_every_n_layers > 0 and
                             layer_idx % global_attn_every_n_layers == 0)

            # Check the actual attention type
            if window_size == (-1, -1) or window_size is None:
                actual_type = "Global"
                is_correct = expected_global
            else:
                actual_type = f"Local({window_size})"
                is_correct = not expected_global

            if is_correct:
                status = "✅"
                correct_count += 1
            else:
                status = "❌"

            total_count += 1
            expected_type = "Global" if expected_global else f"Local({(sliding_window//2, sliding_window//2)})"

            print(f"    Layer {layer_idx:2d}: {actual_type:<15} {status} Expected: {expected_type}")

        # Summary
        if correct_count == total_count:
            print(f"\n  ✓ All {total_count} layers have correct attention configuration!")
        else:
            print(f"\n  ⚠️  {correct_count}/{total_count} layers have correct attention configuration")

    except Exception as e:
        print(f"  ❌ Error verifying attention patterns: {e}")
        print(f"     This might be due to model structure differences")
        # Let's print some debug information about the model structure
        try:
            print(f"  Encoder type: {type(mammoth_model.encoder)}")
            if hasattr(mammoth_model.encoder, 'keys'):
                print(f"  Available tasks: {list(mammoth_model.encoder.keys())}")
        except:
            pass


def create_mammoth_model(xt_model, hf_model_path, model_type='modernbert', save_dir="."):
    """
    Create Mammoth multi-task model from x-transformer

    Creates:
    - 1 shared encoder (ModernBERT weights loaded)
    - 16 language-specific decoders (randomly initialized)
    - 16 task configurations matching train.yaml
    """
    print("Creating Mammoth multi-task model...")
    print("  - 1 shared encoder (ModernBERT)")
    print("  - 16 language-specific decoders")

    config = AutoConfig.from_pretrained(hf_model_path)

    # Create multilingual vocabulary (shared across all tasks)
    vocabs_dict = create_vocabs_dict_from_hf_tokenizer(hf_model_path, src_lang="en", save_dir=save_dir)

    model_opts = create_model_opts_from_xt_model(xt_model, hf_model_path, model_type)
    task_queue_manager = create_task_queue_manager(vocabs_dict, config.num_hidden_layers)
    opts = create_opts()

    # Build Mammoth multi-task model (no single_task - all 16 tasks)
    mammoth_model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=None,  # Multi-task training
    )

    # Map ModernBERT weights to all 16 task encoders
    # (They share parameters during training via gradient synchronization)
    # Decoders remain randomly initialized (one per language)
    target_languages = [
        "hi", "bn", "gu", "kn",  # Indo-Aryan
        "be", "bs", "hr", "eo",  # Slavic + Esperanto
        "ar", "he", "fa", "sw",  # Semitic + Iranian + Bantu
        "eu", "fi", "gl", "is",  # Mixed
    ]
    task_names = [f"task_en_{lang}" for lang in target_languages]

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
    print("  ✓ Decoders: 16 randomly initialized (one per language)")
    return mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer


# =============================================================================
# SECTION 4: Main conversion pipeline
# =============================================================================


def convert_hf_to_mammoth(hf_model_path, save_path, model_type='modernbert'):
    """
    Convert HuggingFace ModernBERT to Mammoth multi-task format

    Creates 16-task setup matching train.yaml:
    - 1 shared encoder (ModernBERT weights)
    - 16 language-specific decoders (random init)
    """
    print(f"Converting HuggingFace ModernBERT: {hf_model_path}")
    print(f"Save path: {save_path}")
    print(f"Multi-task setup: 1 shared encoder + 16 language-specific decoders")
    print("=" * 70)

    # Determine save directory for debug files
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    # Stage 1: HuggingFace → x-transformers
    print("\n[Stage 1] HuggingFace ModernBERT → x-transformers")
    xt_model = create_xtransformer_model(hf_model_path, model_type)
    xt_model = load_hf_weights_to_xtransformer(hf_model_path, xt_model)

    # Save x-transformer keys for debugging
    xt_keys_path = os.path.join(save_dir, "xt_model_keys_modernbert.txt")
    with open(xt_keys_path, "w") as f:
        for key in xt_model.state_dict().keys():
            f.write(key + "\n")
    print(f"✓ x-transformer keys saved to {xt_keys_path}")

    # Stage 2: x-transformers → Mammoth multi-task
    print("\n[Stage 2] x-transformers → Mammoth (16-task)")
    mammoth_model, model_opts, vocabs_dict, task_queue_manager, optimizer = (
        create_mammoth_model(xt_model, hf_model_path, model_type, save_dir)
    )

    # Save Mammoth model keys
    mammoth_keys_path = os.path.join(save_dir, "mammoth_model_keys_modernbert.txt")
    with open(mammoth_keys_path, "w") as f:
        for key in mammoth_model.state_dict().keys():
            f.write(key + "\n")
    print(f"✓ Mammoth model keys saved to {mammoth_keys_path}")

    # Stage 2.5: Verify attention patterns
    print("\n[Stage 2.5] Verifying attention patterns")
    config = AutoConfig.from_pretrained(hf_model_path)
    verify_attention_patterns(mammoth_model, config)

    # Stage 3: Save Mammoth model
    print("\nStage 3: Saving Mammoth model")

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
  # Convert ModernBERT-base with 16-language multi-task setup
  python hfModernBERT2mammoth.py answerdotai/ModernBERT-base ./models/modernbert_base.pt

  # Convert ModernBERT-large
  python hfModernBERT2mammoth.py answerdotai/ModernBERT-large ./models/modernbert_large.pt

Output:
  - 1 shared encoder (ModernBERT weights loaded)
  - 16 language-specific decoders (randomly initialized)
  - Languages: hi, bn, gu, kn, be, bs, hr, eo, ar, he, fa, sw, eu, fi, gl, is
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
        "--model-type",
        default="modernbert",
        help="Architecture type (default: modernbert)"
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

    try:
        # Run conversion
        convert_hf_to_mammoth(
            hf_model_path=args.hf_model_path,
            save_path=args.save_path,
            model_type=args.model_type,
        )

    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())