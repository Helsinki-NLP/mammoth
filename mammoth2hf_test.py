# This script is used to convert a Mammoth model to a HuggingFace model

import os
import torch
import warnings
from collections import OrderedDict
from argparse import Namespace
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    AutoTokenizer,
    PreTrainedTokenizerFast,
)
from huggingface_hub import HfApi, Repository, login
from mammoth.utils.model_saver import (
    load_frame_checkpoint,
    load_parameters_from_checkpoint,
)
from mammoth.model_builder import build_model
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.distributed.tasks import TaskQueueManager
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS


# Load Mammoth model (natively trained on Mammoth)
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


def create_simple_task_queue_manager(frame_checkpoint, task_id=None):
    """Create a simple TaskQueueManager to load the Mammoth model

    Args:
        frame_checkpoint: The loaded checkpoint containing model configuration
        task_id: Optional specific task ID to load. If None, uses first task.

    Returns:
        tuple: (task_queue_manager, corpus_id)
    """
    opts = Namespace()

    # Extract task information from frame checkpoint
    frame_opts = frame_checkpoint["opts"]
    tasks_dict = getattr(frame_opts, "tasks", {})

    if not tasks_dict:
        raise ValueError(
            "No tasks found in checkpoint. The checkpoint may be corrupted or "
            "from an incompatible MAMMOTH version."
        )

    # Select which task to load
    if task_id is not None:
        # User specified a task
        if task_id not in tasks_dict:
            available_tasks = list(tasks_dict.keys())
            raise ValueError(
                f"Task '{task_id}' not found in checkpoint. "
                f"Available tasks: {available_tasks}"
            )
        corpus_id = task_id
        print(f"✓ Loading user-specified task: '{corpus_id}'")
    else:
        # Auto-detect from first task
        corpus_id = next(iter(tasks_dict.keys()))
        if len(tasks_dict) > 1:
            available_tasks = list(tasks_dict.keys())
            print(f"⚠ Multiple tasks found: {available_tasks}")
            print(f"✓ Auto-selecting first task: '{corpus_id}'")
            print(f"  (Use --task to specify a different task)")
        else:
            print(f"✓ Detected task: '{corpus_id}'")

    opts.tasks = tasks_dict
    opts.enc_layers = getattr(frame_opts, "enc_layers", [6])
    opts.dec_layers = getattr(frame_opts, "dec_layers", [6])
    opts.task_distribution_strategy = "weighted_sampling"
    opts.accum_count = [1]
    opts.seed = 42

    world_context = SimpleWorldContext()
    task_manager = TaskQueueManager.from_opts(opts, world_context)

    # Set vocabularies from frame checkpoint
    vocabs_dict = frame_checkpoint["vocab"]
    for task in task_manager.tasks:
        task.src_vocab = vocabs_dict.get(("src", task.src_lang))
        task.tgt_vocab = vocabs_dict.get(("tgt", task.tgt_lang))

    local_task_manager = task_manager.global_to_local(
        node_rank=0, local_rank=0, opts=opts
    )
    local_task_manager.create_all_distributed_components(
        use_attention_bridge=False
    )

    return local_task_manager, corpus_id


def load_mammoth_model(mammoth_model_path, task_id=None):
    """Load Mammoth model from checkpoint files

    Args:
        mammoth_model_path: Path to the checkpoint directory
        task_id: Optional task ID to load. If None, auto-detects from checkpoint.

    Returns:
        tuple: (model, model_opts, vocabs_dict, task_queue_manager)
    """
    print(f"Loading Mammoth model from {mammoth_model_path}")

    # Load frame checkpoint
    frame_checkpoint, frame_checkpoint_path = load_frame_checkpoint(mammoth_model_path)
    if frame_checkpoint is None:
        raise ValueError(f"Could not load frame checkpoint from {mammoth_model_path}")

    print("✓ Frame checkpoint loaded")

    # Extract model configuration and vocabularies
    model_opts = frame_checkpoint["opts"]
    vocabs_dict = frame_checkpoint["vocab"]

    # Create task queue manager with user-specified or auto-detected task
    task_queue_manager, corpus_id = create_simple_task_queue_manager(frame_checkpoint, task_id)

    # Build model
    opts = Namespace()
    opts.train_from = None
    opts.reset_optim = "all"
    opts.model_dtype = getattr(model_opts, "model_dtype", "fp32")
    opts.log_model_structure = False  # Disable logging to reduce output
    opts.adapters = None
    opts.gpu_ranks = []

    model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=corpus_id,
    )

    # Load parameters from checkpoint components
    load_parameters_from_checkpoint(
        frame_checkpoint_path,
        model,
        optim=None,
        task_queue_manager=task_queue_manager,
        reset_optim=True,
        yes_i_messed_with_the_checkpoint=False,
    )

    model.eval()
    print("✓ Mammoth model loaded successfully")

    return model, model_opts, vocabs_dict, task_queue_manager


# Create a Huggingface model skeleton
def create_hf_config_from_mammoth(model_opts, vocabs_dict):
    """Create HuggingFace BartConfig from Mammoth model options"""

    # Extract basic parameters
    vocab_size = len(next(iter(vocabs_dict.values())))
    d_model = getattr(model_opts, "model_dim", 768)
    encoder_layers = getattr(model_opts, "enc_layers", [6])[0]
    decoder_layers = getattr(model_opts, "dec_layers", [6])[0]

    # Extract x_transformers options if available
    xt_opts = getattr(model_opts, "x_transformers_opts", {})
    print(xt_opts)
    encoder_attention_heads = xt_opts.get("heads", 12)
    decoder_attention_heads = xt_opts.get("heads", 12)
    encoder_ffn_dim = int(d_model * xt_opts.get("ff_mult", 4))
    decoder_ffn_dim = encoder_ffn_dim

    config = BartConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        encoder_layers=encoder_layers,
        decoder_layers=decoder_layers,
        encoder_attention_heads=encoder_attention_heads,
        decoder_attention_heads=decoder_attention_heads,
        encoder_ffn_dim=encoder_ffn_dim,
        decoder_ffn_dim=decoder_ffn_dim,
        max_position_embeddings=254, 
        dropout=xt_opts.get("ff_dropout", 0.1),
        attention_dropout=xt_opts.get("attn_dropout", 0.1),
        activation_dropout=xt_opts.get("attn_dropout", 0.1),
        activation_function=getattr(model_opts, "pos_ffn_activation_fn", "relu"),
        normalize_before=xt_opts.get("pre_norm", False),
        normalize_embedding=xt_opts.get("post_emb_norm", True),
        scale_embedding=False,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        forced_eos_token_id=2,
        forced_bos_token_id=0,
        torch_dtype="float32",
        use_cache=True,
    )

    return config


def load_tokenizer_from_vocab(vocab, tokenizer_path_override=None):
    """
    Load HuggingFace tokenizer from MAMMOTH vocab object.

    Automatically detects if the vocab is an HFTokenizerVocab (from training with
    HuggingFace tokenizers) or traditional Vocab (SentencePiece-based).

    Args:
        vocab: MAMMOTH Vocab or HFTokenizerVocab object
        tokenizer_path_override: Optional path to override the tokenizer location

    Returns:
        PreTrainedTokenizerFast: HuggingFace tokenizer compatible with the model
    """
    from mammoth.inputters.vocab import HFTokenizerVocab
    from tokenizers import Tokenizer

    # Determine tokenizer path
    if tokenizer_path_override:
        tokenizer_path = tokenizer_path_override
    elif hasattr(vocab, 'path'):
        tokenizer_path = vocab.path
    else:
        raise ValueError("Could not determine tokenizer path from vocab")

    # Check if this is an HFTokenizerVocab (has .tokenizer attribute)
    if isinstance(vocab, HFTokenizerVocab):
        print(f"✓ Detected HuggingFace tokenizer from training")
        # Load the low-level tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Add post-processor to automatically add BOS/EOS tokens (like BART does)
        from tokenizers.processors import TemplateProcessing
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>",
            pair="<s> $A </s> </s> $B </s>",
            special_tokens=[
                ("<s>", 0),   # BOS token with ID 0
                ("</s>", 2),  # EOS token with ID 2
            ],
        )
        # Wrap with PreTrainedTokenizerFast for HF compatibility
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            mask_token="<mask>",
        )
    else:
        print(f"✓ Detected traditional vocab, loading as SentencePiece tokenizer")
        # For backward compatibility with SentencePiece models
        # Use AutoTokenizer which can handle various tokenizer types
        try:
            hf_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                local_files_only=True
            )
        except Exception as e:
            raise ValueError(
                f"Could not load tokenizer from {tokenizer_path}. "
                f"For SentencePiece models, please provide a HuggingFace tokenizer directory. "
                f"Error: {e}"
            )

    print(f"✓ Tokenizer loaded from {tokenizer_path}")
    return hf_tokenizer


# Map the weights from the mammoth model to the HuggingFace model
def map_mammoth_to_hf_model(mammoth_model, hf_model):
    """
    Maps weights from a MAMMOTH model to a HuggingFace model and returns the updated HF model.

    Args:
        mammoth_model: MAMMOTH model instance
        hf_model: HuggingFace BART model instance

    Returns:
        hf_model: HuggingFace model with mapped weights loaded
    """
    # Get state dict from mammoth model
    mammoth_state_dict = mammoth_model.state_dict()

    # Map weights to HF format
    hf_state_dict = _map_mammoth_to_hf_weights(mammoth_state_dict)

    # Load mapped weights into HF model with diagnostic information
    hf_original_state_dict = hf_model.state_dict()
    n_copied = 0
    n_missed = 0
    missed_keys = []

    print("=== Weight Mapping Diagnostics ===")

    for hf_key, hf_val in hf_state_dict.items():
        if hf_key not in hf_original_state_dict:
            n_missed += 1
            missed_keys.append(f"Missing HF key: '{hf_key}'")
            continue

        hf_original_val = hf_original_state_dict[hf_key]

        if hf_val.shape != hf_original_val.shape:
            n_missed += 1
            missed_keys.append(
                f"Shape mismatch: '{hf_key}' - mapped: {hf_val.shape} vs expected: {hf_original_val.shape}"
            )
            continue

        n_copied += 1

    # Load mapped weights into HF model
    missing_keys, unexpected_keys = hf_model.load_state_dict(
        hf_state_dict, strict=False
    )

    print(f"✓ Weight mapping summary: {n_copied} copied, {n_missed} missed")
    print(f"  - Missing keys from mapping: {len(missing_keys)} keys")
    print(f"  - Unexpected keys: {len(unexpected_keys)} keys")

    if missed_keys:
        print("Mapping issues:")
        for key in missed_keys:
            print(f"  - {key}")

    if missing_keys:
        print("Missing keys in HF model (not mapped):")
        for key in missing_keys[:10]:  # Show first 10 to avoid cluttering
            print(f"  - {key}")
        if len(missing_keys) > 10:
            print(f"  ... and {len(missing_keys) - 10} more")

    if unexpected_keys:
        print("Unexpected keys in mapping:")
        for key in unexpected_keys[:10]:  # Show first 10 to avoid cluttering
            print(f"  - {key}")
        if len(unexpected_keys) > 10:
            print(f"  ... and {len(unexpected_keys) - 10} more")

    return hf_model


def _map_mammoth_to_hf_weights(mammoth_state_dict):
    """
    Maps MAMMOTH model weights to HuggingFace BART format.

    Args:
        mammoth_state_dict (dict): State dictionary from MAMMOTH model

    Returns:
        dict: Mapped state dictionary compatible with HuggingFace BART
    """
    hf_state_dict = {}

    print(f"Mapping {len(mammoth_state_dict)} MAMMOTH parameters to HF format...")

    # Auto-detect task identifier from the state dict keys
    task_id = None
    for key in mammoth_state_dict.keys():
        if key.startswith("encoder."):
            # Extract task ID from pattern: encoder.{task_id}.token_emb...
            parts = key.split(".")
            if len(parts) >= 2:
                task_id = parts[1]
                break

    if task_id is None:
        raise ValueError(
            "Could not auto-detect task identifier from MAMMOTH state dict. "
            "Expected keys like 'encoder.{task_id}.token_emb...'"
        )

    print(f"✓ Detected task identifier: '{task_id}'")

    # Special mappings as specified
    if f"decoder.{task_id}.to_logits.bias" in mammoth_state_dict:
        # HuggingFace expects final_logits_bias to have shape [1, vocab_size]
        bias_tensor = mammoth_state_dict[f"decoder.{task_id}.to_logits.bias"]
        hf_state_dict["final_logits_bias"] = bias_tensor.unsqueeze(0)

    if f"decoder.{task_id}.to_logits.weight" in mammoth_state_dict:
        hf_state_dict["lm_head.weight"] = mammoth_state_dict[
            f"decoder.{task_id}.to_logits.weight"
        ]

    # Shared embedding weight mapped to both locations
    if f"encoder.{task_id}.token_emb.emb.weight" in mammoth_state_dict:
        shared_weight = mammoth_state_dict[
            f"encoder.{task_id}.token_emb.emb.weight"
        ]
        hf_state_dict["model.shared.weight"] = shared_weight
        hf_state_dict["model.encoder.embed_tokens.weight"] = shared_weight

    # Decoder token embeddings
    if f"decoder.{task_id}.token_emb.emb.weight" in mammoth_state_dict:
        hf_state_dict["model.decoder.embed_tokens.weight"] = mammoth_state_dict[
            f"decoder.{task_id}.token_emb.emb.weight"
        ]

    # Position embeddings
    if f"encoder.{task_id}.pos_emb.emb.weight" in mammoth_state_dict:
        hf_state_dict["model.encoder.embed_positions.weight"] = mammoth_state_dict[
            f"encoder.{task_id}.pos_emb.emb.weight"
        ]

    if f"decoder.{task_id}.pos_emb.emb.weight" in mammoth_state_dict:
        hf_state_dict["model.decoder.embed_positions.weight"] = mammoth_state_dict[
            f"decoder.{task_id}.pos_emb.emb.weight"
        ]

    # Layer normalization after embeddings
    if f"encoder.{task_id}.post_emb_norm.ln.weight" in mammoth_state_dict:
        hf_state_dict["model.encoder.layernorm_embedding.weight"] = mammoth_state_dict[
            f"encoder.{task_id}.post_emb_norm.ln.weight"
        ]

    if f"encoder.{task_id}.post_emb_norm.ln.bias" in mammoth_state_dict:
        hf_state_dict["model.encoder.layernorm_embedding.bias"] = mammoth_state_dict[
            f"encoder.{task_id}.post_emb_norm.ln.bias"
        ]

    if f"decoder.{task_id}.post_emb_norm.ln.weight" in mammoth_state_dict:
        hf_state_dict["model.decoder.layernorm_embedding.weight"] = mammoth_state_dict[
            f"decoder.{task_id}.post_emb_norm.ln.weight"
        ]

    if f"decoder.{task_id}.post_emb_norm.ln.bias" in mammoth_state_dict:
        hf_state_dict["model.decoder.layernorm_embedding.bias"] = mammoth_state_dict[
            f"decoder.{task_id}.post_emb_norm.ln.bias"
        ]

    # Map encoder layers
    encoder_layers_found = 0
    for layer_idx in range(
        12
    ):  # Assuming 6 layers based on the data, but being safe with 12
        if _map_encoder_layer(mammoth_state_dict, hf_state_dict, layer_idx, task_id):
            encoder_layers_found += 1

    # Map decoder layers
    decoder_layers_found = 0
    for layer_idx in range(18):  # Assuming more decoder layers based on the data
        if _map_decoder_layer(mammoth_state_dict, hf_state_dict, layer_idx, task_id):
            decoder_layers_found += 1

    print(
        f"✓ Found {encoder_layers_found} encoder layers, {decoder_layers_found//3} decoder layers"
    )
    print(f"✓ Created {len(hf_state_dict)} HF parameter mappings")

    return hf_state_dict


def _map_encoder_layer(mammoth_state_dict, hf_state_dict, layer_idx, task_id):
    """Map a single encoder layer from MAMMOTH to HuggingFace format."""
    mammoth_prefix = f"encoder.{task_id}.attn_layers.attention_layers_stack.0.layers.{layer_idx * 2}"
    hf_prefix = f"model.encoder.layers.{layer_idx}"

    # Check if this layer exists
    if f"{mammoth_prefix}.0.2.ln.weight" not in mammoth_state_dict:
        return False

    # Self attention layer normalization
    mammoth_ln_key = f"{mammoth_prefix}.0.2.ln"
    hf_ln_key = f"{hf_prefix}.self_attn_layer_norm"
    _map_layer_norm(mammoth_state_dict, hf_state_dict, mammoth_ln_key, hf_ln_key)

    # Self attention projections
    mammoth_attn_prefix = f"{mammoth_prefix}.1"
    hf_attn_prefix = f"{hf_prefix}.self_attn"
    _map_attention_projections(
        mammoth_state_dict, hf_state_dict, mammoth_attn_prefix, hf_attn_prefix
    )

    # Feed forward layer
    ff_layer_idx = layer_idx * 2 + 1
    mammoth_ff_prefix = f"encoder.{task_id}.attn_layers.attention_layers_stack.0.layers.{ff_layer_idx}"
    hf_ff_prefix = f"{hf_prefix}"

    # FF layer normalization
    mammoth_ff_ln_key = f"{mammoth_ff_prefix}.0.2.ln"
    hf_ff_ln_key = f"{hf_prefix}.final_layer_norm"
    _map_layer_norm(mammoth_state_dict, hf_state_dict, mammoth_ff_ln_key, hf_ff_ln_key)

    # FF projections
    _map_ff_projections(
        mammoth_state_dict, hf_state_dict, f"{mammoth_ff_prefix}.1.ff", hf_ff_prefix
    )

    return True


def _map_decoder_layer(mammoth_state_dict, hf_state_dict, layer_idx, task_id):
    """Map a single decoder layer from MAMMOTH to HuggingFace format."""
    # Convert mammoth layer index to actual layer number (every 3 mammoth layers = 1 HF layer)
    if layer_idx % 3 == 0:  # Self attention layer
        hf_layer_idx = layer_idx // 3
        mammoth_prefix = f"decoder.{task_id}.attn_layers.attention_layers_stack.0.layers.{layer_idx}"
        hf_prefix = f"model.decoder.layers.{hf_layer_idx}"

        # Check if this layer exists
        if f"{mammoth_prefix}.0.2.ln.weight" not in mammoth_state_dict:
            return False

        # Self attention layer normalization
        mammoth_ln_key = f"{mammoth_prefix}.0.2.ln"
        hf_ln_key = f"{hf_prefix}.self_attn_layer_norm"
        _map_layer_norm(mammoth_state_dict, hf_state_dict, mammoth_ln_key, hf_ln_key)

        # Self attention projections
        mammoth_attn_prefix = f"{mammoth_prefix}.1"
        hf_attn_prefix = f"{hf_prefix}.self_attn"
        _map_attention_projections(
            mammoth_state_dict, hf_state_dict, mammoth_attn_prefix, hf_attn_prefix
        )

        return True

    elif layer_idx % 3 == 1:  # Cross attention layer
        hf_layer_idx = layer_idx // 3
        mammoth_prefix = f"decoder.{task_id}.attn_layers.attention_layers_stack.0.layers.{layer_idx}"
        hf_prefix = f"model.decoder.layers.{hf_layer_idx}"

        # Check if this layer exists
        if f"{mammoth_prefix}.0.2.ln.weight" not in mammoth_state_dict:
            return False

        # Cross attention layer normalization
        mammoth_ln_key = f"{mammoth_prefix}.0.2.ln"
        hf_ln_key = f"{hf_prefix}.encoder_attn_layer_norm"
        _map_layer_norm(mammoth_state_dict, hf_state_dict, mammoth_ln_key, hf_ln_key)

        # Cross attention projections
        mammoth_attn_prefix = f"{mammoth_prefix}.1"
        hf_attn_prefix = f"{hf_prefix}.encoder_attn"
        _map_attention_projections(
            mammoth_state_dict, hf_state_dict, mammoth_attn_prefix, hf_attn_prefix
        )

        return True

    elif layer_idx % 3 == 2:  # Feed forward layer
        hf_layer_idx = layer_idx // 3
        mammoth_prefix = f"decoder.{task_id}.attn_layers.attention_layers_stack.0.layers.{layer_idx}"
        hf_prefix = f"model.decoder.layers.{hf_layer_idx}"

        # Check if this layer exists
        if f"{mammoth_prefix}.0.2.ln.weight" not in mammoth_state_dict:
            return False

        # FF layer normalization
        mammoth_ln_key = f"{mammoth_prefix}.0.2.ln"
        hf_ln_key = f"{hf_prefix}.final_layer_norm"
        _map_layer_norm(mammoth_state_dict, hf_state_dict, mammoth_ln_key, hf_ln_key)

        # FF projections
        _map_ff_projections(
            mammoth_state_dict, hf_state_dict, f"{mammoth_prefix}.1.ff", hf_prefix
        )

        return True

    return False


def _map_layer_norm(mammoth_state_dict, hf_state_dict, mammoth_prefix, hf_prefix):
    """Map layer normalization weights."""
    if f"{mammoth_prefix}.weight" in mammoth_state_dict:
        hf_state_dict[f"{hf_prefix}.weight"] = mammoth_state_dict[
            f"{mammoth_prefix}.weight"
        ]

    if f"{mammoth_prefix}.bias" in mammoth_state_dict:
        hf_state_dict[f"{hf_prefix}.bias"] = mammoth_state_dict[
            f"{mammoth_prefix}.bias"
        ]


def _map_attention_projections(
    mammoth_state_dict, hf_state_dict, mammoth_prefix, hf_prefix
):
    """Map attention projection weights (q, k, v, out)."""
    projections = ["q", "k", "v"]

    for proj in projections:
        mammoth_key = f"{mammoth_prefix}.to_{proj}"
        hf_key = f"{hf_prefix}.{proj}_proj"

        if f"{mammoth_key}.weight" in mammoth_state_dict:
            hf_state_dict[f"{hf_key}.weight"] = mammoth_state_dict[
                f"{mammoth_key}.weight"
            ]

        if f"{mammoth_key}.bias" in mammoth_state_dict:
            hf_state_dict[f"{hf_key}.bias"] = mammoth_state_dict[f"{mammoth_key}.bias"]

    # Output projection
    mammoth_out_key = f"{mammoth_prefix}.to_out"
    hf_out_key = f"{hf_prefix}.out_proj"

    if f"{mammoth_out_key}.weight" in mammoth_state_dict:
        hf_state_dict[f"{hf_out_key}.weight"] = mammoth_state_dict[
            f"{mammoth_out_key}.weight"
        ]

    if f"{mammoth_out_key}.bias" in mammoth_state_dict:
        hf_state_dict[f"{hf_out_key}.bias"] = mammoth_state_dict[
            f"{mammoth_out_key}.bias"
        ]


def _map_ff_projections(mammoth_state_dict, hf_state_dict, mammoth_prefix, hf_prefix):
    """Map feed-forward layer weights (fc1, fc2)."""
    # First linear layer (fc1)
    if f"{mammoth_prefix}.0.weight" in mammoth_state_dict:
        hf_state_dict[f"{hf_prefix}.fc1.weight"] = mammoth_state_dict[
            f"{mammoth_prefix}.0.weight"
        ]

    if f"{mammoth_prefix}.0.bias" in mammoth_state_dict:
        hf_state_dict[f"{hf_prefix}.fc1.bias"] = mammoth_state_dict[
            f"{mammoth_prefix}.0.bias"
        ]

    # Second linear layer (fc2)
    if f"{mammoth_prefix}.3.weight" in mammoth_state_dict:
        hf_state_dict[f"{hf_prefix}.fc2.weight"] = mammoth_state_dict[
            f"{mammoth_prefix}.3.weight"
        ]

    if f"{mammoth_prefix}.3.bias" in mammoth_state_dict:
        hf_state_dict[f"{hf_prefix}.fc2.bias"] = mammoth_state_dict[
            f"{mammoth_prefix}.3.bias"
        ]


def convert_mammoth_to_hf(
    mammoth_model_path, hf_model_path, tokenizer_path=None, task_id=None, push_to_hub=False
):
    """
    Convert a MAMMOTH model to HuggingFace format.

    Args:
        mammoth_model_path (str): Path to the MAMMOTH model checkpoint
        hf_model_path (str): Path where to save the converted HuggingFace model
        tokenizer_path (str, optional): Path to tokenizer (auto-detected from vocab if not provided)
        task_id (str, optional): Task ID to convert (auto-detected if not provided)
        push_to_hub (bool): Whether to push the model to HuggingFace Hub

    Returns:
        tuple: (converted_hf_model, tokenizer, config)
    """
    # Load the Mammoth model (natively trained on Mammoth)
    mammoth_model, model_opts, vocabs_dict, task_queue_manager = load_mammoth_model(
        mammoth_model_path, task_id=task_id
    )

    # Create the HuggingFace model
    mammoth_model_config = create_hf_config_from_mammoth(model_opts, vocabs_dict)
    converted_hf_model = BartForConditionalGeneration(mammoth_model_config)

    # Load tokenizer from vocab (automatically detects HF tokenizer vs SentencePiece)
    # Use any vocab from vocabs_dict (they should all point to the same tokenizer)
    sample_vocab = next(iter(vocabs_dict.values()))
    tokenizer = load_tokenizer_from_vocab(sample_vocab, tokenizer_path_override=tokenizer_path)

    # Save state dict keys for structure verification
    os.makedirs(hf_model_path, exist_ok=True)
    mammoth_keys_path = os.path.join(hf_model_path, "mammoth_keys.txt")
    with open(mammoth_keys_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("MAMMOTH MODEL PARAMETER KEYS\n")
        f.write("=" * 80 + "\n\n")
        for key in mammoth_model.state_dict().keys():
            f.write(f"{key}\n")
    print(f"✓ Mammoth parameter keys saved to {mammoth_keys_path}")

    # Save HF state dict keys
    hf_keys_path = os.path.join(hf_model_path, "hf_keys.txt")
    with open(hf_keys_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("HUGGINGFACE MODEL PARAMETER KEYS\n")
        f.write("=" * 80 + "\n\n")
        for key in converted_hf_model.state_dict().keys():
            f.write(f"{key}\n")
    print(f"✓ HF parameter keys saved to {hf_keys_path}")

    # Map the weights from the Mammoth model to the HuggingFace model
    map_mammoth_to_hf_model(mammoth_model, converted_hf_model)
    print("✓ Weight mapping completed successfully")

    # Save the converted model
    converted_hf_model.save_pretrained(hf_model_path)
    tokenizer.save_pretrained(hf_model_path)

    mammoth_model_config.save_pretrained(hf_model_path)
    print("✓ Model saved successfully")

    # Push to HuggingFace Hub if requested
    if push_to_hub:
        converted_hf_model.push_to_hub(hf_model_path)
        tokenizer.push_to_hub(hf_model_path)
        mammoth_model_config.push_to_hub(hf_model_path)
        print(
            "✓ Model, config, and tokenizer successfully pushed to Hugging Face Model Hub"
        )

    return converted_hf_model, tokenizer, mammoth_model_config


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert MAMMOTH model to HuggingFace format"
    )
    parser.add_argument(
        "mammoth_model_path", help="Path to the MAMMOTH model checkpoint"
    )
    parser.add_argument(
        "hf_model_path", help="Path where to save the converted HuggingFace model"
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Task ID to convert (optional - auto-detects first task if not provided)"
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Path to tokenizer (optional - auto-detected from checkpoint if not provided)"
    )
    parser.add_argument(
        "--push-to-hub", action="store_true", help="Push the model to HuggingFace Hub"
    )

    args = parser.parse_args()

    convert_mammoth_to_hf(
        args.mammoth_model_path,
        args.hf_model_path,
        tokenizer_path=args.tokenizer,
        task_id=args.task,
        push_to_hub=args.push_to_hub,
    )
