# This script is used to convert a Mammoth model to a HuggingFace model
import os
import argparse
from argparse import Namespace
from transformers import (
    BartForConditionalGeneration,
    BartConfig,
    AutoTokenizer,
)
from huggingface_hub import HfApi, login
from mammoth.utils.model_saver import (
    load_frame_checkpoint,
    load_parameters_from_checkpoint,
)
from mammoth.model_builder import build_model
from mammoth.distributed.contexts import WorldContext, DeviceContext, DeviceContextEnum
from mammoth.distributed.tasks import TaskQueueManager


# load mammoth model
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


def create_simple_task_queue_manager(frame_checkpoint, corpus_id="bart_translation"):
    """Create a simple TaskQueueManager to load the Mammoth model"""
    opts = Namespace()

    # Extract task information from frame checkpoint
    frame_opts = frame_checkpoint["opts"]
    tasks_dict = getattr(frame_opts, "tasks", {})

    if not tasks_dict:
        # Create default task if none found
        src_lang = "en"  # Default, will be overridden
        tgt_lang = "es"  # Default, will be overridden
        tasks_dict = {
            corpus_id: {
                "src_tgt": f"{src_lang}-{tgt_lang}",
                "weight": 1.0,
                "introduce_at_training_step": 0,
                "node_gpu": "0:0",
                "enc_sharing_group": [src_lang],
                "dec_sharing_group": [tgt_lang],
            }
        }

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
        use_attention_bridge=False, new_group_func=lambda ranks: None
    )

    return local_task_manager


def load_mammoth_model(mammoth_model_path, corpus_id="bart_translation"):
    """Load Mammoth model from checkpoint files"""
    print(f"Loading Mammoth model from {mammoth_model_path}")

    # Load frame checkpoint
    frame_checkpoint, frame_checkpoint_path = load_frame_checkpoint(mammoth_model_path)
    if frame_checkpoint is None:
        raise ValueError(f"Could not load frame checkpoint from {mammoth_model_path}")

    print("✓ Frame checkpoint loaded")

    # Extract model configuration and vocabularies
    model_opts = frame_checkpoint["opts"]
    vocabs_dict = frame_checkpoint["vocab"]

    # Create task queue manager
    task_queue_manager = create_simple_task_queue_manager(frame_checkpoint, corpus_id)

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


# Build HuggingFace config from mammoth model
def create_hf_config_from_mammoth(model_opts, vocabs_dict):
    """Create HuggingFace BartConfig from Mammoth model options"""

    # Extract basic parameters
    vocab_size = len(next(iter(vocabs_dict.values())))
    d_model = getattr(model_opts, "model_dim", 768)
    encoder_layers = getattr(model_opts, "enc_layers", [6])[0]
    decoder_layers = getattr(model_opts, "dec_layers", [6])[0]

    # Extract x_transformers options if available
    xt_opts = getattr(model_opts, "x_transformers_opts", {})
    # print(xt_opts)
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
        max_position_embeddings=1024,
        dropout=xt_opts.get("ff_dropout", 0.1),
        attention_dropout=xt_opts.get("attn_dropout", 0.1),
        activation_dropout=xt_opts.get("attn_dropout", 0.1),
        activation_function=getattr(model_opts, "pos_ffn_activation_fn", "relu"),
        normalize_before=xt_opts.get("pre_norm", False),
        normalize_embedding=xt_opts.get("post_emb_norm", True),
        # add_bias_logits=True,
        # add_final_layer_norm=True,
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


# Map weights from mammoth model to HuggingFace model
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

    # Special mappings as specified
    if "decoder.bart_translation.to_logits.bias" in mammoth_state_dict:
        # HuggingFace expects final_logits_bias to have shape [1, vocab_size]
        bias_tensor = mammoth_state_dict["decoder.bart_translation.to_logits.bias"]
        hf_state_dict["final_logits_bias"] = bias_tensor.unsqueeze(0)

    if "decoder.bart_translation.to_logits.weight" in mammoth_state_dict:
        hf_state_dict["lm_head.weight"] = mammoth_state_dict[
            "decoder.bart_translation.to_logits.weight"
        ]

    # Shared embedding weight mapped to both locations
    if "encoder.bart_translation.token_emb.emb.weight" in mammoth_state_dict:
        shared_weight = mammoth_state_dict[
            "encoder.bart_translation.token_emb.emb.weight"
        ]
        hf_state_dict["model.shared.weight"] = shared_weight
        hf_state_dict["model.encoder.embed_tokens.weight"] = shared_weight

    # Decoder token embeddings
    if "decoder.bart_translation.token_emb.emb.weight" in mammoth_state_dict:
        hf_state_dict["model.decoder.embed_tokens.weight"] = mammoth_state_dict[
            "decoder.bart_translation.token_emb.emb.weight"
        ]

    # Position embeddings
    if "encoder.bart_translation.pos_emb.emb.weight" in mammoth_state_dict:
        hf_state_dict["model.encoder.embed_positions.weight"] = mammoth_state_dict[
            "encoder.bart_translation.pos_emb.emb.weight"
        ]

    if "decoder.bart_translation.pos_emb.emb.weight" in mammoth_state_dict:
        hf_state_dict["model.decoder.embed_positions.weight"] = mammoth_state_dict[
            "decoder.bart_translation.pos_emb.emb.weight"
        ]

    # Layer normalization after embeddings
    if "encoder.bart_translation.post_emb_norm.ln.weight" in mammoth_state_dict:
        hf_state_dict["model.encoder.layernorm_embedding.weight"] = mammoth_state_dict[
            "encoder.bart_translation.post_emb_norm.ln.weight"
        ]

    if "encoder.bart_translation.post_emb_norm.ln.bias" in mammoth_state_dict:
        hf_state_dict["model.encoder.layernorm_embedding.bias"] = mammoth_state_dict[
            "encoder.bart_translation.post_emb_norm.ln.bias"
        ]

    if "decoder.bart_translation.post_emb_norm.ln.weight" in mammoth_state_dict:
        hf_state_dict["model.decoder.layernorm_embedding.weight"] = mammoth_state_dict[
            "decoder.bart_translation.post_emb_norm.ln.weight"
        ]

    if "decoder.bart_translation.post_emb_norm.ln.bias" in mammoth_state_dict:
        hf_state_dict["model.decoder.layernorm_embedding.bias"] = mammoth_state_dict[
            "decoder.bart_translation.post_emb_norm.ln.bias"
        ]

    # Map encoder layers
    encoder_layers_found = 0
    for layer_idx in range(
        12
    ):  # Assuming 6 layers based on the data, but being safe with 12
        if _map_encoder_layer(mammoth_state_dict, hf_state_dict, layer_idx):
            encoder_layers_found += 1

    # Map decoder layers
    decoder_layers_found = 0
    for layer_idx in range(18):  # Assuming more decoder layers based on the data
        if _map_decoder_layer(mammoth_state_dict, hf_state_dict, layer_idx):
            decoder_layers_found += 1

    print(
        f"✓ Found {encoder_layers_found} encoder layers, {decoder_layers_found//3} decoder layers"
    )
    print(f"✓ Created {len(hf_state_dict)} HF parameter mappings")

    return hf_state_dict


def _map_encoder_layer(mammoth_state_dict, hf_state_dict, layer_idx):
    """Map a single encoder layer from MAMMOTH to HuggingFace format."""
    mammoth_prefix = f"encoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{layer_idx * 2}"
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
    mammoth_ff_prefix = f"encoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{ff_layer_idx}"
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


def _map_decoder_layer(mammoth_state_dict, hf_state_dict, layer_idx):
    """Map a single decoder layer from MAMMOTH to HuggingFace format."""
    # Convert mammoth layer index to actual layer number (every 3 mammoth layers = 1 HF layer)
    if layer_idx % 3 == 0:  # Self attention layer
        hf_layer_idx = layer_idx // 3
        mammoth_prefix = f"decoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{layer_idx}"
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
        mammoth_prefix = f"decoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{layer_idx}"
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
        mammoth_prefix = f"decoder.bart_translation.attn_layers.attention_layers_stack.0.layers.{layer_idx}"
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


def convert_mammoth_to_hf(mammoth_model_path, hf_model_name, original_tokenizer_path):
    """
    Convert a Mammoth model to HuggingFace format.

    Args:
        mammoth_model_path (str): Path to the Mammoth model checkpoint
        hf_model_name (str): Name/path for the new HuggingFace model output
        original_tokenizer_path (str): Path to the original HuggingFace tokenizer

    Returns:
        tuple: (converted_hf_model, converted_hf_model_config, tokenizer)
    """
    print(
        f"Converting Mammoth model from {mammoth_model_path} to HF format at {hf_model_name}"
    )

    # Load the mammoth model
    mammoth_model, model_opts, vocabs_dict, task_queue_manager = load_mammoth_model(
        mammoth_model_path
    )

    # Build the HuggingFace config
    converted_hf_model_config = create_hf_config_from_mammoth(model_opts, vocabs_dict)

    # Build the HuggingFace model
    converted_hf_model = BartForConditionalGeneration(converted_hf_model_config)

    # Map the weights from the mammoth model to the HuggingFace model
    map_mammoth_to_hf_model(mammoth_model, converted_hf_model)

    # Save the converted HuggingFace model
    print(f"Saving converted model to {hf_model_name}")
    converted_hf_model.save_pretrained(hf_model_name)
    converted_hf_model_config.save_pretrained(hf_model_name)

    # Load and save the original HF tokenizer
    print(f"Loading tokenizer from {original_tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(original_tokenizer_path)
    tokenizer.save_pretrained(hf_model_name)

    print("✓ Conversion completed successfully!")
    return converted_hf_model, converted_hf_model_config, tokenizer


def push_to_hub(model, config, tokenizer, hub_model_id):
    """
    Push the converted model to HuggingFace Hub.

    Args:
        model: HuggingFace model to push
        config: HuggingFace config to push
        tokenizer: HuggingFace tokenizer to push
    """
    print(f"Pushing model to HuggingFace Hub as {hub_model_id}")

    try:
        # Push model to hub
        print("Uploading model...")
        model.push_to_hub(hub_model_id)

        print("Uploading config...")
        config.push_to_hub(hub_model_id)

        print("Uploading tokenizer...")
        tokenizer.push_to_hub(hub_model_id)

        print(f"✓ Model, config, and tokenizer successfully pushed Hugging Face Model Hub")
        return True

    except Exception as e:
        print(f"Error pushing to hub: {e}")
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert a Mammoth model to HuggingFace format"
    )
    parser.add_argument(
        "--mammoth_model",
        type=str,
        required=True,
        help="Path to the Mammoth model checkpoint",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        required=True,
        help="Name/path for the new HuggingFace model output",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to the original HuggingFace tokenizer",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the converted model to HuggingFace Hub",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert the model
    converted_model, config, tokenizer = convert_mammoth_to_hf(
        args.mammoth_model, args.hf_model, args.tokenizer
    )

    # Push to hub if requested
    if args.push_to_hub:
        success = push_to_hub(
            converted_model,
            config,
            tokenizer,
            args.hf_model,
        )
        if not success:
            exit(1)
