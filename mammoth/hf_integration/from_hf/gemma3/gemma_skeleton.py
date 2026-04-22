#!/usr/bin/env python3
"""
Build a Mammoth multi-task model with the SAME architecture as HuggingFace
Gemma3, but WITHOUT any weight mapping (randomly initialized) and WITHOUT
any HF -> Mammoth tokenizer conversion.

Compared to gemma2mammoth.py, this script:
  - Does NOT load HF Gemma3 weights
  - Does NOT create an intermediate x-transformer model
  - Does NOT run map_xt_to_mammoth_weights
  - Does NOT rename HF special tokens to Mammoth conventions; it loads
    Mammoth-native vocabs directly (either .txt Vocab files or .json
    HFTokenizerVocab files that are already Mammoth-compatible).

The HF model path is still used, but only for its config (hidden_size,
num_hidden_layers, head_dim, rope_theta, sliding_window, etc.) so the
decoder shape matches Gemma3 exactly.

Usage:
    python build_mammoth_gemma3_arch.py <hf_model_path> <save_path> \\
        --src-vocab en /path/to/en_vocab.{txt,json} \\
        --tgt-vocab ar /path/to/ar_vocab.{txt,json} \\
        [--tgt-vocab fr /path/to/fr_vocab.{txt,json} ...]
"""

import os  # noqa: E402
import sys  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)

from argparse import Namespace  # noqa: E402

from transformers import AutoConfig  # noqa: E402

from mammoth.inputters.vocab import get_vocab  # noqa: E402
from mammoth.model_builder import build_model  # noqa: E402
from mammoth.utils.model_saver import build_model_saver  # noqa: E402
from mammoth.utils.optimizers import MultipleOptimizer  # noqa: E402

# Reuse task/optimizer helpers from the converter script. The architecture
# builder `create_model_opts` is defined locally so all architecture knobs
# live in this file and are easy to tweak.
from gemma2mammoth import (  # noqa: E402
    calculate_model_parameters,
    create_opts,
    create_task_queue_manager,
    format_parameter_count,
)


# =============================================================================
# Architecture builder — EDIT THIS FUNCTION TO CHANGE THE MODEL ARCHITECTURE
# =============================================================================
def create_model_opts(hf_model_path):
    """
    Create Mammoth `model_opts` dictating the ENTIRE model architecture.

    This is the single source of truth for architecture. All downstream
    components (build_model, TaskQueueManager's enc_layers/dec_layers,
    the model saver) read from the Namespace returned here.

    Defaults below mirror HF Gemma3 exactly for the decoder:
      - dec_model_dim       = config.hidden_size
      - dec_layers          = [config.num_hidden_layers]
      - dec_heads           = config.num_attention_heads
      - dec_attn_dim_head   = config.head_dim
      - dec_attn_kv_heads   = config.num_key_value_heads   (MQA/GQA)
      - dec_ff_mult         = config.intermediate_size / config.hidden_size
      - dec_ff_glu          = True  (gate_proj * GELU(up_proj))
      - dec_use_rmsnorm     = True
      - dec_sandwich_norm   = True  (4 norms per layer)
      - dec_attn_qk_norm    = True  (+ learnable per-elem scale)
      - dec_scaled_embeddings = True (multiply by sqrt(hidden_size))
      - dual RoPE: global_rope_theta + local_rope_theta + sliding_window

    Encoder is a small randomly-initialized transformer whose dimension may
    differ from the decoder. The cross-attention layer in each decoder block
    projects encoder outputs via dec_cross_attn_dim_context = enc_model_dim.
    """
    config = AutoConfig.from_pretrained(
        hf_model_path, local_files_only=False, trust_remote_code=False
    )
    if hasattr(config, "text_config"):
        config = config.text_config

    dec_ff_mult = config.intermediate_size / config.hidden_size
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )

    model_opts = Namespace()

    # ---- Basic ----
    model_opts.model_dtype = "bf16"

    # ---- Layer counts / dims ----
    model_opts.enc_layers = [6]                       # EDIT: encoder depth
    model_opts.dec_layers = [config.num_hidden_layers]
    model_opts.enc_model_dim = 512                    # EDIT: encoder dim
    model_opts.dec_model_dim = config.hidden_size
    # model_opts.model_dim = config.hidden_size       # fallback shared-dim

    # ---- Sliding window / dual RoPE (decoder matches Gemma3) ----
    model_opts.dec_sliding_window = getattr(config, "sliding_window", -1)
    model_opts.dec_global_attn_every_n_layers = getattr(
        config, "global_attn_every_n_layers", 3
    )
    model_opts.dec_global_rope_theta = getattr(config, "rope_theta", 1_000_000.0)
    model_opts.dec_local_rope_theta = getattr(config, "rope_local_base_freq", 10_000.0)

    # ---- Encoder uses standard full attention ----
    model_opts.enc_sliding_window = -1
    model_opts.enc_global_attn_every_n_layers = 3
    model_opts.enc_global_rope_theta = 10_000.0
    model_opts.enc_local_rope_theta = 10_000.0

    # ---- x_transformers_opts: the bulk of per-layer architecture ----
    model_opts.x_transformers_opts = {
        # Shared
        "attn_flash": True,

        # Decoder TransformerWrapper (must match Gemma3)
        "dec_post_emb_norm": False,
        "dec_max_seq_len": config.max_position_embeddings,
        "dec_scaled_embeddings": True,

        # Decoder AttentionLayers (must match Gemma3)
        "dec_heads": config.num_attention_heads,
        "dec_attn_dim_head": head_dim,
        "dec_attn_dropout": config.attention_dropout,
        "dec_attn_qk_norm": True,
        "dec_attn_qk_norm_dim_scale": True,
        "dec_attn_kv_heads": config.num_key_value_heads,

        "dec_ff_mult": dec_ff_mult,
        "dec_ff_dropout": 0.0,
        "dec_ff_glu": True,
        "dec_ff_no_bias": True,

        "dec_rotary_pos_emb": True,
        "dec_rotary_pos_emb_base": config.rope_theta,
        "dec_use_rmsnorm": True,
        "dec_sandwich_norm": True,
        "dec_cross_attn_dim_context": 512,  # must equal enc_model_dim below

        # Encoder TransformerWrapper
        "enc_post_emb_norm": True,
        "enc_max_seq_len": 512,
        "enc_scaled_embeddings": False,

        # Encoder AttentionLayers
        "enc_heads": 8,
        "enc_attn_dim_head": 64,
        "enc_ff_mult": 4.0,
        "enc_ff_glu": False,
        "enc_ff_no_bias": True,
        "enc_rotary_pos_emb": True,
    }
    # Keep decoder cross-attn context in sync with enc_model_dim
    model_opts.x_transformers_opts["dec_cross_attn_dim_context"] = model_opts.enc_model_dim

    # ---- Init / misc ----
    model_opts.param_init = 0.0
    model_opts.param_init_glorot = True
    model_opts.attention_bridge = None
    model_opts.ab_layers = []
    model_opts.adapters = None
    model_opts.enable_embeddingless = False
    model_opts.normformer = False
    model_opts.dropout = [0.0]
    model_opts.attention_dropout = [config.attention_dropout]

    # ---- Logging ----
    print(f"  decoder: dim={model_opts.dec_model_dim}, "
          f"layers={model_opts.dec_layers[0]}, "
          f"heads={config.num_attention_heads}, "
          f"kv_heads={config.num_key_value_heads}, "
          f"head_dim={head_dim}, ff_mult={dec_ff_mult:.2f}")
    print(f"  encoder: dim={model_opts.enc_model_dim}, "
          f"layers={model_opts.enc_layers[0]}, heads=8, head_dim=64, ff_mult=4.0")
    if model_opts.dec_sliding_window > 0:
        print(f"  decoder sliding window: {model_opts.dec_sliding_window} "
              f"(global attn every {model_opts.dec_global_attn_every_n_layers} layers)")
        print(f"  RoPE: global={model_opts.dec_global_rope_theta}, "
              f"local={model_opts.dec_local_rope_theta}")
    else:
        print("  decoder sliding window: disabled (full causal attention)")

    return model_opts


def build_vocabs_dict(src_vocab_path, src_lang, tgt_vocab_paths):
    """
    Build a Mammoth vocabs_dict directly from Mammoth-compatible vocab files.

    Supports both:
      - .txt files -> traditional Mammoth `Vocab`
      - .json files -> `HFTokenizerVocab` (already Mammoth-compatible)

    Args:
        src_vocab_path: Path to source vocab (.txt or .json).
        src_lang: Source language code.
        tgt_vocab_paths: Dict {lang: path} for target vocabs.

    Returns:
        dict keyed by (side, lang) -> vocab instance.
    """
    if not os.path.exists(src_vocab_path):
        raise FileNotFoundError(f"Source vocab not found: {src_vocab_path}")

    vocabs_dict = {}
    src_vocab = get_vocab(path=src_vocab_path, lang=f"src_{src_lang}", size=None)
    vocabs_dict[("src", src_lang)] = src_vocab
    print(f"  ✓ src [{src_lang}]: {len(src_vocab)} tokens  ({src_vocab_path})")

    for tgt_lang, tgt_path in tgt_vocab_paths.items():
        if not os.path.exists(tgt_path):
            raise FileNotFoundError(f"Target vocab not found: {tgt_path}")
        tgt_vocab = get_vocab(path=tgt_path, lang=f"tgt_{tgt_lang}", size=None)
        vocabs_dict[("tgt", tgt_lang)] = tgt_vocab
        print(f"  ✓ tgt [{tgt_lang}]: {len(tgt_vocab)} tokens  ({tgt_path})")

    return vocabs_dict


def build_mammoth_gemma3_arch(
    hf_model_path,
    save_path,
    src_vocab_path,
    src_lang,
    tgt_vocab_paths,
):
    """
    Build a Mammoth model with Gemma3 architecture (random init) and save it.

    Args:
        hf_model_path: HF model name or local path; used ONLY for its config
                       (hidden_size, num_hidden_layers, head_dim, rope_theta, ...).
        save_path: Path to save the initialized Mammoth model.
        src_vocab_path: Path to a Mammoth-native source vocab file.
        src_lang: Source language code.
        tgt_vocab_paths: Dict {lang: mammoth_vocab_path} for target vocabs.
    """
    tgt_langs = list(tgt_vocab_paths.keys())

    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else "."
    os.makedirs(save_dir, exist_ok=True)

    print(f"Building Mammoth (Gemma3 arch, random init) from: {hf_model_path}")
    print(f"Save path: {save_path}")
    print(
        f"Multi-task setup: 1 shared decoder + {len(tgt_langs)} "
        f"language-specific encoder(s) ({', '.join(tgt_langs)})"
    )
    print("=" * 70)

    # Stage 1: vocabularies (Mammoth-native, no HF conversion)
    print("\n[Stage 1] Loading Mammoth-native vocabularies")
    vocabs_dict = build_vocabs_dict(src_vocab_path, src_lang, tgt_vocab_paths)

    # Stage 2: build the Mammoth model (random init, Gemma3-shape decoder)
    print("\n[Stage 2] Building Mammoth model (random init)")
    model_opts = create_model_opts(hf_model_path)

    # dec_layers is needed to build the TaskQueueManager; fetch it from opts.
    num_decoder_layers = model_opts.dec_layers[0]

    task_queue_manager = create_task_queue_manager(
        vocabs_dict,
        num_decoder_layers,
        src_lang=src_lang,
        tgt_langs=tgt_langs,
    )
    opts = create_opts()

    mammoth_model = build_model(
        model_opts=model_opts,
        opts=opts,
        vocabs_dict=vocabs_dict,
        task_queue_manager=task_queue_manager,
        single_task=None,
    )

    mammoth_keys_path = os.path.join(save_dir, "mammoth_model_keys_gemma3_randinit.txt")
    with open(mammoth_keys_path, "w") as f:
        for key, value in mammoth_model.state_dict().items():
            f.write(f"{key}\t{tuple(value.shape)}\n")
    print(
        f"✓ Mammoth model keys saved to {mammoth_keys_path} "
        f"({len(mammoth_model.state_dict())} keys)"
    )

    # Stage 3: save the model
    print("\n[Stage 3] Saving Mammoth model")
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

    # Summary
    total_params, trainable_params = calculate_model_parameters(mammoth_model)
    model_size_gb = total_params * 2 / (1024 ** 3)  # bf16 = 2 bytes/param
    print("\n[Model Size Analysis]")
    print(
        f"  Total parameters: {format_parameter_count(total_params)} "
        f"({total_params:,})"
    )
    print(
        f"  Trainable parameters: {format_parameter_count(trainable_params)} "
        f"({trainable_params:,})"
    )
    print(f"  Estimated model size (bf16): ~{model_size_gb:.2f} GB")

    print("=" * 50)
    print(f"✓ Randomly initialized Mammoth model saved to: {save_path}")
    print(
        f"  Architecture matches HF Gemma3 "
        f"(dec_layers={num_decoder_layers}, "
        f"dec_hidden={model_opts.dec_model_dim}) — no HF weights copied, "
        f"Mammoth-native vocabs used directly."
    )
    return mammoth_model


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build a Mammoth multi-task model with HF Gemma3 architecture "
            "(randomly initialized; no weight mapping; Mammoth-native vocabs)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "hf_model_path",
        help="HuggingFace model name or local path (used for config only).",
    )
    parser.add_argument(
        "save_path",
        help="Path to save the initialized Mammoth model.",
    )
    parser.add_argument(
        "--src-vocab",
        nargs=2,
        required=True,
        metavar=("LANG", "PATH"),
        help="Source vocab: LANG PATH (.txt or .json, Mammoth-native).",
    )
    parser.add_argument(
        "--tgt-vocab",
        nargs=2,
        action="append",
        required=True,
        metavar=("LANG", "PATH"),
        help="Target vocab: LANG PATH (.txt or .json). May be repeated.",
    )

    args = parser.parse_args()

    save_dir = os.path.dirname(args.save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    src_lang, src_vocab_path = args.src_vocab
    tgt_vocab_paths = {lang: path for lang, path in args.tgt_vocab}

    try:
        build_mammoth_gemma3_arch(
            hf_model_path=args.hf_model_path,
            save_path=args.save_path,
            src_vocab_path=src_vocab_path,
            src_lang=src_lang,
            tgt_vocab_paths=tgt_vocab_paths,
        )
    except Exception as e:
        print(f"Error while building Mammoth model: {e}")
        import traceback

        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())