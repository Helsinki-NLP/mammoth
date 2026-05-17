#!/usr/bin/env python3
"""
Theoretical memory calculator for Mammoth training.

Estimates GPU memory requirements from a Mammoth YAML config WITHOUT
instantiating a model or running any training. Given your GPU memory budget,
it recommends a safe maximum batch size.

Usage:
    python mammoth/scripts/report_theoretical_memory.py \\
        --config csc_env/roihu/test.yaml \\
        --gpu_memory_gb 96

The GH200 (Roihu) has 96 GB HBM3.
The MI250X (LUMI) has 64 GB HBM2e per GCD.
"""

import argparse
import sys

try:
    import yaml
except ImportError:
    print("PyYAML not found. Install with: uv pip install pyyaml", file=sys.stderr)
    sys.exit(1)

BYTES_PER_GB = 1024 ** 3


# ---------------------------------------------------------------------------
# Core math functions (tested independently)
# ---------------------------------------------------------------------------

def bytes_per_param(model_dtype: str, optim: str) -> int:
    """
    Total bytes per trainable parameter including optimizer state.

    bf16/fp16 + Adam/AdamW:
      2 (bf16 param) + 4 (fp32 master weight) + 4 (fp32 grad) + 4 (m1) + 4 (m2) = 18
    fp32 + Adam/AdamW:
      4 (param) + 4 (grad) + 4 (m1) + 4 (m2) = 16
    sgd / adagrad / adadelta:
      dtype_bytes × 2  (param + grad only, no moment states)
    """
    low_precision = model_dtype in ('bf16', 'fp16')
    if optim in ('adam', 'adamw'):
        return 18 if low_precision else 16
    elif optim == 'adafactor':
        return 8  # factored second moments — rough lower bound
    else:
        dtype_bytes = 2 if low_precision else 4
        return dtype_bytes * 2


def count_params_per_component(
    model_dim: int, num_layers: int, ff_mult: float, component_type: str
) -> int:
    """
    Number of trainable parameters in one encoder or decoder component.

    Encoder layer  = self-attention (QKVO, 4×d²) + FFN (2×d×d×ff_mult) + 2 LayerNorms
    Decoder layer  = self-attn (4×d²) + cross-attn (4×d²) + FFN + 3 LayerNorms
    Plus one final LayerNorm per component.

    Biases are omitted (x-transformers uses bias=False for projections by default).
    """
    d = model_dim
    ff = int(d * ff_mult)

    if component_type == 'encoder':
        attn = 4 * d * d           # Q, K, V, O
        ffn = 2 * d * ff           # W1, W2
        ln = 2 * 2 * d             # 2 LayerNorms × (weight + bias)
    else:  # decoder
        attn = 4 * d * d           # self-attn QKVO
        cross = 4 * d * d          # cross-attn QKVO
        ffn = 2 * d * ff
        ln = 3 * 2 * d             # 3 LayerNorms
        attn = attn + cross

    params_per_layer = attn + ffn + ln
    final_ln = 2 * d
    return num_layers * params_per_layer + final_ln


def compute_activation_per_token(
    model_dim: int,
    num_enc_layers: int,
    num_dec_layers: int,
    ff_mult: float,
    use_flash: bool,
    seq_len: int,
    heads: int,
    dtype_bytes: int,
    vocab_size: int = 0,
) -> dict:
    """
    Activation memory per token stored during a forward pass.

    Formula (derived from Megatron-LM / "Reducing Activation Recomputation" paper):

    Encoder layer per token:
      = d × (18 + 4×ff_mult) × dtype_bytes     [with flash attention]
      + heads × seq_len × dtype_bytes           [attention weight matrix, without flash only]

    The "18 + 4×ff_mult" breaks down as bytes per (seq×batch×hidden) element:
      - 2 LayerNorm inputs/outputs: 4
      - Q, K, V projections: 6
      - Attention output (pre O-proj): 2
      - Post-attention tensor: 2
      - FFN input + output: 4
      - FFN intermediate (W1 activation output): 4×ff_mult
      Total: 18 + 4×ff_mult

    Decoder layer adds cross-attention (no flash for cross-attn, or flash — same saving):
      Extra tensors: K, V projected from encoder + Q from decoder + LN input + cross-attn output
      ≈ +10 bytes per (seq×batch×hidden) element → use 28 + 4×ff_mult

    vocab_size: when non-zero, adds logits + log_softmax activation.
      CrossEntropyLoss materializes both: logits (B×T×V) and log_softmax output (B×T×V).
      That is 2 × vocab_size × dtype_bytes per token.
    """
    d = model_dim

    enc_hidden = dtype_bytes * d * (18 + 4 * ff_mult)
    attn_matrix = 0 if use_flash else dtype_bytes * heads * seq_len
    enc_per_layer = enc_hidden + attn_matrix

    # Cross-attention adds ~10 extra hidden-state tensors per dec token
    dec_per_layer = enc_per_layer + dtype_bytes * d * 10

    enc_total = num_enc_layers * enc_per_layer
    dec_total = num_dec_layers * dec_per_layer

    # logits tensor + log_softmax intermediate in CrossEntropyLoss
    logits_per_token = 2 * vocab_size * dtype_bytes

    return {
        'enc_per_token_per_layer': enc_per_layer,
        'dec_per_token_per_layer': dec_per_layer,
        'enc_total': enc_total,
        'dec_total': dec_total,
        'logits_per_token': logits_per_token,
        'total': enc_total + dec_total + logits_per_token,
    }


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _infer_vocab_size_from_path(path: str) -> int | None:
    """Try to read vocab size from a Hugging Face tokenizer.json file."""
    try:
        import json
        with open(path) as f:
            tok = json.load(f)
        vocab = tok.get('model', {}).get('vocab') or tok.get('added_tokens', [])
        if isinstance(vocab, dict):
            return len(vocab)
    except Exception:
        pass
    return None


def parse_mammoth_config(config_path: str) -> dict:
    """
    Load a Mammoth YAML config and extract all parameters needed for
    memory estimation.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    xtf = cfg.get('x_transformers_opts', {})

    model_dim = cfg.get('model_dim', -1)
    enc_model_dim = cfg.get('enc_model_dim', model_dim)
    dec_model_dim = cfg.get('dec_model_dim', model_dim)

    enc_layers = cfg.get('enc_layers', [6])
    dec_layers = cfg.get('dec_layers', [6])

    heads = xtf.get('heads', cfg.get('heads', 8))
    ff_mult = float(xtf.get('ff_mult', cfg.get('ff_mult', 4.0)))
    use_flash = bool(xtf.get('attn_flash', False))
    tie_embedding = bool(xtf.get('tie_embedding', True))

    model_dtype = cfg.get('model_dtype', 'fp32')
    optim = cfg.get('optim', 'sgd')

    max_length = cfg.get('max_length', 256)
    batch_size = cfg.get('batch_size', 4096)
    batch_type = cfg.get('batch_type', 'sents')

    accum_raw = cfg.get('accum_count', [1])
    accum_count = accum_raw[0] if isinstance(accum_raw, list) else int(accum_raw)

    src_vocab = cfg.get('src_vocab', {})
    tgt_vocab = cfg.get('tgt_vocab', {})

    # Per-language vocab sizes: try reading tokenizer files, fall back to None
    src_vocab_sizes = {}
    for lang, path in src_vocab.items():
        size = _infer_vocab_size_from_path(path)
        src_vocab_sizes[lang] = size

    tgt_vocab_sizes = {}
    for lang, path in tgt_vocab.items():
        size = _infer_vocab_size_from_path(path)
        tgt_vocab_sizes[lang] = size

    # Explicit overrides from config
    if cfg.get('src_vocab_size'):
        for lang in src_vocab_sizes:
            src_vocab_sizes[lang] = cfg['src_vocab_size']
    if cfg.get('tgt_vocab_size'):
        for lang in tgt_vocab_sizes:
            tgt_vocab_sizes[lang] = cfg['tgt_vocab_size']

    # GPU task distribution
    tasks = cfg.get('tasks', {})
    gpu_map = _analyze_gpu_distribution(tasks)

    return {
        'model_dim': model_dim,
        'enc_model_dim': enc_model_dim,
        'dec_model_dim': dec_model_dim,
        'enc_layers': enc_layers,
        'dec_layers': dec_layers,
        'heads': heads,
        'ff_mult': ff_mult,
        'use_flash': use_flash,
        'tie_embedding': tie_embedding,
        'model_dtype': model_dtype,
        'optim': optim,
        'max_length': max_length,
        'batch_size': batch_size,
        'batch_type': batch_type,
        'accum_count': accum_count,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'src_vocab_sizes': src_vocab_sizes,
        'tgt_vocab_sizes': tgt_vocab_sizes,
        'world_size': cfg.get('world_size', 1),
        'n_nodes': cfg.get('n_nodes', 1),
        'gpu_map': gpu_map,
    }


def _analyze_gpu_distribution(tasks: dict) -> dict:
    """
    Parse task definitions to find which encoder/decoder components and
    vocabulary embeddings live on each GPU.

    Returns:
        { "node:gpu" -> { enc_groups: set, dec_groups: set,
                          src_langs: set, tgt_langs: set } }
    """
    gpu_map: dict = {}
    for task_name, task in tasks.items():
        node_gpu = str(task.get('node_gpu', '0:0'))
        enc_groups = task.get('enc_sharing_group', [])
        dec_groups = task.get('dec_sharing_group', [])
        src_tgt = task.get('src_tgt', 'unk-unk')
        src_lang, tgt_lang = src_tgt.split('-', 1)

        if node_gpu not in gpu_map:
            gpu_map[node_gpu] = {
                'enc_groups': set(),
                'dec_groups': set(),
                'src_langs': set(),
                'tgt_langs': set(),
            }
        gpu_map[node_gpu]['enc_groups'].update(enc_groups)
        gpu_map[node_gpu]['dec_groups'].update(dec_groups)
        gpu_map[node_gpu]['src_langs'].add(src_lang)
        gpu_map[node_gpu]['tgt_langs'].add(tgt_lang)

    return gpu_map


# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------

def report_theoretical_memory(
    config_path: str,
    gpu_memory_gb: float,
    vocab_size_override: int | None = None,
    headroom: float = 0.80,
    allocator_overhead: float = 1.50,
) -> None:
    """
    Print a full memory breakdown and batch-size recommendation table.

    Args:
        config_path:        Path to the Mammoth YAML training config.
        gpu_memory_gb:      Total GPU memory in GB (e.g. 96 for GH200, 64 for MI250X).
        vocab_size_override: Override per-language vocab size (default: infer from config).
        headroom:           Fraction of GPU memory treated as usable (default 0.80).
                            Lower than 1.0 because CUDA/HIP reserves memory for its own use.
        allocator_overhead: Multiplier applied to active-tensor memory to account for
                            PyTorch allocator fragmentation, NCCL/RCCL communication buffers,
                            and DDP gradient buckets (default 1.50, empirically calibrated
                            on LUMI MI250X: actual peak ≈ 1.50× the formula-computed active).
    """
    cfg = parse_mammoth_config(config_path)

    d = cfg['model_dim']
    if d < 0:
        d = max(cfg['enc_model_dim'], cfg['dec_model_dim'])

    enc_layers = cfg['enc_layers']
    dec_layers = cfg['dec_layers']
    num_enc_layers = sum(enc_layers)
    num_dec_layers = sum(dec_layers)

    ff_mult = cfg['ff_mult']
    heads = cfg['heads']
    use_flash = cfg['use_flash']
    tie_embedding = cfg['tie_embedding']
    model_dtype = cfg['model_dtype']
    optim = cfg['optim']
    max_length = cfg['max_length']
    batch_size = cfg['batch_size']
    batch_type = cfg['batch_type']
    accum_count = cfg['accum_count']

    dtype_bytes = 2 if model_dtype in ('bf16', 'fp16') else 4
    bpp = bytes_per_param(model_dtype, optim)

    # Vocab sizes
    if vocab_size_override:
        vocab_size = vocab_size_override
    else:
        # Collect all inferred sizes; fall back to 32000
        all_sizes = list(cfg['src_vocab_sizes'].values()) + list(cfg['tgt_vocab_sizes'].values())
        known = [s for s in all_sizes if s is not None]
        vocab_size = max(known) if known else 32000

    n_src_langs = len(cfg['src_vocab']) or 1
    n_tgt_langs = len(cfg['tgt_vocab']) or 1

    # Worst-case GPU: maximum components any single GPU holds
    gpu_map = cfg['gpu_map']
    if gpu_map:
        max_enc_comp = max(len(v['enc_groups']) for v in gpu_map.values())
        max_dec_comp = max(len(v['dec_groups']) for v in gpu_map.values())
        max_src_emb = max(len(v['src_langs']) for v in gpu_map.values())
        max_tgt_emb = max(len(v['tgt_langs']) for v in gpu_map.values())
    else:
        max_enc_comp = max_dec_comp = max_src_emb = max_tgt_emb = 1

    # Parameter counts
    enc_comp_params = count_params_per_component(d, num_enc_layers, ff_mult, 'encoder')
    dec_comp_params = count_params_per_component(d, num_dec_layers, ff_mult, 'decoder')
    # When tie_embedding=False, each decoder component has a separate to_logits = LinearNoBias(d, V)
    to_logits_params = 0 if tie_embedding else vocab_size * d
    dec_comp_params += to_logits_params
    src_emb_params = vocab_size * d
    tgt_emb_params = vocab_size * d

    gpu_params = (
        max_enc_comp * enc_comp_params
        + max_dec_comp * dec_comp_params
        + max_src_emb * src_emb_params
        + max_tgt_emb * tgt_emb_params
    )

    weight_opt_bytes = gpu_params * bpp

    # Activation memory — vocab_size enables logits + log_softmax accounting
    act = compute_activation_per_token(
        d, num_enc_layers, num_dec_layers, ff_mult, use_flash, max_length, heads, dtype_bytes,
        vocab_size=vocab_size,
    )
    act_per_token = act['total']

    # batch_size IS the per-forward-pass token count in Mammoth.
    # accum_count only determines how many forward passes before one optimizer step.
    # It does NOT split the batch — activation memory is determined by batch_size directly.
    if batch_type == 'tokens':
        micro_batch_tokens = batch_size
        micro_batch_seqs = micro_batch_tokens // max_length
    else:  # 'sents'
        micro_batch_seqs = batch_size
        micro_batch_tokens = micro_batch_seqs * max_length

    # Budget
    gpu_budget_bytes = gpu_memory_gb * BYTES_PER_GB
    usable_bytes = gpu_budget_bytes * headroom
    avail_for_act = usable_bytes - weight_opt_bytes

    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    W = 62

    def hdr(title=''):
        if title:
            print(f"\n{'─' * W}\n  {title}\n{'─' * W}")
        else:
            print('=' * W)

    print()
    hdr()
    print(f"{'  Mammoth Theoretical Memory Report':^{W}}")
    hdr()

    hdr("Model Architecture")
    print(f"  model_dim        : {d}")
    print(f"  encoder layers   : {num_enc_layers}  ({len(enc_layers)} stack(s): {enc_layers})")
    print(f"  decoder layers   : {num_dec_layers}  ({len(dec_layers)} stack(s): {dec_layers})")
    print(f"  attention heads  : {heads}  (head_dim = {d // heads})")
    print(f"  FFN multiplier   : {ff_mult}  →  FFN hidden = {int(d * ff_mult)}")
    print(f"  flash attention  : {'yes  (attn weights NOT stored)' if use_flash else 'no   (attn weights stored)'}")
    print(f"  tie_embedding    : {'yes' if tie_embedding else 'no   (separate to_logits per decoder)'}")
    print(f"  dtype            : {model_dtype}  ({dtype_bytes} bytes/activation element)")
    print(f"  optimizer        : {optim}  ({bpp} bytes/param incl. state)")

    hdr("Task Distribution")
    print(f"  world_size       : {cfg['world_size']}  ({cfg['n_nodes']} node(s), "
          f"{cfg['world_size'] // max(cfg['n_nodes'], 1)} GPU(s)/node)")
    print(f"  unique src langs : {n_src_langs}")
    print(f"  unique tgt langs : {n_tgt_langs}")
    print(f"  GPUs in config   : {len(gpu_map)}")
    if gpu_map:
        print(f"  max enc comp/GPU : {max_enc_comp}")
        print(f"  max dec comp/GPU : {max_dec_comp}")

    hdr("Parameter Counts")
    # enc_comp_params is without to_logits; dec_comp_params already includes to_logits
    dec_base_params = dec_comp_params - to_logits_params
    print(f"  1 encoder ({num_enc_layers} layers)  : {(enc_comp_params) / 1e6:7.1f} M params")
    print(f"  1 decoder ({num_dec_layers} layers)  : {dec_base_params / 1e6:7.1f} M params")
    if to_logits_params:
        print(f"    + to_logits (V={vocab_size}×d) : {to_logits_params / 1e6:7.1f} M params")
        print(f"    decoder total              : {dec_comp_params / 1e6:7.1f} M params")
    print(f"  1 src embedding (V={vocab_size}) : {src_emb_params / 1e6:7.1f} M params")
    print(f"  1 tgt embedding (V={vocab_size}) : {tgt_emb_params / 1e6:7.1f} M params")
    print()
    print(f"  Worst-case GPU ({max_enc_comp} enc + {max_dec_comp} dec + "
          f"{max_src_emb} src emb + {max_tgt_emb} tgt emb):")
    print(f"    Total params   : {gpu_params / 1e6:.1f} M")
    print(f"    Weight + opt   : {weight_opt_bytes / BYTES_PER_GB:.2f} GB  "
          f"({bpp} bytes × {gpu_params / 1e6:.1f} M)")

    hdr("Activation Memory  (per token, all layers, no grad checkpointing)")
    print(f"  {'Layer type':<32}  {'B/token/layer':>14}  {'KB (all layers)':>16}")
    print(f"  {'─'*32}  {'─'*14}  {'─'*16}")
    enc_label = f"Encoder ({num_enc_layers} layers)"
    dec_label = f"Decoder ({num_dec_layers} layers, +cross-attn)"
    print(f"  {enc_label:<32}  {int(act['enc_per_token_per_layer']):>14,}  "
          f"{act['enc_total'] / 1024:>13.0f}")
    print(f"  {dec_label:<32}  {int(act['dec_per_token_per_layer']):>14,}  "
          f"{act['dec_total'] / 1024:>13.0f}")
    if act['logits_per_token']:
        logits_label = f"Logits+log_softmax (V={vocab_size})"
        print(f"  {logits_label:<32}  {act['logits_per_token']:>14,}  "
              f"{'(flat, not per-layer)':>16}")
    print(f"  {'─'*32}  {'─'*14}  {'─'*16}")
    print(f"  {'Total':<32}  {'':>14}  {act_per_token / 1024:>13.0f} KB/token")

    # "Safe budget" factors in allocator overhead on top of the headroom ceiling.
    # On HIP/ROCm (LUMI MI250X) we consistently see ~13-15% of active-tensor memory
    # end up in the "reserved but unallocated" pool due to fragmentation.
    # total_memory_used ≈ (weight_opt + activation) × allocator_overhead
    # So the safe activation ceiling is: usable_bytes / allocator_overhead - weight_opt
    safe_budget_bytes = usable_bytes / allocator_overhead
    avail_for_act_safe = safe_budget_bytes - weight_opt_bytes

    hdr("GPU Memory Budget")
    print(f"  GPU total            : {gpu_memory_gb:.0f} GB")
    print(f"  Usable ({headroom*100:.0f}% headroom) : {usable_bytes / BYTES_PER_GB:.1f} GB")
    print(f"  Allocator overhead   : ×{allocator_overhead:.2f}  "
          f"(fragmentation + NCCL/RCCL buffers + DDP buckets, empirical on LUMI)")
    print(f"  Safe active budget   : {safe_budget_bytes / BYTES_PER_GB:.1f} GB")
    print(f"  Weight + opt         : {weight_opt_bytes / BYTES_PER_GB:.2f} GB")
    print(f"  Available for acts   : {max(0.0, avail_for_act_safe) / BYTES_PER_GB:.2f} GB")

    if avail_for_act_safe <= 0:
        print()
        print("  *** WARNING: weights + optimizer ALREADY EXCEED safe budget! ***")
        print(f"  Need at least {weight_opt_bytes / BYTES_PER_GB:.2f} GB just for model weights.")
        return

    hdr("Batch Size Table  (per-forward-pass, seq_length=" + str(max_length) + ")")
    print(f"  {'batch_size':>12}  {'Seqs':>6}  {'Act (GB)':>10}  {'Peak* (GB)':>11}  Status")
    print(f"  {'─'*12}  {'─'*6}  {'─'*10}  {'─'*11}  ──────")

    for n_tokens in range(5000, 105000, 3000):
        n_seqs = n_tokens // max_length
        act_bytes = act_per_token * n_tokens
        peak_bytes = (weight_opt_bytes + act_bytes) * allocator_overhead
        fits = peak_bytes <= usable_bytes
        status = "likely OK" if fits else "may OOM"
        print(f"  {n_tokens:>12,}  {n_seqs:>6}  {act_bytes / BYTES_PER_GB:>10.2f}  "
              f"{peak_bytes / BYTES_PER_GB:>11.2f}  {status}")
        if not fits:
            break

    max_tokens = int(avail_for_act_safe / act_per_token)
    max_seqs = max_tokens // max_length

    print()
    print(f"  * Peak = (weight+opt + activation) × {allocator_overhead:.2f} overhead")
    print()
    print(f"  Suggested batch_size : ~{max_tokens:,} tokens  ({max_seqs} seqs × {max_length})  [rough estimate]")
    eff_grad_batch = max_tokens * accum_count * cfg['world_size']
    print(f"  Est. grad batch      : ~{eff_grad_batch:,} tokens"
          f"  (× accum {accum_count} × {cfg['world_size']} GPUs)")

    hdr("Config Validation")
    config_act = act_per_token * micro_batch_tokens
    config_peak = (weight_opt_bytes + config_act) * allocator_overhead
    fits_flag = "likely OK" if config_peak <= usable_bytes else "may OOM"
    print(f"  batch_size={batch_size} {batch_type}  (per forward pass per GPU)")
    print(f"  accum_count={accum_count}  (gradient accumulation steps, no memory impact)")
    print(f"  → forward-pass tokens : {micro_batch_tokens:,}  ({micro_batch_seqs} seqs)")
    print(f"  Active memory (est.)  : {(weight_opt_bytes + config_act) / BYTES_PER_GB:.2f} GB")
    print(f"  Peak incl. overhead   : {config_peak / BYTES_PER_GB:.2f} GB  →  {fits_flag}")
    utilization = config_peak / (gpu_memory_gb * BYTES_PER_GB) * 100
    print(f"  GPU utilization       : {utilization:.1f}%  "
          f"({'comfortable' if utilization < 70 else 'tight' if utilization < 85 else 'very tight'})")

    print()
    print("  Tip: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce")
    print("  allocator fragmentation and potentially fit slightly larger batches.")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        'config',
        help='Path to the Mammoth training YAML config file.',
    )
    parser.add_argument(
        'gpu_memory_gb', type=float,
        help='Total GPU memory in GB (e.g. 96 for GH200, 64 for MI250X GCD).',
    )
    parser.add_argument(
        '--vocab_size', type=int, default=None,
        help='Override per-language vocabulary size. '
             'By default, inferred from the config (falls back to 32000).',
    )
    parser.add_argument(
        '--headroom', type=float, default=0.80,
        help='Fraction of GPU memory treated as usable (default: 0.80).',
    )
    parser.add_argument(
        '--allocator_overhead', type=float, default=1.50,
        help='Multiplier for unmodeled memory overhead: PyTorch allocator fragmentation, '
             'NCCL/RCCL communication buffers, DDP gradient buckets (default: 1.50). '
             'Empirically calibrated on LUMI MI250X: actual OOM matched at 1.50×.',
    )
    args = parser.parse_args()

    report_theoretical_memory(
        config_path=args.config,
        gpu_memory_gb=args.gpu_memory_gb,
        vocab_size_override=args.vocab_size,
        headroom=args.headroom,
        allocator_overhead=args.allocator_overhead,
    )


if __name__ == '__main__':
    main()
