# Native PyTorch Backend Migration Guide

This document explains how to get started with training and inference using the PyTorch-backend Mammoth.

---

## Background

Mammoth previously shipped the entire x-transformers library (~8,300 lines,
18+ files) inside `mammoth/x_transformers/`. Its abstractions over transformer
components made it difficult to export Mammoth models to LiteRT's `.tflite` format.

The vendored code has been replaced with a small, purpose-built native module
(`mammoth/modules/transformer/`) built entirely from standard `torch.nn`
primitives. The multi-task and distributed infrastructure is unchanged.

---

## CLI / YAML config changes

The new transformer module comes with a more streamlined configuration format.

Most config options are unchanged. There are two notable differences:

1. The `x_transformers_opts` block is gone. Its keys are now top-level
   arguments, at the same level as `model_dim` and `enc_layers`.

2. The supported options are more focused:
   - **Positional embedding:** RoPE only. Absolute positional embedding support for legacy models is planned.
   - **Normalization:** RMSNorm only, bias-free.
   - **Feedforward activation:** `swiglu` or `gelu`; defaults to `swiglu` for better convergence.
  
**Old config (no longer valid):**
```yaml
# Model Architecture Options
enc_layers: [6]
dec_layers: [6]

model_dim: 1024
dropout: 0.1
model_dtype: bf16
add_language_tokens: false

# x-transformers specific options
x_transformers_opts:
  attn_flash: true     
  rotary_pos_emb: true     
  tie_embedding: false    
  heads: 16
  pre_norm: true
  post_emb_norm: true
  post_emb_norm_bias: false
  attn_dropout: 0.1
  ff_dropout: 0.1
  layernorm_bias: false
  use_abs_pos_emb: false
  use_fused_rmsnorm: true

```

**New config:**
```yaml
# Model Architecture Options
enc_layers: [6]
dec_layers: [6]

model_dim: 1024
model_dtype: bf16
add_language_tokens: false

# Native pytorch transformer options
heads: 16              # attention heads (model_dim must be divisible by heads)
rotary_pos_emb: true   # rotary positional embeddings (RoPE, currently the only option)
post_emb_norm: true    # RMSNorm applied after token embedding (RMSNorm, the only option)
attn_dropout: 0.1
ff_dropout: 0.1
ff_activation: swiglu  # choose between "swiglu" and "gelu", "swiglu" by default
```
A training template is available at: `/scratch/project_462001087/members/wangchao/train/compare/train_pytorch.yaml`

For inference, since the model architecture is fixed at the training time, you do not need to specify any architecture options in the inference config. An inference recipe is at: `/scratch/project_462001087/members/wangchao/train/compare/inference/inference_pytorch.yaml`


**Keys that no longer exist and should not be used with the PyTorch-backend Mammoth:**

| Old key | Reason |
|---|---|
| `use_abs_pos_emb` | RoPE is currently the only supported positional embedding. Absolute positional embedding support for legacy models is planned. |
| `tie_embedding` | Tied input/output embeddings are not supported |
| `layernorm_bias` | All norms are bias-free by design |
| `use_fused_rmsnorm` | `nn.RMSNorm` automatically uses PyTorch's built-in fused kernel and is the only supported norm |
| `pre_norm` | Pre-norm is always enabled |
| `post_emb_norm_bias` | The post-embedding norm is always bias-free |

## Working environments:

### Roihu:
Mammoth Pytorch-backend on Roihu: `/scratch/project_2017852/mammoth-shared/mammoth_pytorch/mammoth`
Slurm job launch wrapper: `mammoth/csc_env/roihu/single-node.sh` and `mammoth/csc_env/roihu/multi-nodes.sh`
Note: 
Now on Roihu-GPU it's only possible to reserve 217G of memory and 72 CPU cores per GPU.


### LUMI
Mammoth Pytorch-backend on LUMI: `/scratch/project_462001087/shared/mammoth_pytorch/mammoth`
Slurm job launch wrapper: `mammoth/csc_env/lumi/single_node.sh` and `mammoth/csc_env/lumi/multi_nodes.sh`

## Further reading


### What changed

#### New module: `mammoth/modules/transformer/`

A self-contained transformer library, one file per concept:

| File | What it provides |
|---|---|
| `rotary.py` | `RotaryEmbedding` with lazy cos/sin cache; `apply_rotary` |
| `cache.py` | `LayerCache` (dataclass per layer) and `KVCache` (list of `LayerCache`) with `reorder_beams` for beam search |
| `attention.py` | `MultiHeadAttention` — `F.scaled_dot_product_attention`, rotary injection, self-attention and cross-attention, KV-cache append |
| `ffn.py` | `FeedForward` — two bias-free `nn.Linear` layers, GELU, dropout |
| `block.py` | `EncoderBlock` (norm→self_attn→norm→ff) and `DecoderBlock` (adds cross-attn as a third sublayer) |
| `stack.py` | `TransformerStack` — iterates blocks, applies a final RMSNorm, carries `(layer_stack_index, xcoder_id)` identity for distributed sharding |
| `wrapper.py` | `NativeTransformerWrapper` — the per-task top-level module: `token_emb → post_emb_norm → stacks → to_logits` |

All norms are `nn.RMSNorm` (no bias). All linear layers are bias-free.
Attention uses `F.scaled_dot_product_attention`, which selects FlashAttention
or memory-efficient attention automatically on supported hardware.

#### Files rewritten or deleted

| File | Change |
|---|---|
| `mammoth/model_builder.py` | Rewritten — builds `TransformerStack` / `NativeTransformerWrapper` directly; ~480 lines removed |
| `mammoth/modules/layer_stack.py` | Simplified — `StackXcoder` now holds `NativeTransformerWrapper` objects; `AdaptedAttentionLayersStack` deleted |
| `mammoth/translate/decode_strategy.py` | KV cache ported — `LayerIntermediates` replaced with `KVCache`; beam reordering calls `cache.reorder_beams()` |
| `mammoth/translate/translator.py` | Updated to pass `KVCache` through the forward path |
| `mammoth/x_transformers/` | Deleted entirely (~8,300 lines) |

---

### What stayed the same

- **`StackXcoder` / `DistributedComponent` abstraction** — multi-task
  parameter sharing works identically. Tasks still declare `enc_sharing_group`
  and `dec_sharing_group` in the YAML to control which stacks they share.
- **Checkpoint format** — state-dict keys were remapped during the migration.
  Checkpoints trained on the old backend can be loaded after the remap is applied
  (see `docs/x_transformers_migration_plan.md` § Phase 4 for the key mapping).
- **HF conversion** (`mammoth/hf_integration/to_hf/`) — unchanged; targets the
  native modules.
- **LiteRT export** (`mammoth/litert/`) — unchanged; the native module hierarchy
  maps cleanly to the encoder/prefill/decode signature split.
- **Training YAML structure** — everything other than `x_transformers_opts` is
  identical.

---

### Module hierarchy (for orientation)

```
StackXcoder                         (nn.ModuleDict, owns all shared params)
└── NativeTransformerWrapper        (per-task view, no extra param ownership)
    └── TransformerStack(s)         (iterates blocks, carries stack identity)
        └── EncoderBlock / DecoderBlock
            ├── nn.RMSNorm
            ├── MultiHeadAttention  (F.scaled_dot_product_attention + RoPE)
            └── FeedForward         (bias-free linears, GELU)
```

`NativeTransformerWrapper` holds references to shared parameters (stacks,
embeddings, norms, output projection) **without re-registering them** as
`nn.Module` children — `StackXcoder` owns and registers all of them. This keeps
`state_dict()` keys clean and prevents double-counting in distributed all-reduces.
