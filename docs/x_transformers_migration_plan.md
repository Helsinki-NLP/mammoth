# Mammoth → Native PyTorch Migration Plan

Replacing the vendored **x-transformers** backend with native `torch.nn`
implementations. The goal is a clean cut — x-transformers is dropped outright,
no compatibility shims or gradual toggle flags.

---

## How x-transformers is wired in today

Mammoth vendors the **entire** x-transformers library (`mammoth/x_transformers/`,
~30 files; `x_transformers.py` alone is 3,917 lines), but uses only a thin slice
of it. The main model is assembled in `model_builder.py` from x-transformers
primitives, then wrapped in Mammoth-specific glue for **multi-task / distributed
sharding** (`StackXcoder`, `DistributedComponent`).

Key insight: **the attention bridge and several `modules/*.py` files
(`multi_headed_attn.py`, `transformer_encoder.py`, `position_ffn.py`) are already
native `torch.nn`.** Mammoth already contains a native transformer
implementation — it just isn't used for the primary stacks.

---

## 1. Components directly dependent on x-transformers

| Component | File | Nature of coupling | Severity |
|---|---|---|---|
| **Model builder** | `model_builder.py:11-19` | Imports `TransformerWrapper`, `TokenEmbedding`, `ScaledTokenEmbedding`, `AbsolutePositionalEmbedding`, `ScaledSinusoidalEmbedding`, `LayerNorm`, `LinearNoBias`; builds every model | High (glue-level) |
| **Layer stack** | `modules/layer_stack.py:3-4` | `StackXcoder` wraps `TransformerWrapper`, `LayerIntermediates`, `TokenEmbedding` | High |
| **Forward path** | `models/model.py:61-95` | Calls forward with `return_embeddings`, `context`, `context_mask`, `return_logits_and_embeddings` | Medium (interface) |
| **Beam search KV cache** | `translate/decode_strategy.py:4-5,329` | Imports `Intermediates` / `LayerIntermediates`, rebuilds `cached_kv` tuples to reorder beams | High (cache format) |
| **Translator** | `translate/translator.py:856,941-946` | Same forward surface + `cache=` | Medium |
| **Distributed sharding** | `distributed/components.py:94-128` | Shards on encoder/decoder blocks via `get_sub_modules()` + x-transformers parameter names in state dicts | High (checkpoint keys) |
| **FLOPs** | `utils/flops.py:33-35` | Reads `ff_mult` / `ff_glu` from `x_transformers_opts` | Low |

---

## 2. Minimum feature set for native implementation

The current training config uses exactly these x-transformers options:

```yaml
x_transformers_opts:
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

**Minimum required features** (the only things that need to work on day 1):

- **Embedding layer**: token embedding only (no absolute positional embedding),
  optional tied output projection, `to_logits` (linear, no bias)
- **Post-embedding norm**: RMSNorm without bias applied after embeddings
- **Rotary positional embeddings**: injected into attention, no absolute pos emb
- **Attention block**: pre-norm, RMSNorm without bias, multi-head self-attention
  (`heads=16`) with `F.scaled_dot_product_attention`, attention dropout,
  cross-attention for decoder
- **Feed-forward block**: standard FFN, FF dropout
- **KV cache**: `cached_kv = [(k, v), ...]` per layer, compatible with beam reordering
- **Stacked encoder/decoder**: preserves the ability to share encoder or decoder
  component instances across tasks

**Add later** (when a config actually requires it):

- GQA/MQA, qk-norm, GLU feed-forward, sliding-window attention, sandwich-norm,
  layer dropout, residual scaling, absolute positional embeddings, `scaled_embeddings`,
  `norm_add_unit_offset`

**Delete outright** (~25 unused wrapper files): `belief_state_wrapper`, `gpt_vae`,
`xval`, `dpo`, `nonautoregressive_wrapper`, `neo_mlp`, `continuous`,
`entropy_based_tokenizer`, `xl_autoregressive_wrapper`, `up_wrapper`, `multi_input`.

---

## 3. What migration requires

1. **A native building-block library** (`mammoth/modules/transformer/`):
   `RMSNorm`/`LayerNorm`, `Attention` (SDPA), `FFN`/`GLUFFn`, an encoder block,
   a decoder block, and a `TransformerWrapper` equivalent.
2. **A KV-cache type** replacing `LayerIntermediates`/`Intermediates` with a simple
   `cached_kv = [(k, v), ...]` list so beam reordering in `decode_strategy.py`
   is a drop-in.
3. **Parameter-name compatibility or a remap.** Distributed sharding (`components.py`)
   and existing checkpoints key on x-transformers names. Ship a deterministic
   state-dict remap rather than reproducing the old naming scheme.
4. **Preserve the stacked encoder/decoder architecture.** `StackXcoder` must survive
   as a concept: multiple tasks can share encoder or decoder component instances.
   This is critical for HF conversion — a shared component in Mammoth should map
   to a single shared module in the converted model, not duplicated per task.
5. **Re-target FLOPs** (`utils/flops.py`) to the native interface.

---

## 4. Architectural risks

- **Numerical parity.** Subtle details break training continuity: GLU gate activation
  (GELU vs SiLU), RMSNorm `unit_offset`, sandwich-norm ordering, weight-init scheme.
  Validate native logits against golden captures before switching.
- **Checkpoint breakage.** A rewrite renames parameters → existing trained models
  break. Must ship a tested state-dict remap.
- **KV-cache reordering in beam search.** `decode_strategy.py:329` reconstructs
  cache per step; layout must match exactly.
- **Distributed all-reduce keys on parameter identity** within blocks — changing
  module nesting can scramble component grouping and gradient sync.
- **Mixed-dtype init** via `torch.set_default_dtype` (`model_builder.py:592`) —
  native modules must honor it.

---

## 5. Step-by-step migration plan

### Phase 0 — Lock behavior (no model code change)
Build golden tests: for representative configs capture forward logits, KV-cache
contents, and a single train-step loss. These become the parity oracle. *Nothing
proceeds without them.*

### Phase 1 — Implement native blocks
Build the minimum `torch.nn` primitives (Section 2). Use `F.scaled_dot_product_attention`
for attention. Write the native `TransformerWrapper`, encoder block, and decoder block.
Validate native forward logits against golden captures within tolerance.

### Phase 2 — Swap model builder
Point `model_builder.py`, `layer_stack.py`, and `model.py` at the native blocks.
Preserve the `StackXcoder` abstraction with shared-component semantics intact.
Run golden tests end-to-end.

### Phase 3 — Port the KV cache
Swap `decode_strategy.py` and `translator.py` to the native cache type; verify
beam-search output parity.

### Phase 4 — Verify distributed sharding
Confirm `distributed/components.py` still groups and all-reduces components correctly
under the new module hierarchy. Ship the state-dict remap for checkpoint loading.

### Phase 5 — Delete x-transformers
Remove vendored `mammoth/x_transformers/` and unused wrapper files. Update FLOPs.
Codebase shrinks by ~6 k lines.

---

## 6. Shared encoder/decoder architecture goal

The stacked encoder/decoder design must be preserved through the migration because it
is the key to correct HF conversion.

**Current problem:** `convert_mammoth_to_hf.py` cannot represent a model where tasks
share a component — it serializes each task as a separate independent model, duplicating
any shared encoder or decoder.

**Target:** a Mammoth model where tasks A and B share an encoder instance should
produce a single HF model with one encoder, with both task heads pointing at it.
The native `StackXcoder` replacement must track which component instances are shared
so that the converter can detect identity and emit a single module.

Implementation note: keeping component instances as named, registered `nn.Module`
children of a shared registry (rather than duplicating them per task) is the
structural change that makes this possible.

---

## Bottom line

The migration is tractable because Mammoth uses only a conventional transformer
subset of x-transformers, and already contains native transformer code. The main
constraints are **(a) numerical parity** for checkpoint fidelity and **(b)
parameter-name compatibility + distributed-sharding contracts**. Starting with a
minimum feature set and expanding iteratively keeps the initial rewrite scoped and
verifiable.
