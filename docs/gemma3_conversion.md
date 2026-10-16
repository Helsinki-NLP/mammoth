# Gemma3-270M → Mammoth conversion (native PyTorch backend)

**Status: complete (2026-09-16), bf16 default added 2026-10-07.** This file is the history and decision log. The short list of lessons that should shape future code lives in `CLAUDE.md`.

**Goal:** convert `hf_models/gemma3_270m` (HF `Gemma3ForCausalLM`, text-only, 270M) into Mammoth using the **native PyTorch backend** (`mammoth/mammoth/modules/transformer/`), not the deprecated x_transformers backend. A prior attempt (`mammoth/mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py`, `gemma_skeleton.py`) targeted x_transformers and was an imperfect conversion — several Gemma3 features weren't actually implemented, just papered over via x_transformers flags.

## Gemma3-270M architecture facts (from `hf_models/gemma3_270m/config.json`)

- 18 layers, hidden=640, heads=4, **kv_heads=1 (MQA)**, head_dim=256 (explicit, ≠ hidden/heads=160)
- **Sandwich norm**: 4 RMSNorms per layer — pre-attn, post-attn (applied to attn output *before* residual add), pre-ff, post-ff (before residual add)
- **QK-norm**: RMSNorm on q/k per-head (dim=head_dim=256) applied before RoPE
- **Dual RoPE**: theta=1e6 on `full_attention` layers, theta=1e4 on `sliding_attention` layers; `layer_types` in config gives the exact per-layer pattern (full attention every 6th layer: indices 5, 11, 17)
- **Sliding window** = 512 tokens on local layers (causal + windowed)
- **GeGLU MLP**: `down_proj(gelu_tanh(gate_proj(x)) * up_proj(x))` — gated like SwiGLU but GELU-tanh instead of SiLU
- Scaled embeddings (× √hidden_size=√640)
- RMSNorm uses **unit-offset** convention: `x_norm * (1 + weight)`, eps=1e-6
- Decoder-only, **no cross-attention**
- `query_pre_attn_scalar=256` — attention softmax scale is `1/sqrt(query_pre_attn_scalar)`, not `1/sqrt(head_dim)` (equal for this checkpoint since both are 256, so a no-op here, but not general)
- `attn_logit_softcapping` / `final_logit_softcapping` = null (not needed for this checkpoint)

## Native backend gaps found (audit as of 2026-09-16) — all since fixed

The native backend (`modules/transformer/{attention,block,ffn,rotary,stack,wrapper}.py`) and `model_builder.py::build_xcoder` were audited against the facts above.

| Feature | Gap at audit time |
|---|---|
| Sandwich norm | `block.py` only did pre-norm, 2 norms/layer, not 4 |
| QK-norm | not in `attention.py` at all |
| GQA/MQA (kv_heads < heads) | `MultiHeadAttention` always used the same head count for q/k/v |
| Dual/per-layer RoPE theta | `RotaryEmbedding` was one instance per component with a single base |
| Sliding-window attention mask | `causal` was just a bool → `is_causal=True` in SDPA |
| GeGLU | `ffn.py` only had `"swiglu"` and ungated `"gelu"` |
| Scaled embeddings | not implemented in `wrapper.py` |
| RMSNorm unit-offset `(1+w)` | not a real gap — loading `mammoth_weight = hf_weight + 1.0` at conversion time is mathematically equivalent |
| Custom attention scale | SDPA call had no `scale=` override |
| Decoder-only (no cross-attn) | **still open** — see decision below |

Also: `opts.py` already declared `--dec_sliding_window`, `--dec_global_attn_every_n_layers`, `--dec_global_rope_theta`, `--dec_local_rope_theta` (and `enc_` equivalents), but `model_builder.py` never read them. They are now wired.

## Decision: fake-encoder path (2026-09-16)

Two options for fitting a decoder-only model into Mammoth's encoder-decoder NMT framework:

1. **True decoder-only path** — extend the native backend with a real decoder-only mode (no cross-attn module built or run), plus a plain-LM task path. Cleaner, matches HF exactly, but requires changing `model_builder.py`/`StackXcoder` task plumbing, which assumes every task has both an encoder and a decoder.
2. **Fake encoder** (like `gemma2mammoth.py` before) — bolt on a small randomly-initialized encoder + unused cross-attention layers to satisfy the NMT shape.

**Chosen: option 2**, to keep the change scoped to the transformer building blocks rather than the task/model-builder plumbing. **Option 1 remains the preferred long-term direction** and should be revisited if Mammoth needs to host more decoder-only LMs — at that point cross-attention should become optional per stack rather than mandatory.

## Implementation approach: tests first

Per-feature unit tests were written in `mammoth/tests/` *before* implementing each feature (sandwich norm, QK-norm, GQA, dual RoPE + sliding window, GeGLU, scaled embeddings, attention scale), then the converter was built on the extended backend.

## What was implemented

- `modules/transformer/masking.py` — `build_sliding_window_causal_mask` (causal AND `query_pos - key_pos < window`, matches `transformers.masking_utils.sliding_window_overlay` exactly)
- `ffn.py` — `activation="geglu"` (gate through `gelu_pytorch_tanh`, up unactivated, matches `Gemma3MLP`)
- `attention.py` — `kv_heads` (GQA/MQA via SDPA `enable_gqa=True`), `qk_norm` (per-head RMSNorm before RoPE), `attn_scale` (overrides the default `1/sqrt(head_dim)`, for `query_pre_attn_scalar`)
- `block.py` — `DecoderBlock` gained `sandwich_norm` (matches `Gemma3DecoderLayer.forward`: input_layernorm → self_attn → post_attention_layernorm → residual; pre_feedforward_layernorm → mlp → post_feedforward_layernorm → residual), `sliding_window` (builds the windowed mask internally, cache-offset aware), `rotary_emb` (block owns its own RotaryEmbedding for dual-theta support), and `norm_eps`
- `wrapper.py` — `embed_scale` (multiplies token embeddings by √hidden_size, matching `Gemma3TextScaledWordEmbedding`)
- `model_builder.py::build_xcoder` — wires all of the above through `dec_*`/`enc_*`-prefixed `model_opts`, with the exact Gemma3 layer-type pattern (`(i+1) % pattern == 0` → full/global attention, from `Gemma3TextConfig.__init__`)
- `hf_integration/from_hf/gemma3/convert_gemma3_native.py` — the converter. Unlike the old `gemma2mammoth.py` (string-key weight mapping — the source of the earlier "imperfect conversion"), it **walks the built nn.Module objects directly on both sides** and copies tensors module-by-module; there is no string-based key table to get subtly wrong. It also zero-initializes every decoder block's `cross_attn.to_out.weight`, so the fake encoder's output contributes exactly zero to the residual stream — the decoder path is mathematically Gemma3 alone, regardless of what the fake encoder produces (verified by feeding it random-int garbage).
- `tests/test_gemma3_native_conversion.py` — loads the real checkpoint, runs HF and converted Mammoth on the same input ids, asserts **exact** logit match (`max abs diff: 0.0`) in fp32.

## Gotcha: RMSNorm eps

torch's `nn.RMSNorm` defaults to `eps=None` (a dtype-based internal default, ≈1.19e-7 for fp32), but Gemma3 uses a fixed `eps=1e-6` throughout (`rms_norm_eps` in config). On activation rows with near-zero variance combined with a large-magnitude norm weight (this genuinely occurs — e.g. layer 5's `post_feedforward_layernorm` weight has channels up to ~14), logits diverged by as much as 59 in absolute value despite every weight and every upstream activation matching exactly.

Every `nn.RMSNorm` in `block.py` (including `attention.py`'s `qk_norm`) and in `model_builder.py` (final_norm, post_emb_norm) now takes an explicit `norm_eps`; the converter sets `dec_norm_eps = config.rms_norm_eps`. Regression test: `TestDecoderBlockNormEps` in `test_native_transformer_gemma3_features.py`.

**Lesson:** when converting any HF model, matching every module's epsilon (and any other "small print" default) is as load-bearing as matching the weights — verify with an exact end-to-end logit comparison against the real checkpoint, not just shape/smoke tests.

## Update log

**2026-09-16 — enc/dec dim mismatch fixed.** `MultiHeadAttention` accepts `context_dim`: when set, cross-attn's `to_k`/`to_v` project from that size instead of the block's own `dim`. `DecoderBlock` forwards it, and `build_xcoder` resolves it as `dec_cross_attn_dim_context` (explicit opt) falling back to `enc_model_dim`, mirroring the old x_transformers-backend option. The converter's fake encoder now defaults to `enc_model_dim=64` (down from 640). Exact logit match still holds. Tests: `test_native_transformer_cross_attn_context_dim.py`.

**2026-09-16 — converter CLI exposes task/group/lang parameters.** `convert()`/`main()` take `--task-id` (default `"<src-lang>-<tgt-lang>"`), `--encoder-group`/`--decoder-group`, `--src-lang`/`--tgt-lang`, `--weight`. Task scheduling is **always** `WeightedSamplingTaskDistributionStrategy` (not round-robin), matching real training. Also found and fixed: `build_model_opts` had `model_dtype = "bf16"`, which made `build_model()` silently cast the freshly-built Mammoth model to bf16 *before* the fp32 HF weights were copied in — `.copy_()` then downcast every value in place. (Superseded by the `dtype` argument below.)

**Lesson:** `model_dtype` affects correctness, not just storage size. Verify it matches the weights you're about to copy in, and re-run the exact-match test after any opts change — a silent downcast raises no error; it just produces slightly-wrong-looking logits that are easy to mistake for floating-point noise.

**2026-10-07 — dtype is a parameter; default is bf16.** `convert()`/`build_model_opts` take a `dtype` argument (`--dtype` on the CLI) applied to *both* the HF reference model and the Mammoth model. **Default is `"bf16"`**, what we use in practice. `test_gemma3_native_conversion.py` passes `dtype="fp32"` explicitly so the exact comparison is meaningful.

**The bf16 path is not covered by the exactness guarantee.** In bf16 the two implementations can differ by rounding (op ordering, fused vs. unfused kernels, bf16 accumulation in norms/softmax), so expect small non-zero logit differences. The fp32 test proves the *conversion logic* (weights, eps, masks, scales); it does not prove bf16 numerics. When validating a bf16 conversion, compare with a tolerance (ideally also check argmax/top-k agreement and generated text), and don't treat a bf16 diff as a conversion bug unless the fp32 test also fails.
