# Gemma3-270M → Mammoth conversion (native PyTorch backend)

**Status: complete (2026-09-16), bf16 default added 2026-10-07.** The first half is the how-to (formerly `docs/LOADING_HF_MODELS.md`); the second half is the history and decision log. The short list of lessons that should shape future code lives in `CLAUDE.md`.

## Usage

This guide explains how to convert a pretrained HuggingFace model into a Mammoth checkpoint, on the **native PyTorch transformer backend** (`mammoth/mammoth/modules/transformer/`).

**Currently supported: Gemma3-270M** (`google/gemma3-270m`-style text-only checkpoints). 

### How it works

Gemma3 is decoder-only, but the native backend's transformer building blocks (sandwich norm, QK-norm, GQA/MQA, dual-theta RoPE + sliding window, GeGLU, scaled embeddings, custom attention scale) were extended to match it exactly. Both converters below copy weights by walking the actual built `nn.Module` objects on both sides (Mammoth's and HF's) and calling `.copy_()` module-by-module — there is no string-keyed weight-name table to get subtly wrong, unlike the old x_transformers-backend converter.

Two converters exist, because Mammoth's task/model-builder plumbing was originally written assuming every task has an encoder and a decoder:

- **`convert_gemma3_native.py`** — bolts on a small, randomly-initialized "fake" encoder to satisfy that plumbing. Its cross-attention output projection (`cross_attn.to_out.weight`) is zero-initialized, so the fake encoder contributes exactly nothing to the decoder's residual stream: the converted model's forward pass is numerically **identical** to running Gemma3 alone (under data type float32), verified by exact logit match in `mammoth/tests/test_gemma3_native_conversion.py`. Use this if you want an encoder-decoder-shaped task (e.g. to later pair the decoder with a real pretrained encoder, or to fit existing NMT-style training/inference code paths without changes).
- **`convert_gemma3_decoder_only.py`** — the more direct conversion: sets `model_opts.decoder_only = True`, which makes `build_model()` skip building an encoder entirely and builds every `DecoderBlock` with `use_cross_attn=False`. No cross-attention module is constructed or run. The converted model is architecturally identical to Gemma3, with no fake encoder to reason about. Its forward pass is numerically **identical** to running original Gemma3-270m (under data type float32). Verified in `mammoth/tests/test_gemma3_decoder_only_conversion.py`.

Both scripts otherwise convert the same decoder architecture:
- RMSNorm with Gemma3's unit-offset convention (`(1 + weight)`), folded into the copied weight so Mammoth's plain `nn.RMSNorm` matches exactly
- Sandwich norm (4 RMSNorms per layer: pre-attn, post-attn, pre-ff, post-ff)
- Q/K normalization
- Grouped/multi-query attention (`num_key_value_heads` < `num_attention_heads`)
- GeGLU MLP (GELU-tanh gated)
- Dual RoPE (separate theta for sliding-window vs. full-attention layers) + sliding-window attention mask
- Scaled word embeddings, bias-free linears throughout
- The exact epsilon from the HF config (`rms_norm_eps`) on every norm — a mismatch here produces large logit divergence despite every weight matching (see "Gotcha: RMSNorm eps" below)

### Commands

Both scripts take the same two positional arguments — an HF model directory and a save path — plus converter-specific flags:

```bash
python mammoth/hf_integration/from_hf/gemma3/convert_gemma3_native.py \
    <hf_model_path> <save_path> \
    [--enc-layers N] [--enc-model-dim N] \
    [--task-id <id>] [--encoder-group <xcoder_id>] [--decoder-group <xcoder_id>] \
    [--src-lang <lang>] [--tgt-lang <lang>] [--weight <float>]

python mammoth/hf_integration/from_hf/gemma3/convert_gemma3_decoder_only.py \
    <hf_model_path> <save_path> \
    [--task-id <id>] [--decoder-group <xcoder_id>] [--lang <lang>] [--weight <float>]
```

All flags are optional:
- `--task-id` defaults to `<src-lang>-<tgt-lang>` (fake-encoder) or `<lang>-lm` (decoder-only)
- `--encoder-group` / `--decoder-group` (enc/dec sharing-group xcoder ids — the keys other tasks would reference to share this stack) default to `fake_enc` / `gemma3_dec`
- `--src-lang` / `--tgt-lang` / `--lang` default to `gemma3` — Gemma3's own tokenizer is used for every language tag; the target-side tag is the meaningful one since it owns the converted decoder weights
- `--enc-layers` (default `1`) / `--enc-model-dim` (default `64`) size the fake encoder (`convert_gemma3_native.py` only) — it can be much smaller than the decoder's hidden size since cross-attn K/V project from the encoder's own dimension
- `--weight` (default `1.0`) sets the task's weight; task scheduling is always `weighted_sampling` (what real training actually uses), regardless of task count

`model_opts.model_dtype` is `"bf16"` in both scripts — the HF model is loaded as bf16 (`dtype=torch.bfloat16`) and the freshly-built Mammoth model is cast to bf16 to match before weights are copied in, so `.copy_()` never silently downcasts a value already in the target tensor.

#### Example: SLURM conversion jobs on LUMI

Fake-encoder conversion (with extra parameters for the encoder):

```bash
#!/bin/bash
#SBATCH -A <your project>
#SBATCH -J train
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

echo "Starting at `date`"
set -e

singularity exec \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" \
    -B /scratch/project_462001509:/scratch/project_462001509:rw \
    /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif \
    /scratch/project_462001509/shared/mammoth-shared/.venv/bin/python path/to/convert_gemma3_native.py \
    --enc-layers 8 \
    --enc-model-dim 640 \
    --task-id eng_spa \
    --encoder-group "eng" \
    --decoder-group "spa" \
    --src-lang eng \
    --tgt-lang spa \
    /path/to/hf_models/gemma3_270m \
    /path/to/converted/model/
echo "Finishing at `date`"
```

True decoder-only conversion (defaults, no extra flags needed):

```bash
#!/bin/bash
#SBATCH -A <your project>
#SBATCH -J train
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1

echo "Starting at `date`"
set -e

singularity exec \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" \
    -B /scratch/project_462001509:/scratch/project_462001509:rw \
    /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif \
    /scratch/project_462001509/shared/mammoth-shared/.venv/bin/python path/to/convert_gemma3_decoder_only.py \
    /path/to/hf_models/gemma3_270m \
    /path/to/converted/model/
echo "Finishing at `date`"
```

Both produce a standard Mammoth checkpoint (frame + per-component shards) at `<save_path>`, loadable like any other Mammoth checkpoint for further fine-tuning or inference.

#### Verifying a conversion

Each converter has a matching comparison script that runs the same input through both the original HF model and the converted Mammoth model:

```bash
# Fake-encoder: in-memory (no checkpoint needed) or against a saved checkpoint
python mammoth/hf_integration/from_hf/gemma3/inference_compare_native.py <hf_model_path>
python mammoth/hf_integration/from_hf/gemma3/inference_compare_native.py <hf_model_path> \
    --mammoth-checkpoint <save_path>

# Decoder-only: always loads the on-disk checkpoint (validates the save/load round-trip too)
python mammoth/hf_integration/from_hf/gemma3/inference_compare.py <hf_model_path> <save_path> \
    --sentences "The quick brown fox" "2 + 2 =" --max-new-tokens 30
```

Pass matching `--task-id`/`--encoder-group`/`--decoder-group`/`--src-lang`/`--tgt-lang`/`--lang` flags if you converted with non-default values — both scripts reuse the converter's own `build_task_queue_manager()`/`build_model_opts()` so the task wiring lines up automatically otherwise.

The automated tests (`mammoth/tests/test_gemma3_native_conversion.py`, `mammoth/tests/test_gemma3_decoder_only_conversion.py`) go further and assert an **exact** logit match against the real checkpoint, not just a visual/greedy-decoding comparison.

## History and decisions

**Goal:** convert `hf_models/gemma3_270m` (HF `Gemma3ForCausalLM`, text-only, 270M) into Mammoth using the **native PyTorch backend** (`mammoth/mammoth/modules/transformer/`), not the deprecated x_transformers backend. A prior attempt (`mammoth/mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py`, `gemma_skeleton.py`) targeted x_transformers and was an imperfect conversion — several Gemma3 features weren't actually implemented, just papered over via x_transformers flags.

### Gemma3-270M architecture facts (from `hf_models/gemma3_270m/config.json`)

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

### Native backend gaps found (audit as of 2026-09-16) — all since fixed

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

### Decision: fake-encoder path (2026-09-16)

Two options for fitting a decoder-only model into Mammoth's encoder-decoder NMT framework:

1. **True decoder-only path** — extend the native backend with a real decoder-only mode (no cross-attn module built or run), plus a plain-LM task path. Cleaner, matches HF exactly, but requires changing `model_builder.py`/`StackXcoder` task plumbing, which assumes every task has both an encoder and a decoder.
2. **Fake encoder** (like `gemma2mammoth.py` before) — bolt on a small randomly-initialized encoder + unused cross-attention layers to satisfy the NMT shape.

**Chosen: option 2**, to keep the change scoped to the transformer building blocks rather than the task/model-builder plumbing. **Option 1 remains the preferred long-term direction** and should be revisited if Mammoth needs to host more decoder-only LMs — at that point cross-attention should become optional per stack rather than mandatory.

*Update 2026-10-07:* option 1 now exists for Gemma3 as `convert_gemma3_decoder_only.py` (`model_opts.decoder_only = True`, no encoder, `use_cross_attn=False`); see Usage above. The fake-encoder converter is kept for encoder-decoder-shaped tasks.

### Implementation approach: tests first

Per-feature unit tests were written in `mammoth/tests/` *before* implementing each feature (sandwich norm, QK-norm, GQA, dual RoPE + sliding window, GeGLU, scaled embeddings, attention scale), then the converter was built on the extended backend.

### What was implemented

- `modules/transformer/masking.py` — `build_sliding_window_causal_mask` (causal AND `query_pos - key_pos < window`, matches `transformers.masking_utils.sliding_window_overlay` exactly)
- `ffn.py` — `activation="geglu"` (gate through `gelu_pytorch_tanh`, up unactivated, matches `Gemma3MLP`)
- `attention.py` — `kv_heads` (GQA/MQA via SDPA `enable_gqa=True`), `qk_norm` (per-head RMSNorm before RoPE), `attn_scale` (overrides the default `1/sqrt(head_dim)`, for `query_pre_attn_scalar`)
- `block.py` — `DecoderBlock` gained `sandwich_norm` (matches `Gemma3DecoderLayer.forward`: input_layernorm → self_attn → post_attention_layernorm → residual; pre_feedforward_layernorm → mlp → post_feedforward_layernorm → residual), `sliding_window` (builds the windowed mask internally, cache-offset aware), `rotary_emb` (block owns its own RotaryEmbedding for dual-theta support), and `norm_eps`
- `wrapper.py` — `embed_scale` (multiplies token embeddings by √hidden_size, matching `Gemma3TextScaledWordEmbedding`)
- `model_builder.py::build_xcoder` — wires all of the above through `dec_*`/`enc_*`-prefixed `model_opts`, with the exact Gemma3 layer-type pattern (`(i+1) % pattern == 0` → full/global attention, from `Gemma3TextConfig.__init__`)
- `hf_integration/from_hf/gemma3/convert_gemma3_native.py` — the converter. Unlike the old `gemma2mammoth.py` (string-key weight mapping — the source of the earlier "imperfect conversion"), it **walks the built nn.Module objects directly on both sides** and copies tensors module-by-module; there is no string-based key table to get subtly wrong. It also zero-initializes every decoder block's `cross_attn.to_out.weight`, so the fake encoder's output contributes exactly zero to the residual stream — the decoder path is mathematically Gemma3 alone, regardless of what the fake encoder produces (verified by feeding it random-int garbage).
- `tests/test_gemma3_native_conversion.py` — loads the real checkpoint, runs HF and converted Mammoth on the same input ids, asserts **exact** logit match (`max abs diff: 0.0`) in fp32.

### Gotcha: RMSNorm eps

torch's `nn.RMSNorm` defaults to `eps=None` (a dtype-based internal default, ≈1.19e-7 for fp32), but Gemma3 uses a fixed `eps=1e-6` throughout (`rms_norm_eps` in config). On activation rows with near-zero variance combined with a large-magnitude norm weight (this genuinely occurs — e.g. layer 5's `post_feedforward_layernorm` weight has channels up to ~14), logits diverged by as much as 59 in absolute value despite every weight and every upstream activation matching exactly.

Every `nn.RMSNorm` in `block.py` (including `attention.py`'s `qk_norm`) and in `model_builder.py` (final_norm, post_emb_norm) now takes an explicit `norm_eps`; the converter sets `dec_norm_eps = config.rms_norm_eps`. Regression test: `TestDecoderBlockNormEps` in `test_native_transformer_gemma3_features.py`.

**Lesson:** when converting any HF model, matching every module's epsilon (and any other "small print" default) is as load-bearing as matching the weights — verify with an exact end-to-end logit comparison against the real checkpoint, not just shape/smoke tests.

### Update log

**2026-09-16 — enc/dec dim mismatch fixed.** `MultiHeadAttention` accepts `context_dim`: when set, cross-attn's `to_k`/`to_v` project from that size instead of the block's own `dim`. `DecoderBlock` forwards it, and `build_xcoder` resolves it as `dec_cross_attn_dim_context` (explicit opt) falling back to `enc_model_dim`, mirroring the old x_transformers-backend option. The converter's fake encoder now defaults to `enc_model_dim=64` (down from 640). Exact logit match still holds. Tests: `test_native_transformer_cross_attn_context_dim.py`.

**2026-09-16 — converter CLI exposes task/group/lang parameters.** `convert()`/`main()` take `--task-id` (default `"<src-lang>-<tgt-lang>"`), `--encoder-group`/`--decoder-group`, `--src-lang`/`--tgt-lang`, `--weight`. Task scheduling is **always** `WeightedSamplingTaskDistributionStrategy` (not round-robin), matching real training. Also found and fixed: `build_model_opts` had `model_dtype = "bf16"`, which made `build_model()` silently cast the freshly-built Mammoth model to bf16 *before* the fp32 HF weights were copied in — `.copy_()` then downcast every value in place. (Superseded by the `dtype` argument below.)

**Lesson:** `model_dtype` affects correctness, not just storage size. Verify it matches the weights you're about to copy in, and re-run the exact-match test after any opts change — a silent downcast raises no error; it just produces slightly-wrong-looking logits that are easy to mistake for floating-point noise.

**2026-10-07 — dtype is a parameter; default is bf16.** `convert()`/`build_model_opts` take a `dtype` argument (`--dtype` on the CLI) applied to *both* the HF reference model and the Mammoth model. **Default is `"bf16"`**, what we use in practice. `test_gemma3_native_conversion.py` passes `dtype="fp32"` explicitly so the exact comparison is meaningful.

**The bf16 path is not covered by the exactness guarantee.** In bf16 the two implementations can differ by rounding (op ordering, fused vs. unfused kernels, bf16 accumulation in norms/softmax), so expect small non-zero logit differences. The fp32 test proves the *conversion logic* (weights, eps, masks, scales); it does not prove bf16 numerics. When validating a bf16 conversion, compare with a tolerance (ideally also check argmax/top-k agreement and generated text), and don't treat a bf16 diff as a conversion bug unless the fp32 test also fails.
