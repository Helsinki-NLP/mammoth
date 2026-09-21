# Loading HuggingFace Models into Mammoth

This guide explains how to convert a pretrained HuggingFace model into a Mammoth checkpoint, on the **native PyTorch transformer backend** (`mammoth/mammoth/modules/transformer/`).

**Currently supported: Gemma3-270M** (`google/gemma3-270m`-style text-only checkpoints). Earlier converters for ModernBERT, BART, and the ModernBERT+Gemma3 hybrid (`mammoth/hf_integration/from_hf/{modernBERT,BART,hybrid}/`) predate this native-backend migration and target the deprecated x_transformers backend — they are not part of the `pytorch_backend` line and are not covered here.

## How it works

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
- The exact epsilon from the HF config (`rms_norm_eps`) on every norm — a mismatch here produces large logit divergence despite every weight matching (see CLAUDE.md's "Important gotcha" note)

## Usage

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

### Example: SLURM conversion jobs on LUMI

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

### Verifying a conversion

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
