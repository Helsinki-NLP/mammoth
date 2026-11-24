# LiteRT conversion and Android deployment — reference

Status: LiteRT single-task ✓, multi-task ✓. Android deployment on SM 8750 NPU is in progress. For the step-by-step device procedure see `MammothNmtApp/RUNBOOK.md` (authoritative). For NPU failure root causes see the project memory notes (QNN version match, partition/PD exhaustion, KV-cache design, fp16 overflow).

All LiteRT code lives in `mammoth/mammoth/litert/`. `litert_torch` and the NPU tools are Linux-only: conversion and NPU compilation run on **LUMI**, not macOS.

## Commands

**Single-task conversion** (one task → one `.tflite`):
```bash
python mammoth/litert/convert.py \
    --checkpoint-dir path/to/checkpoint_dir \
    --task eng-spa \
    --enc-max-len 128 --dec-max-len 64 \
    --output mammoth_eng_spa.tflite
# On macOS: --export-only is set automatically (litert_torch is Linux-only)
```

**Single-task inference**:
```bash
python mammoth/litert/infer.py \
    --tflite mammoth_eng_spa.tflite \
    --frame path/to/_best_frame.pt \
    --task eng-spa \
    --src "Hello world ."
# Or with saved tokenizer dirs:
python mammoth/litert/infer.py \
    --tflite mammoth_eng_spa.tflite \
    --src-tok ./hf_eng_spa/src_tokenizer \
    --tgt-tok ./hf_eng_spa/tgt_tokenizer \
    --src "Hello world ."
```

**Multi-task conversion** (all tasks → single `.tflite` + manifest):
```bash
python mammoth/litert/convert_multi.py \
    --checkpoint-dir path/to/checkpoint_dir \
    --enc-max-len 128 --dec-max-len 64 \
    --output mammoth_multi.tflite
# Produces: mammoth_multi.tflite + mammoth_multi_manifest.json
```

**Multi-task inference**:
```bash
python mammoth/litert/infer_multi.py \
    --tflite mammoth_multi.tflite \
    --task fin-swe \
    --frame path/to/_best_frame.pt \
    --src "Hei maailma ."
# Manifest is auto-detected from .tflite path; pass --manifest to override
```

**NPU AOT compilation** (Linux/LUMI only — requires `ai-edge-litert-sdk-qualcomm-nightly`):
```bash
# All registered backends:
python mammoth/litert/compile_npu.py \
    --tflite mammoth_eng_spa.tflite \
    --output ./npu_output

# Specific Qualcomm SoC targets:
python mammoth/litert/compile_npu.py \
    --tflite mammoth_multi.tflite \
    --output ./npu_output \
    --targets SM8650 SM8750
# Produces: npu_output/<name>_fallback.tflite + npu_output/<name>_Qualcomm_SM8650.tflite …
```

## Module layout (`mammoth/mammoth/litert/`)

| File | Purpose |
|---|---|
| `wrapper.py` | `MammothLiteRTEncoder`, `MammothLiteRTPrefill`, `MammothLiteRTDecode` + sample-input factories |
| `convert.py` | Single-task checkpoint → `.tflite` (3 signatures: encode/prefill/decode) |
| `infer.py` | Single-task greedy inference loop via `CompiledModel` |
| `convert_multi.py` | Multi-task checkpoint → single `.tflite` with per-language signatures + manifest |
| `infer_multi.py` | Multi-task inference: reads manifest, selects correct signatures per task |
| `compile_npu.py` | AOT-compile a `.tflite` for Qualcomm (SM8450–SM8750) and MediaTek NPUs; Linux only |

## Multi-task signature design

`convert_multi.py` discovers languages and weight-sharing structure from `opts.tasks` — no hardcoded assumptions. For N source and M target languages it emits:
- `encode_{src}` × N — one per source language
- `prefill_{tgt}` × M — one per target language
- `decode_{tgt}` × M — one per target language

Stacks whose `(stack_idx, xcoder_id)` pair appears in multiple language models (e.g. the shared `_all` stacks) reference the same `nn.Parameter` objects, so litert_torch deduplicates them in the flatbuffer.

The **manifest JSON** (`*_manifest.json`) maps task-ids to `(src, tgt)` pairs:
```json
{
  "model_type": "mammoth_litert_multi",
  "enc_max_len": 128,
  "dec_max_len": 64,
  "tasks": { "fin-swe": {"src": "fin", "tgt": "swe"}, … },
  "src_vocab_sizes": { "fin": 32002, … },
  "tgt_vocab_sizes": { "swe": 32000, … }
}
```

## Reference file map

| What | Where |
|---|---|
| Conversion entry point | `litert-torch/docs/pytorch_converter/README.md` |
| Encoder-decoder wrapper pattern | `litert-samples/…/speech_recognition/convert/whisper.py` |
| RMSNorm HLFB patching | `litert-samples/…/speech_recognition/convert/asr_model.py` |
| Quantization recipes | `litert-torch/litert_torch/generative/quantize/quant_recipes.py` |
| AOT NPU compilation | `litert-samples/…/speech_recognition/convert/compile_for_npu.py` |
| Debugging failures | `litert-torch/litert_torch/debug/culprit.py` |
| AOT + PODAI tutorial | `litert-samples/compiled_model_api/colab/LiteRT_AOT_Compilation_Tutorial.ipynb` |

## Android deployment

**Goal:** run the NPU-AOT-compiled `.tflite` inside `MammothNmtApp` on a Snapdragon 8 Elite phone and validate that output token ids match `infer.py`.

```
LUMI: checkpoint → convert.py → .tflite → compile_npu.py → mammoth_eng_spa_Qualcomm_SM8750.tflite
                                                                        ↓
Mac: scp → Android Studio build → adb install → adb push model.tflite → phone runs NPU inference
```

### `MammothNmtApp/` structure

| Path | Purpose |
|---|---|
| `app/src/main/java/…/MammothTranslator.kt` | Core inference: loads `.tflite`, runs encode/prefill/decode loop via LiteRT CompiledModel API |
| `app/src/main/assets/src_tokenizer.json` | HF tokenizer for source language (copy from LUMI) |
| `app/src/main/assets/tgt_tokenizer.json` | HF tokenizer for target language (copy from LUMI) |
| `litert_npu_runtime_libraries/qualcomm_runtime_v79/` | SM 8750 NPU runtime (`.aar` / `.so`); fetched via `fetch_qualcomm_library.sh` |
| `RUNBOOK.md` | Authoritative step-by-step deployment guide |

### NPU runtime version map

| Qualcomm runtime dir | Snapdragon SoC | Phone example |
|---|---|---|
| `qualcomm_runtime_v79` | SM 8750 (Snapdragon 8 Elite) | **target device** |
| `qualcomm_runtime_v81` | SM 8 Gen 4 / next-gen | future |
| `qualcomm_runtime_v75` | SM 8 Gen 3 | Galaxy S24 |
| `qualcomm_runtime_v73` | SM 8 Gen 2 | Galaxy S23 |
| `qualcomm_runtime_v69` | SM 8 Gen 1 | Galaxy S22 |
