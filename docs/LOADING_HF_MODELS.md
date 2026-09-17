# Loading HuggingFace Models into Mammoth

This guide explains how to load pretrained models from the HuggingFace Model Hub into Mammoth format for training or inference.

## Overview

Mammoth provides converters to import HuggingFace pretrained models as initialization for your translation models. This allows you to leverage existing pretrained weights and fine-tune them for your specific translation tasks.

**Currently Supported Models:**
- **ModernBERT** - Encoder-only model (creates random decoder)
- **Gemma3** - Decoder-only causal language model (creates random encoder)
- **Hybrid ModernBERT + Gemma3** - Combines pretrained encoder and decoder

**Note:** BART support is under development and will be added in a future release.

## ModernBERT Encoder

Convert a HuggingFace ModernBERT model to use as the encoder in Mammoth. A standard 6-layer transformer decoder will be randomly initialized.

### Basic Usage

```bash
python mammoth/hf_integration/from_hf/modernBERT/BERT2mammoth.py \
    <hf_model_path> \
    <save_path> \ 
    <src-tokenizer> \
    --tgt-tokenizer <lang> <tgt-tokenizer-path>
```

### What Gets Converted

**Encoder (ModernBERT pretrained weights):**
- 22 layers (for base model)
- GeGLU MLP activation
- Fused QKV projections (Not implemented yet)
- BERT unpadding (Not implemented yet)
- Bias-free architecture
- Sliding window attention (with different RoPE theta values)

**Decoder (randomly initialized):**
- 6 layers
- Same hidden dimensions as encoder
- Standard transformer with full attention
- Will be trained from scratch

### Output Files

After conversion, you'll find in the save directory:
- Mammoth checkpoint components
- `tokenizer.json` - HuggingFace tokenizer file
- Layer name mappings for debugging

### Further fine-tuning

The converted model (ModernBert + randomly initialized decoder) is ready for further fine-tuning with the example config `mammoth/hf_integration/from_hf/modernBERT/train.yaml`.

## Gemma3 Decoder (native PyTorch backend)

Convert a HuggingFace Gemma3 model (text-only, e.g. `gemma3_270m`) into Mammoth using the **native PyTorch transformer backend** (`mammoth/mammoth/modules/transformer/`), not x_transformers. See `CLAUDE.md`'s "Gemma3-270M -> Mammoth Conversion" section for the full architecture gap analysis.

Gemma3 is decoder-only; the encoder is a small randomly-initialized stack whose cross-attention output is zero-initialized at conversion time, so the converted model's decoder path is mathematically identical to running Gemma3 alone — the encoder only exists to satisfy Mammoth's encoder-decoder task plumbing.

### Basic Usage

```bash
python mammoth/hf_integration/from_hf/gemma3/convert_gemma3_native.py \
    <hf_gemma3_path> \
    <converted_model_save_path> \
    --src-lang <lang> \
    --tgt-lang <lang> \
    --task-id <task_id> \
    --encoder-group <enc_sharing_group> \
    --decoder-group <dec_sharing_group>
```

All flags are optional:
- `--task-id` defaults to `<src-lang>-<tgt-lang>`
- `--encoder-group` / `--decoder-group` (enc/dec sharing-group xcoder ids) default to `fake_enc` / `gemma3_dec`
- `--src-lang` / `--tgt-lang` default to `gemma3` — both share Gemma3's own tokenizer; `--tgt-lang` is the meaningful one, since it owns the converted decoder weights
- `--enc-layers` (default `1`) / `--enc-model-dim` (default `64`) size the fake encoder
- `--weight` (default `1.0`) sets the task's weight; task scheduling is always `weighted_sampling`

### What Gets Converted

**Encoder (randomly initialized, tiny):**
- 1 layer / 64 hidden dim by default — cross-attn K/V project from the encoder's own dimension (`context_dim`), so it does **not** need to match the decoder's hidden size
- Cross-attn output is zero-initialized, so it never contributes to the decoder's residual stream

**Decoder (Gemma3 pretrained weights):**
- RMSNorm with unit offset (`(1 + weight)`, folded into the copied weight at load time)
- Sandwich norm (4 normalizations per layer)
- Q/K normalization
- Grouped/Multi-Query Attention
- GeGLU MLP (GELU-tanh gated)
- Dual RoPE (separate theta for sliding-window vs. full-attention layers) + sliding-window attention
- Scaled word embeddings
- Bias-free architecture

### Verification

`mammoth/tests/test_gemma3_native_conversion.py` loads the real checkpoint and asserts the converted model's logits match HF's own forward pass **exactly** (weights are copied by walking the built `nn.Module` objects directly — no string-key mapping).


## Hybrid: ModernBERT Encoder + Gemma3 Decoder

Combine pretrained ModernBERT encoder with pretrained Gemma3 decoder to create a fully pretrained encoder-decoder model.

### Basic Usage

```bash
python mammoth/hf_integration/from_hf/hybrid/modernBERT_gemma3_2mammoth.py \
    <hf_modernbert_path> \ 
    <hf_gemma3_path> \
    <converted_model_save_path> \
    # --src-tokenizer \ the conversion script uses the encoder's tokenizer by default
    # --tgt-tokenizer \ the conversion script uses the decoder's tokenizer by default
    # --src-lang \ modernbert is trained mainly in English, so 'en' is by default
    --tgt-lang <lang> \
```
Note: the `task_id` in the converted model will be `task_<lang>_<lang>` by default.

### Architecture

**Encoder:** ModernBERT pretrained weights
- 22 layers, 768 hidden dimensions

**Decoder:** Gemma3 pretrained weights
- Variable dimensions (e.g., 640 for 270M model, larger for 4B)

**Cross-Attention Bridge:**
- If encoder and decoder dimensions match: Direct cross-attention
- If dimensions differ: using the parameter `dec_cross_attn_dim_context` in the training to tell decoder to expect encoder's dimension

### Output Files

After conversion, you'll find:
- Mammoth checkpoint with both encoder and decoder components
- Separate tokenizers for encoder and decoder (if different)
- Configuration preserving both model architectures

All converted models are ready for inferencing and further fine-tuning in Mammoth as regular models.
Please find the template config files for single node training and inferencing at `csc_env/lumi/train.yaml` and `csc_env/lumi/inference.yaml`.