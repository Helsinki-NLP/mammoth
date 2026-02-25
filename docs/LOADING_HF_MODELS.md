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

## Gemma3 Decoder

Convert a HuggingFace Gemma3 model to use as a decoder-only model in Mammoth. The encoder will be randomly initialized.

### Basic Usage

```bash
python mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py \
    <hf_gemma3_path> \ 
    <converted_model_save_path> \
    --src-tokenizer <lang> <src_tokenizer_path> \
    --tgt-tokenizer <lang> <hf_gemma3_path>

```
Note: the `task_id` in the converted model will be `task_<lang>_<lang>` by default.

### What Gets Converted
**Encoder (randomly initialized):**
- 6 layers
- Same hidden dimensions as decoder
- Standard transformer with causal attention
- Will be trained from scratch

**Decoder (Gemma3 pretrained weights):**
- RMSNorm with unit offset
- 4 normalizations per layer (double sandwich)
- Q/K normalization
- Multi-Query Attention (MQA)
- Gated MLP with GELU activation
- Scaled word embeddings
- Bias-free architecture
- Sliding window attention

### Output Files

After conversion, you'll find:
- Mammoth checkpoint with decoder-only architecture
- `tokenizer.json` - HuggingFace tokenizer file
- Weight mapping information


## Hybrid: ModernBERT Encoder + Gemma3 Decoder

Combine pretrained ModernBERT encoder with pretrained Gemma3 decoder to create a fully pretrained encoder-decoder model.

### Basic Usage

```bash
python mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py \
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