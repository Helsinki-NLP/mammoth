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
    <tgt-tokenizer> 
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

