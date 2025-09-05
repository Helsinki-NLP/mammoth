# HuggingFace Model Integration Update

## Overview

This update introduces HuggingFace model integration capabilities to Mammoth, enabling seamless conversion and use of pre-trained HuggingFace BART models within the Mammoth translation framework.

For usage instructions, please refer to README.md.

## What Was Updated (05-Sep-2025)

**HuggingFace Converter Enhancement (`hf2mammoth2hf.py`)**

- Convert Mammoth models (previously converted from HF) back to HuggingFace format
- Push converted models directly to HuggingFace model hub

### Enhanced Dependencies

- Added `sentencepiece==0.2.1` for improved tokenization support
- Cleaned up NVIDIA CUDA dependencies for broader compatibility

## What Was Updated (27-Aug-2025)

**Training Configuration Updates**

- Added `--valid_metrics` parameter to training configuration to enable in-training validation with metrics
- Added "BLEU" metric from `sacrebleu` library as an option for `--valid_metrics`. By default, this returns the corpus BLEU score against the reference. 

## What Was Updated (22-Aug-2025)

#### 1. HuggingFace Model Converter (`hf2mammoth.py`)

- **Complete BART to Mammoth conversion pipeline**
- Converts HuggingFace BART models to Mammoth-compatible format
- Three-stage conversion process:
  - Stage 1: HuggingFace BART → X-Transformers
  - Stage 2: X-Transformers → Mammoth model
  - Stage 3: Save as Mammoth checkpoint
- Automatic vocabulary extraction from HF tokenizers for on-the-fly tokenization/detokenization

#### 2. X-Transformers library integration & update (`mammoth/x_transformers/`)

- Integrated the X-Transformers library into the Mammoth directory
- Updated the supported [X-Transformers](https://github.com/lucidrains/x-transformers) library version to 2.7.2