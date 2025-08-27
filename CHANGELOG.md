# HuggingFace Model Integration Update

## Overview

This update introduces HuggingFace model integration capabilities to Mammoth, enabling seamless conversion and use of pre-trained HuggingFace BART models within the Mammoth translation framework.

For usage instructions, please refer to README.md.

## What Was Updated (27-Aug-2025)
Added `--valid_metrics` in training configuration to enable in-training-validation with metrics. 

Added metric "BLEU" from `sacrebleu` library as an option of `--valid_metrics`. By default the result is the corpus BLEU score against the reference. 

## What Was Updated (22-Aug-2025)

### 🆕 New Features

#### 1. HuggingFace Model Converter (`hf2mammoth.py`)
- **Complete BART to Mammoth conversion pipeline**
- Converts HuggingFace BART models to Mammoth-compatible format
- Three-stage conversion process:
  - Stage 1: HuggingFace BART → X-Transformers
  - Stage 2: X-Transformers → Mammoth model
  - Stage 3: Save as Mammoth checkpoint
- Automatic vocabulary extraction from HF tokenizers for on-the-fly tokenization/detokenization

#### 2. X-Transformers library integration & update (`mammoth/x_transformers/`)
- Integrated the X-Transformers library to Mammoth directoy.
- Updated the supported [X-Transformers](https://github.com/lucidrains/x-transformers) library version to 2.7.2.

