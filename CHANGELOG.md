# HuggingFace Model Integration Update

## Overview

This update introduces HuggingFace model integration capabilities to Mammoth, enabling seamless conversion and use of pre-trained HuggingFace BART models within the Mammoth translation framework.

For usage instructions, please refer to README.md.

## What Was Updated (26-Sep-2025)

**Training Improvements**

- **New Option**: Added `--valid_at_start` flag to perform validation before training begins (the validation runs only on master node by far).
- Added `--src_subword_type` and `--tgt_subword_type` parameters for autotokenizer denoising transforms. The current denoising transforms support `sentencepiece`, `bpe`, `none` options for both source and target sides.
- Fixed the bug that the training will crash when no task sampling has occurred

## What Was Updated (15-Sep-2025)

**LUMI Supercomputer Support**

- Added LUMI environment setup script (`lumi/env_setup.sh`) for PyTorch virtual environment configuration
- Added SLURM batch scripts for training (`lumi/train.sh`) and translation (`lumi/translate.sh`) on LUMI
- Minimal requirements file (`requirements_lumi.txt`) for LUMI deployment with essential dependencies

**Improvements**

- Added early validation for save_model path with proper error handling and directory creation. So the destination directory will be validated before the training starts.
- Improved error messages for save_model path validation

## What Was Updated (05-Sep-2025)

**HuggingFace Converter Enhancement (`hf2mammoth2hf.py`)**

- Convert Mammoth models (previously converted from HF) back to HuggingFace format
- Push converted models directly to HuggingFace model hub

### Enhanced Dependencies

- Added `sentencepiece==0.2.1` for tokenization/detokenization support
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
