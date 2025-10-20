# HuggingFace Model Integration Update

## Overview

This update introduces HuggingFace model integration capabilities to Mammoth, enabling seamless conversion and use of pre-trained HuggingFace BART models within the Mammoth translation framework.

For usage instructions, please refer to README.md.

## What Was Updated (20-Oct-2025)

**HuggingFace Integration Module Structure**

- **Module Organization**: Established formal `mammoth/hf_integration/` module structure
  - `from_hf/`: HuggingFace → MAMMOTH model conversion (BART, ModernBERT, etc.)
  - `to_hf/`: MAMMOTH → HuggingFace custom model export
  - Auto-registration with HuggingFace AutoModel classes
- **Moved Legacy Scripts**: Relocated conversion scripts from root to proper module locations

**CSC Environment Configuration Reorganization**

- **Structured Configuration**: Reorganized Puhti and LUMI configs by deployment scale
  - `one_node/`: A testing single-node training configuration
  - `two_nodes/`: A testing two-node training configuration
  - `four_nodes/`: A testing four-node (16 GPU) training configuration

**x-transformers Library Updates**
- Update the x-transformers library to 2.9.2 (https://github.com/lucidrains/x-transformers/releases/tag/2.9.2)
- Add global/local sliding window attention support to x-transformers `attend.py`. 
- Add global/local RoPE theta value support for global/local attention layers.

**Core Framework Improvements**

- **Vocabulary System**: Refined `mammoth/inputters/vocab.py` and added `language_tokens.py` for better HF tokenizer support
- **Transform Pipeline**: Updated denoising, filtering, and tokenization transforms for HF tokenizer compatibility

**Files Added:**
- `mammoth/hf_integration/__init__.py`: Module initialization with AutoModel registration
- `mammoth/hf_integration/from_hf/{__init__.py,hf2mammoth2hf.py}`: HF→MAMMOTH conversion
- `mammoth/hf_integration/from_hf/modernBERT/hfModernBERT2mammoth.py`: ModernBERT converter
- `mammoth/hf_integration/to_hf/README.md`: Export documentation
- `mammoth/models/architecture_config.py`: Architecture configuration utilities
- `mammoth/inputters/language_tokens.py`: Language token management
- `mammoth/utils/x_transformers/*.py`: x-transformers utilities (12 new modules)
- `mammoth/x_transformers/{bert_padding.py,gpt_vae.py}`: New transformer components
- `csc_env/{puhti,lumi}/four_nodes/{train.yaml,inference.yaml}`: 4-node configs
- `csc_env/puhti/four_nodes/{train.sh,multinode_train_script.sh}`: Multi-node training scripts
- `csc_env/puhti/eval.yaml`: Evaluation configuration

## What Was Updated (06-Oct-2025)

**HuggingFace Tokenizers Library Integration**

- **New Feature**: Added support for HuggingFace `tokenizers` library for training from scratch
- **Tokenizer Support**:
  - New `HFTokenizerVocab` class wraps HuggingFace tokenizers with MAMMOTH's vocab interface
  - Automatic detection of `.json` tokenizer files
  - Support for BPE, WordPiece, Unigram, and other modern tokenization algorithms
- **Configuration**:
  - Added `--use_hf_tokenizer` flag to enable HF tokenizer mode
  - Backward compatible with traditional vocabulary files
- **Documentation**:
  - Comprehensive guide at `docs/HF_TOKENIZERS.md`
  - Covers installation, usage, advanced configuration, and troubleshooting

**Files Modified:**
- `mammoth/inputters/vocab.py`: Added `HFTokenizerVocab` class and updated `get_vocab()` factory
- `mammoth/inputters/dataset.py`: Enhanced dataset loading to support HF tokenizers
- `mammoth/translate/translation.py`: Updated translation decoding for HF tokenizers
- `mammoth/opts.py`: Added `--use_hf_tokenizer` configuration option
- `mammoth/bin/train.py`: Modified to pass tokenizer configuration to vocab loading

**Files Added:**
- `examples/hf_tokenizers/train.py`: Example tokenizer training script
- `docs/HF_TOKENIZERS.md`: Complete documentation for HF tokenizers integration

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
