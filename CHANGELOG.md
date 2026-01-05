#  Mammoth Changelog

## Overview

This branch mainly works on two fronts:
- Introduces HuggingFace model integration capabilities to Mammoth, enabling seamless conversion and use of pre-trained HuggingFace models within the Mammoth translation framework.
- Improves Mammoth's core training and model architecture features.

For usage instructions, please refer to README.md.

## What Was Updated (05-Jan-2026)

**Training and Profiling Improvements**

- **Verbose DataLoader Logging**: Added `--verbose_dataloader` option to show dataset continuation lines from all devices at INFO level
  - Previous behavior: Only master device logged at INFO level, others logged at WARN level
  - New behavior: All devices log at INFO level when flag is enabled
  - Useful for debugging multi-device training synchronization

- **PyTorch Profiler Integration**: Added comprehensive profiling support for performance optimization
  - Master switch `--enable_profiling` controls all profiling (main training loop + data pipeline annotations)
  - Removed separate `--profile_data_pipeline` option (now controlled by master switch)
  - Zero profiling overhead when disabled (no-op context managers)
  - Options include: `profile_output_dir`, `profile_wait`, `profile_warmup`, `profile_active`, `profile_repeat`
  - Advanced options: `profile_record_shapes`, `profile_memory`, `profile_with_stack`
  - **Note**: Only use for optimization/debugging due to overhead

- **Default Beam Size for Validation**: Changed default `beam_size` from 5 to 1 for in-training validation
  - Faster validation with greedy search (beam_size=1)
  - Reduces validation time significantly during training
  - Users can still set higher beam sizes for better quality validation if needed

**Files Modified:**
- `mammoth/opts.py`: Verbose dataloader option (+4 lines), profiling consolidation (-8 lines), beam_size default changed (1 line)
- `mammoth/train_single.py`: Verbose dataloader logging (+2 lines), profiling flag updates (+2 lines)
- `mammoth/trainer.py`: Conditional profiler implementation (+20 lines), profiling wrapper updates (+45 lines)
- `mammoth/distributed/communication.py`: Profiling flag update (1 line)


## What Was Updated (30-Dec-2025)

**Multi-Task Validation and Checkpoint Improvements**

- **Multi-Task Best Checkpoint Selection (Issue #148)**: Complete reimplementation of best checkpoint saving with distributed metric aggregation
  - Metrics gathered across all devices/tasks using conservative aggregation strategy
  - For BLEU/accuracy (higher is better): uses `min` across tasks to ensure all tasks perform well
  - For PPL/loss (lower is better): uses `max` across tasks to catch worst-performing tasks
  - Best checkpoints now include all tasks, not just task on device 0:0
  - Enhanced checkpoint discovery supporting explicit `*_best_frame.pt` naming pattern
  - Comprehensive metadata tracking for all validation metrics across checkpoints

- **Beam Search in Validation**: In-training validation now uses same generation method as inference
  - Added `beam_size` parameter to training config for consistent validation metrics
  - New `_generate_predictions_autoregressive()` method implementing beam search/greedy search
  - Validation results now match post-training inference metrics
  - Note: Practically, only `beam_size: 1` is viable due to OOM constraints with larger beams

- **Conditional Validation Execution**: Validation now only runs on devices with validation data configured
  - Only tasks with both `path_valid_src` and `path_valid_tgt` defined will perform validation
  - Devices without validation data skip validation but participate in synchronization
  - Pre-training validation (`valid_at_start`) runs on all devices with validation data, not just master

- **Enhanced Restoration Logging**: Training restoration from checkpoints now shows information across all devices
  - Dataset continuation logging changed from `info` to `warning` level for better visibility
  - Restoration messages now appear for all tasks, showing correct starting points
  - First 5 lines of data logged for verification during continuation

**Distributed Training Stability**

- **NCCL Synchronization Improvements**: Added distributed barriers to prevent timeout issues
  - Barrier after validation steps ensures all devices stay synchronized
  - Barrier after checkpoint file writes ensures all components saved before renaming
  - NCCL backend now uses explicit `device_id` parameter to suppress warnings

**Code Quality and Refactoring**

- **Batch Structure Standardization**: Unified batch tensor access patterns
  - Changed `batch.src[0]` → `batch.src.tensor` and `batch.tgt` → `batch.tgt.tensor`
  - Improved code consistency across dataloader and dataset modules

- **Training Options Consolidation**: Integrated decoding options into training configuration
  - Beam search parameters now available in training config without parameter conflicts
  - Conflict prevention for overlapping parameters (`max_length`, reproducibility options)

- **Config Cleanup**: Removed obsolete configuration files
  - Deleted `training_ft.yaml` and `translation_config.yaml`

**Files Modified:**
- `mammoth/distributed/communication.py`: NCCL device_id parameter (+7 lines)
- `mammoth/inputters/dataloader.py`: Batch structure refactoring (+8 lines)
- `mammoth/inputters/dataset.py`: Logging level adjustments (+4 lines)
- `mammoth/opts.py`: Decoding options integration (+11 lines)
- `mammoth/train_single.py`: Pre-training validation enhancement (+7 lines)
- `mammoth/trainer.py`: Validation with beam search, distributed synchronization (+332 lines)
- `mammoth/utils/model_saver.py`: Multi-task metric aggregation, checkpoint management (+287 lines)

**Files Deleted:**
- `training_ft.yaml`: Obsolete configuration (-92 lines)
- `translation_config.yaml`: Obsolete configuration (-29 lines)

## What Was Updated (11-Dec-2025)

**Training Enhancements and Bug Fixes**

- **Best Metrics Saving Strategy**: Added `save_strategy` feature to save checkpoints based on validation metrics
  - Supports saving best models according to perplexity, BLEU, or other validation metrics

- **Validation Timing Fixes**: Fixed `valid_at_start` option to work properly during training initialization

- **Improved Training Continuation Logging**: Enhanced line checker to only log dataset position at training start rather during validation


**Files Modified:**
- `mammoth/inputters/dataset.py`: Dataset continuation logic improvements (+15 lines)
- `mammoth/opts.py`: Added save_strategy and validation options (+28 lines)
- `mammoth/train_single.py`: Training loop integration for new features (+12 lines)
- `mammoth/trainer.py`: Validation timing fixes (+8 lines)
- `mammoth/utils/model_saver.py`: Best metrics saving implementation (+45 lines)
- `mammoth/utils/report_manager.py`: Enhanced logging controls (+6 lines)

## What Was Updated (09-Dec-2025)

**Documentation Updates and Model Validation**

- **HuggingFace Tokenizer Instructions**: Updated documentation with clearer usage instructions
  - Enhanced examples and configuration guidelines
  - Improved getting started guide for HF tokenizers

- **Landing Page Refresh**: Updated main README.md with current project information
  - Added latest features and capabilities
  - Improved project description and setup instructions
  - Hid unfinished README sections for cleaner presentation

- **ModernBERT Validation**: Validated and tested all ModernBERT conversion scripts
  - Ensured compatibility with latest model versions
  - Verified weight mapping correctness

**Files Modified:**
- `docs/LOADING_HF_MODELS.md`: Updated tokenizer instructions (+50 lines)
- `README.md`: Landing page updates (+120 lines)
- `mammoth/hf_integration/from_hf/modernBERT/`: Validation updates (+15 lines)

## What Was Updated (08-Dec-2025)

**Logging Improvements and LUMI Configuration**

- **Refined Special Token Logging**: Fixed "Stripped special tokens" message to appear only when tokens are actually stripped

- **LUMI Onboarding Enhancements**: Updated example configuration files for LUMI supercomputer
  - Now all the training examples are validated for quick starting
  - Improved documentation for new users
  - Streamlined configuration process

**Files Modified:**
- `mammoth/inputters/dataset.py`: Selective logging for special tokens (+8 lines)
- `csc_env/lumi/`: Updated configuration examples (+35 lines)

## What Was Updated (05-Dec-2025)

**Dataset Continuation and Codebase Cleanup**

- **Fixed Dataset Continuation**: Resumed training now correctly starts from the last stopping point
  - Proper state restoration for interrupted training
  - Debug logging shows first 5 lines of source text for verification
  - Maintains training data integrity across sessions

- **Puhti Configuration Removal**: Cleaned up branch by removing Puhti-related configurations
  - Removed outdated Puhti-specific documentation
  - Streamlined codebase focus on LUMI platform

**Files Modified:**
- `mammoth/inputters/dataloader.py`: Dataset continuation fixes (+25 lines)
- `mammoth/inputters/dataset.py`: State restoration improvements (+18 lines)

**Files Removed:**
- `csc_env/puhti/`: Entire Puhti configuration directory (-200+ lines)

## What Was Updated (25-Nov-2025)

**HuggingFace BART Model Integration**

- **Full BART Architecture Support**: Added comprehensive features to accommodate HuggingFace BART models
  - **QKV Bias Support**: Query, Key, Value projection layers now support bias terms
  - **LayerNorm Bias**: Added bias parameter support for normalization layers
  - **Final Logits Bias**: Support for bias term in final output projection
  - **Tied Embeddings**: Implemented weight sharing between encoder and decoder embeddings
  - New converter: `mammoth/hf_integration/from_hf/BART/BART2mammoth.py` (+1302 lines)

- **Advanced Freezing Mechanism**: Complete reimplementation of parameter freezing with fine-grained control
  - Freeze encoder embedding independently
  - Freeze encoder layers (excluding embeddings)
  - Freeze cross-attention layers only
  - Freeze decoder layers (including or excluding cross-attention)
  - Freeze decoder embeddings independently
  - Removed legacy freezing implementation
  - Note: Output projection layer remains trainable in all configurations

- **Sliding Window Attention Reinstatement**: Restored sliding window attention functionality
  - Local/global attention pattern support
  - Configurable window sizes per layer

- **RoPE Theta Configuration**: Reinstated dual theta value support for RoPE
  - Separate theta values for local attention layers (`local_rope_theta`)
  - Separate theta values for global attention layers (`global_rope_theta`)

**Files Modified:**
- `mammoth/distributed/components.py`: Component freezing support (+5 lines)
- `mammoth/model_builder.py`: BART features, freezing logic, tied embeddings (+282 lines)
- `mammoth/opts.py`: New command-line options for freezing and BART features (+143 lines)
- `mammoth/train_single.py`: Training loop freezing integration (+5 lines)
- `mammoth/utils/parse.py`: Configuration parsing for new features (+36 lines)
- `mammoth/x_transformers/attend.py`: Sliding window attention restoration (+80 lines)
- `mammoth/x_transformers/x_transformers.py`: BART architecture support, tied embeddings (+203 lines)

**Files Added:**
- `mammoth/hf_integration/from_hf/BART/BART2mammoth.py`: Comprehensive BART to Mammoth converter

**Files Removed:**
- `mammoth/hf_integration/from_hf/__init__.py`: Outdated converter code (-12 lines)
- `mammoth/hf_integration/from_hf/modernBERT/ROPE_QUICK_REFERENCE.md`: Documentation consolidation (-243 lines)


## What Was Updated (19-Nov-2025)

**Hybrid Model Architecture and Training Enhancements**

- **Hybrid ModernBERT-Gemma3 Model**: Added support for combining HuggingFace ModernBERT as encoder with Gemma3 as decoder
  - New converter: `mammoth/hf_integration/from_hf/hybrid/modernBERT_gemma3_2mammoth.py`
  - Complete training configuration: `mammoth/hf_integration/from_hf/hybrid/train.yaml`

- **Scaled Embedding Support**: Added `scaled_embedding` parameter to accommodate Gemma3 training requirements

- **Enhanced Optimizer State Management**: Improved `reset_optim` logic with four distinct scenarios
  - Correct handling of optimizer states, momentum, and steps from previous checkpoints

- **Advanced Sentence Filtering**: Enhanced `filtertoolong` transform with dual-range filtering
  - Can now filter sentences longer than maximum tokens AND shorter than minimum tokens

**Files Modified:**
- `mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py`: Updated for hybrid compatibility
- `mammoth/hf_integration/from_hf/hf2mammoth2hf.py`: Removed outdated converter
- `mammoth/model_builder.py`: Added scaled_embedding parameter support
- `mammoth/opts.py`: Added new training and filtering options
- `mammoth/train_single.py`: Enhanced optimizer reset logic
- `mammoth/transforms/filtering.py`: Dual-range sentence length filtering
- `mammoth/utils/optimizers.py`: Improved optimizer state management
- `mammoth/x_transformers/x_transformers.py`: Scaled embedding integration

**Files Added:**
- `mammoth/hf_integration/from_hf/hybrid/modernBERT_gemma3_2mammoth.py`: Hybrid model converter
- `mammoth/hf_integration/from_hf/hybrid/train.yaml`: Hybrid model training configuration

## What Was Updated (18-Nov-2025)

**Enhanced Encoder-Decoder Architecture Support**

- **Dual Architecture Model Support**: Added support for using different transformer architectures for encoder and decoder in the same model
  - ModernBERT can now be used as encoder while Gemma3 serves as decoder in a single translation model
  - Enables optimal architecture selection per component (encoder/decoder) rather than one-size-fits-all approach

- **Encoder/Decoder-Specific Parameter Prefixing**: Implemented `enc_` and `dec_` prefix system for more x_transformers parameters

**HuggingFace Model Converter Expansion**

- **Gemma3 Decoder Support**: Added comprehensive Gemma3 270M to Mammoth decoder converter (`mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py`)

- **ModernBERT Encoder Enhancements**: Updated ModernBERT to Mammoth encoder converter 
  - Added `mammoth/hf_integration/from_hf/modernBERT/train.yaml` for single-architecture training
  - Enhanced weight mapping for encoder-only use case
  - Better integration with multilingual training pipeline

**Files Added:**
- `mammoth/hf_integration/from_hf/gemma3/gemma2mammoth.py`: Comprehensive Gemma3 decoder converter
- `mammoth/hf_integration/from_hf/gemma3/train.yaml`: Gemma3 training configuration
- `mammoth/hf_integration/from_hf/modernBERT/train.yaml`: ModernBERT training configuration

## What Was Updated (29-Oct-2025)

**Advanced Attention Mechanisms**

- **Sliding Window Attention**: Added support for local sliding window attention with configurable window sizes
  - Native integration with Flash Attention library for optimal performance
  - Fallback to PyTorch SDPA with manual window masking when Flash Attention is unavailable
  - Per-layer window size configuration via `sliding_window` and `global_attn_every_n_layers` parameters
  - Efficient caching mechanism for window masks to reduce overhead

- **Flexible RoPE Theta for Global/Local Attention**: Per-layer RoPE (Rotary Position Embedding) theta configuration
  - Support for different theta values for global attention layers (`global_rope_theta`) and local attention layers (`local_rope_theta`)
  - Enables ModernBERT-style architecture with alternating global/local attention patterns
  - Layer-specific RoPE instances with customizable interpolation and scaling factors

**Training Precision and Performance**

- **BFloat16 (bf16) Support**: Added native support for BF16 mixed precision training
  - New `--model_dtype` option now accepts `'bf16'` in addition to `'fp32'` and `'fp16'`
  - Automatic dtype selection in autocast context for both training and validation
  - Better numerical stability than FP16 for large-scale training

- **Unpadding for Variable-Length Sequences**: ModernBERT-style unpadding implementation for efficiency
  - Removes padding tokens before attention computation to reduce wasted computation
  - Implements `IndexFirstAxis` and `IndexPutFirstAxis` custom autograd functions for efficient indexing
  - `unpad_input()` and `pad_input()` utilities for removing and restoring padding
  - Significant speedup for batches with variable-length sequences


**Distributed Training Improvements**

- **Detach and Reattach Method for Data Preparation**: New serialization approach for multi-node training
  - `_detach_batch_tensors()`: Recursively converts all batch tensors to CPU NumPy arrays before serialization
  - `_reattach_batch_tensors()`: Converts NumPy arrays back to PyTorch tensors after deserialization
  - Prevents `/dev/shm` race conditions in containerized multi-node environments
  - Bypasses PyTorch's automatic shared memory pickling which can cause issues on CSC supercomputers
  - Preserves tensor dtypes through serialization via string representation
  - Handles nested structures (dicts, lists, tuples, namedtuples, custom objects)
**NOTE:** 
1. This method is currently experimental and may have performance overhead due to CPU-GPU transfers.
2. The training will crash at very end after the model is fully saved.

**Model Architecture Configuration**

- **Architecture-Specific Configuration Registry**: New system for architecture-specific parameters
  - Centralized configuration for BART, ModernBERT, and potentially other transformer architectures
  - Automatic parameter routing for attention layers, feed-forward networks, and normalization
  - Configurable activation functions, bias terms, and initialization schemes
  - Files added: `mammoth/models/architecture_config.py`

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
