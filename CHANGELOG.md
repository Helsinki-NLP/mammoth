# HuggingFace Model Integration Update

## Overview

This update introduces HuggingFace model integration capabilities to Mammoth, enabling seamless conversion and use of pre-trained HuggingFace BART models within the Mammoth translation framework.

## What Was Updated

### 🆕 New Features

#### 1. HuggingFace Model Converter (`hf_converter.py`)
- **Complete BART to Mammoth conversion pipeline**
- Converts HuggingFace BART models to Mammoth-compatible format
- Three-stage conversion process:
  - Stage 1: HuggingFace BART → X-Transformers
  - Stage 2: X-Transformers → Mammoth model
  - Stage 3: Save as Mammoth checkpoint
- Automatic vocabulary extraction from HF tokenizers for on-the-fly tokenization/detokenization

#### 2. X-Transformers library integration & update (`mammoth/x_transformers/`)
- Updated the supported [X-Transformers](https://github.com/lucidrains/x-transformers) library version to 2.7.2.


### 📁 File Structure Changes

```
mammoth/
├── hf_converter.py                    # NEW: HF to Mammoth converter
├── translation_config.yaml           # NEW: Translation configuration
├── x_transformers/                   # NEW: X-Transformers library integration
│   ├── __init__.py
│   ├── x_transformers.py            # Core transformer implementation
│   ├── attend.py                    # Attention mechanisms
│   ├── autoregressive_wrapper.py    # AR model wrapper
│   └── ... (10+ additional modules)
├── constants.py                     # UPDATED: Token definitions
├── inputters/
│   ├── dataset.py                   # UPDATED: Token sequence handling
│   └── vocab.py                     # UPDATED: Vocabulary management
└── ... (40+ files with import/formatting updates)
```


#### With HuggingFace Ecosystem
- **Model Hub**: Direct loading from HF model hub
- **Tokenizers**: Automatic vocabulary extraction, on-the-fly tokenization/detokenization
- **Configurations**: Preserves original model hyperparameters

For usage instructions, please refer to README.md.