# HuggingFace Tokenizers Integration

This document describes how to use HuggingFace tokenizers library with MAMMOTH for training from scratch.

## Overview

MAMMOTH now supports using HuggingFace's `tokenizers` library as an alternative to traditional vocabulary files. This enables:

- **Increasing compatibility**: Now Mammoth interacts more smoothly with HuggingFace models and tokenizers.
- **Easier implementation**: Users don't have to worry about separate vocab and tokenizing models.
- **Tokenization algorithms preserved**: BPE, WordPiece, Unigram are still supported.
- **Faster tokenization**: HuggingFace tokenizers are implemented in Rust for speed

## Installation

Install the HuggingFace tokenizers library:

```bash
pip install tokenizers
```

## Quick Start

### 1. Train a Tokenizer

Use the provided example script to train a tokenizer:

```bash
cd examples/hf_tokenizers
python train.py --input_file your_corpus --output_dir your_output_dir --vocab_size 32000
```

**Key Configuration:**
- **Vocabulary size**: 32,000 tokens (configurable via `VOCAB_SIZE`)
- **Model type**: BPE (Byte Pair Encoding)
- **Special tokens**: `</s>`, `<pad>`, `<s>`, `<unk>`, `<mask>` (Hard coded for Mammoth by default. Please do not modify unless you know what you are doing.)
- **Pre-tokenizer**: Metaspace (uses `▁` for word boundaries, inherented from sentencepiece style)

### 2. Use the Tokenizer in Mammoth training and inferencing

Add the `--use_hf_tokenizer` flag to your training configuration:

```yaml
# In your training config YAML
use_hf_tokenizer: true

src_vocab:
  en: /scratch/project_462000964/members/wangchao/training/hf_models/modernbert/tokenizer.json

tgt_vocab:
  fi: /scratch/project_462000964/shared/hplt_bilingual/fi-en.tmx/tgt_tokenizer/tokenizer.json
```

MAMMOTH will automatically detect the `.json` extension and load it as a HuggingFace tokenizer on the fly.

## How It Works

### Tokenizer Training

The example script (`examples/hf_tokenizers/train.py`) demonstrates:

1. **Initialization**: Create a BPE tokenizer with special tokens
2. **Pre-tokenization**: Configure Metaspace pre-tokenizer for word boundaries
3. **Training**: Train on your corpus to learn subword vocabulary
4. **Decoding**: Set up proper decoder for text reconstruction
5. **Padding**: Enable padding for batch processing

**Important Design Choice:**
- Non-mammoth native HF models often have their input/output templates, hence the special tokens are added during tokenization encoding.
- Self-trained tokenizers perform **ONLY** subword tokenization. Special tokens (BOS/EOS) are added by MAMMOTH during data loading.
- This gives flexibility to experiment with different special token strategies

### Integration with MAMMOTH

The integration happens in several places:

**1. Vocabulary Loading** (`mammoth/inputters/vocab.py`):
- New `HFTokenizerVocab` class wraps HuggingFace tokenizers
- Provides the same interface as MAMMOTH's traditional `Vocab` class
- Factory function `get_vocab()` automatically detects `.json` files

**2. Dataset Processing** (`mammoth/inputters/dataset.py`):
- Modified to handle both traditional vocabs and HF tokenizers
- Tokenization happens transparently during data loading
- Special tokens (BOS/EOS/PAD) are added by MAMMOTH

**3. Translation** (`mammoth/translate/translation.py`):
- Decoding adapted to use tokenizer's built-in decoder
- Properly handles BPE merging and special token removal
- Maintains compatibility with traditional vocab files

**4. Configuration** (`mammoth/opts.py`):
- New `--use_hf_tokenizer` flag to enable HF tokenizer mode
- Backward compatible: omit flag to use traditional spm model and vocabs

## Tokenizer Formats

### Sentencepiece-style (Default Example)

Uses `▁` (U+2581 LOWER ONE EIGHTH BLOCK) to mark word boundaries:

```python
# Input:  "Hello, world!"
# Tokens: ['▁Hello', ',', '▁world', '!']
# IDs:    [1234, 45, 5678, 90]
```
Note: punctuations are not pretokenized.

### Other Supported Formats

You can customize the tokenizer training script to use:

- **WordPiece**: Used by BERT (`##` prefix for continuations)
- **Unigram**: SentencePiece unigram model
- **Character**: Character-level tokenization

See the [HuggingFace tokenizers documentation](https://huggingface.co/docs/tokenizers/index) for more options.


### Using Pretrained Tokenizers

Download and use tokenizers from HuggingFace hub:

```python
from tokenizers import Tokenizer

# Download from hub
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
tokenizer.save("my_tokenizer.json")
```

Then use `my_tokenizer.json` in your MAMMOTH config.
