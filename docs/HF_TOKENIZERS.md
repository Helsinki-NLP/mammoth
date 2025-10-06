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

Use the provided example script to train a MARIAN-style BPE tokenizer:

```bash
cd examples/hf_tokenizers
python train.py
```

This creates a `tokenizer.json` file in the `tokenizer_output/` directory.

**Key Configuration:**
- **Vocabulary size**: 64,000 tokens (configurable via `VOCAB_SIZE`)
- **Model type**: BPE (Byte Pair Encoding)
- **Special tokens**: `</s>`, `<pad>`, `<s>`, `<unk>`, `<mask>`
- **Pre-tokenizer**: Metaspace (uses `▁` for word boundaries, MARIAN-style)

### 2. Configure MAMMOTH to Use the Tokenizer

Add the `--use_hf_tokenizer` flag to your training configuration:

```yaml
# In your training config YAML
use_hf_tokenizer: true
```

### 3. Train Your Model

Run training as usual:

```bash
cd mammoth
python train.py -config your_config.yaml
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
- The tokenizer performs **ONLY** subword tokenization
- Special tokens (BOS/EOS) are added by MAMMOTH during data loading
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
- Backward compatible: omit flag to use traditional vocabs

## Tokenizer Formats

### MARIAN-style (Default Example)

Uses `▁` (U+2581 LOWER ONE EIGHTH BLOCK) to mark word boundaries:

```python
# Input:  "Hello, world!"
# Tokens: ['▁Hello', ',', '▁world', '!']
# IDs:    [1234, 45, 5678, 90]
```

**Advantages:**
- Preserves word boundaries in subword tokens
- No extra spaces in decoded output
- Common in many multilingual models

### Other Supported Formats

You can customize the tokenizer training script to use:

- **WordPiece**: Used by BERT (`##` prefix for continuations)
- **Unigram**: SentencePiece unigram model
- **Character**: Character-level tokenization

See the [HuggingFace tokenizers documentation](https://huggingface.co/docs/tokenizers/index) for more options.

## Advanced Usage

### Using Pretrained Tokenizers

Download and use tokenizers from HuggingFace hub:

```python
from tokenizers import Tokenizer

# Download from hub
tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
tokenizer.save("my_tokenizer.json")
```

Then use `my_tokenizer.json` in your MAMMOTH config.


## Example: Train Tokenizer for English-Finnish

```bash
# 1. Prepare combined corpus
cat train.en train.fi > combined_bilingual.txt

# 2. Train tokenizer
cd examples/hf_tokenizers
python train.py  # Uses combined_bilingual.txt by default

# 3. Configure MAMMOTH
# In train.yaml:
use_hf_tokenizer: true

# 4. Train model
cd mammoth
python train.py -config train.yaml
```

