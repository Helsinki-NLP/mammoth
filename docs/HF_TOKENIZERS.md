# HuggingFace Tokenizers Integration

This document describes how to use HuggingFace tokenizers library with MAMMOTH for training from scratch.

## Overview

MAMMOTH now supports using HuggingFace's `tokenizers` library as an alternative to traditional vocabulary files. This enables:

- **Modern tokenization algorithms**: BPE, WordPiece, Unigram, and more
- **Faster tokenization**: HuggingFace tokenizers are implemented in Rust for speed
- **Pretrained tokenizers**: Use tokenizers from HuggingFace model hub
- **Custom tokenizer training**: Train your own tokenizers with flexible configuration

## Installation

Install the HuggingFace tokenizers library:

```bash
pip install tokenizers
```

Or using uv (recommended for this project):

```bash
uv pip install tokenizers
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

tasks:
  task_id_1:
    path_src: /path/to/tokenizer_output/tokenizer.json
    path_tgt: /path/to/tokenizer_output/tokenizer.json
    # ... other task configuration
```

Or via command line:

```bash
python train.py \
  -config train.yaml \
  -use_hf_tokenizer \
  -tasks task_id_1 path_src /path/to/tokenizer.json
```

### 3. Train Your Model

Run training as usual:

```bash
cd mammoth
python train.py -config your_config.yaml
```

MAMMOTH will automatically detect the `.json` extension and load it as a HuggingFace tokenizer.

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

### Custom Tokenizer Configuration

Modify `examples/hf_tokenizers/train.py` to customize:

- **Vocabulary size**: Change `VOCAB_SIZE` variable
- **Special tokens**: Modify `special_tokens` list in `BpeTrainer`
- **Pre-tokenizer**: Use different pre-tokenizers (Whitespace, ByteLevel, etc.)
- **Training parameters**: min_frequency, show_progress, etc.

Example for larger vocabulary:

```python
VOCAB_SIZE = 128000  # Increase to 128k tokens
```

### Multilingual Tokenizers

Train a single tokenizer on combined multilingual data:

```python
# Combine data from multiple languages
INPUT_FILE = "combined_multilingual.txt"  # Contains text from all languages

# Train as usual
tokenizer.train([INPUT_FILE], trainer)
```

Then use the same tokenizer for all language pairs:

```yaml
tasks:
  en_to_fi:
    path_src: tokenizer.json  # Same tokenizer
    path_tgt: tokenizer.json  # for both sides
  fi_to_en:
    path_src: tokenizer.json
    path_tgt: tokenizer.json
```

## Troubleshooting

### Import Error: tokenizers not found

```
ImportError: No module named 'tokenizers'
```

**Solution**: Install the tokenizers library:
```bash
uv pip install tokenizers
```

### Decoding Issues: Extra Spaces or Missing Punctuation

**Problem**: Decoded text has extra spaces before punctuation:
```
"Hello , world !"  # Wrong
"Hello, world!"    # Correct
```

**Solution**: Ensure you're using the tokenizer's `.decode()` method, not manually joining tokens:

```python
# ❌ Wrong - manual joining
text = ' '.join(tokens)

# ✓ Correct - use tokenizer's decoder
text = tokenizer.decode(token_ids)
```

The decoder handles metaspace markers (`▁`) correctly.

### Special Token Mismatch

**Problem**: Error about missing special tokens like `<s>`, `</s>`, `<pad>`

**Solution**: Ensure your tokenizer has all required special tokens:
```python
special_tokens = ["</s>", "<pad>", "<s>", "<unk>", "<mask>"]
```

These must match MAMMOTH's expected special tokens.

### File Extension Detection

**Problem**: MAMMOTH doesn't recognize my tokenizer file

**Solution**: Ensure your tokenizer file has `.json` extension:
```bash
# Correct
tokenizer.json

# Wrong (will be treated as traditional vocab)
tokenizer.txt
```

Or explicitly enable HF tokenizer mode:
```yaml
use_hf_tokenizer: true
```

## Performance Considerations

### Speed

HuggingFace tokenizers are implemented in Rust and are typically **faster** than Python-based tokenization, especially for large corpora.

### Memory

HF tokenizers are memory-efficient:
- Vocabulary stored in efficient data structures
- Minimal overhead compared to traditional vocab files
- Suitable for very large vocabularies (100k+ tokens)

### Disk Space

Tokenizer files (`.json`) are typically:
- Smaller than traditional vocab files for large vocabularies
- Self-contained (no separate merges file needed for BPE)

## Comparison: HF Tokenizers vs. Traditional Vocabs

| Feature | HF Tokenizers | Traditional Vocabs |
|---------|---------------|-------------------|
| **Format** | `.json` (single file) | `.txt` (vocab) + merges file |
| **Speed** | Fast (Rust implementation) | Moderate (Python) |
| **Algorithms** | BPE, WordPiece, Unigram, etc. | Fixed vocab lookup |
| **Pretrained** | Available on HF hub | Custom only |
| **Training** | Built-in trainers | External tools |
| **Flexibility** | Highly configurable | Limited |
| **Compatibility** | Requires `tokenizers` library | Pure Python |

## Examples

### Example 1: Train BPE Tokenizer for English-Finnish

```bash
# 1. Prepare combined corpus
cat train.en train.fi > combined_bilingual.txt

# 2. Train tokenizer
cd examples/hf_tokenizers
python train.py  # Uses combined_bilingual.txt by default

# 3. Configure MAMMOTH
# In train.yaml:
use_hf_tokenizer: true
tasks:
  en_to_fi:
    path_src: examples/hf_tokenizers/tokenizer_output/tokenizer.json
    path_tgt: examples/hf_tokenizers/tokenizer_output/tokenizer.json

# 4. Train model
cd mammoth
python train.py -config train.yaml
```

### Example 2: Use Different Tokenizers for Source/Target

```yaml
# Separate tokenizers for source and target languages
use_hf_tokenizer: true
tasks:
  en_to_fi:
    path_src: tokenizers/en_tokenizer.json
    path_tgt: tokenizers/fi_tokenizer.json
```

### Example 3: Mixed Mode (HF + Traditional)

```yaml
# Use HF tokenizer for source, traditional vocab for target
use_hf_tokenizer: true  # Enables HF tokenizer detection
tasks:
  en_to_fi:
    path_src: tokenizers/en_tokenizer.json  # HF tokenizer (auto-detected)
    path_tgt: vocabs/fi_vocab.txt           # Traditional vocab
```

MAMMOTH automatically detects the format based on file extension.

## References

- [HuggingFace Tokenizers Documentation](https://huggingface.co/docs/tokenizers/index)
- [BPE (Byte Pair Encoding) Paper](https://arxiv.org/abs/1508.07909)
- [SentencePiece: A simple and language independent approach to subword tokenization](https://arxiv.org/abs/1808.06226)
- [MARIAN NMT Framework](https://marian-nmt.github.io/)

## Support

For issues or questions about HuggingFace tokenizers integration:

1. Check this documentation first
2. Review the example script: `examples/hf_tokenizers/train.py`
3. Open an issue on the MAMMOTH GitHub repository
