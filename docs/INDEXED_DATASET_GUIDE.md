# Indexed Dataset Guide

Pre-tokenized indexed datasets eliminate tokenization overhead during training, significantly improving data loading performance.

## Overview

Indexed datasets store pre-tokenized data in efficient binary format:
- **`.bin` file**: Memory-mapped array of token IDs
- **`.idx` file**: Index with document boundaries and metadata

Benefits:
- **Fast loading**: No tokenization during training
- **Memory efficient**: Uses memory mapping
- **Deterministic**: Same tokenization across runs

## Workflow

### 1. Preprocess Your Data

Tokenize your text data once and save to indexed format:

```bash
python -m mammoth.scripts.preprocess_indexed \
    --input data/train.en.txt \
    --output_prefix data/train.en \
    --vocab_path vocab.txt \
    --workers 8
```

**For HuggingFace tokenizers:**

```bash
python -m mammoth.scripts.preprocess_indexed \
    --input data/train.en.txt \
    --output_prefix data/train.en \
    --vocab_path tokenizer.json \
    --workers 8
```

**Options:**

- `--input`: Input text file (one document per line)
- `--output_prefix`: Output path prefix (creates `.bin` and `.idx` files)
- `--vocab_path`: Vocabulary file (`.txt`) or HF tokenizer (`.json`)
- `--workers`: Number of parallel worker processes
- `--append_eos`: Add EOS token to each document
- `--json_keys`: Extract specific keys from JSON input (e.g., `--json_keys text`)
- `--log_interval`: Progress logging frequency (default: 1000)

### 2. Configure Training

In your training config YAML, set `data_type: indexed`:

```yaml
tasks:
  my_corpus:
    # Enable indexed dataset
    data_type: indexed

    # Paths without .bin/.idx extensions
    path_src: data/train.src
    path_tgt: data/train.tgt
    path_valid_src: data/valid.src
    path_valid_tgt: data/valid.tgt

    # Language pair
    src_lang: en
    tgt_lang: de

    # Other task settings
    weight: 1
    introduce_at_training_step: 0

    # IMPORTANT: Do NOT specify transforms for indexed datasets!
    # Indexed data is already tokenized and transforms expect different data format.
    # transforms: []  # Leave empty or omit
```

### Important: Transforms Are Not Supported

**Indexed datasets do NOT support transforms during training.** This is a fundamental limitation:

- **Why:** Indexed datasets contain pre-tokenized token IDs (integers), while transforms expect token strings
- **What breaks:** Operations like `prefix` transform try to concatenate strings with tensors
- **Solution:** Apply all preprocessing (filtering, prefix tokens, denoising, etc.) in the preprocessing script

If you specify transforms in your config for indexed datasets:
- A warning will be logged
- Transforms will be **silently ignored**
- Data will be used as-is from the binary files

**To apply transforms:** Modify the preprocessing script or preprocess separately before creating indexed datasets.

### 3. Train

Run training as usual:

```bash
mammoth train -config your_config.yaml
```

## File Format

### Binary File (`.bin`)

Contiguous array of token IDs stored in numpy format. The dtype is automatically chosen based on vocabulary size:
- `uint8` for vocab size < 256
- `uint16` for vocab size < 65536
- `int32` for larger vocabularies

### Index File (`.idx`)

Binary index containing:
- Magic string: `MMTHIDX\x00\x00`
- Version: 1
- Data type code
- Number of documents
- Total tokens
- Document offsets (one per document + 1)
- Sentence lengths (for each document)

## Multiprocessing Implementation

The preprocessing script uses a Megatron-LM-style multiprocessing architecture:

```
Main Process
    ├── Create Encoder with tokenizer configuration
    ├── Create multiprocessing.Pool(workers)
    │   ├── Worker 1: encoder.initializer() → load tokenizer
    │   ├── Worker 2: encoder.initializer() → load tokenizer
    │   └── Worker N: encoder.initializer() → load tokenizer
    └── pool.imap(encoder.encode, input_lines, chunksize=32)
        → Parallel tokenization with ordered output
```

**Key design decisions:**

1. **Encoder class** contains tokenization logic
2. **initializer()** method sets up tokenizer in each worker (once per worker)
3. **encode()** method processes individual lines (called many times)
4. **imap()** returns results in input order with chunking for efficiency
5. Class-level attributes avoid pickling tokenizer for every function call

This architecture:
- Minimizes serialization overhead
- Enables true parallelism
- Maintains deterministic output order
- Provides progress tracking

## Example: Complete Workflow

```bash
# 1. Preprocess training data
python -m mammoth.scripts.preprocess_indexed \
    --input data/train.en.txt \
    --output_prefix data/preprocessed/train.en \
    --vocab_path vocab.en.txt \
    --workers 16 \
    --append_eos

python -m mammoth.scripts.preprocess_indexed \
    --input data/train.de.txt \
    --output_prefix data/preprocessed/train.de \
    --vocab_path vocab.de.txt \
    --workers 16 \
    --append_eos

# 2. Preprocess validation data
python -m mammoth.scripts.preprocess_indexed \
    --input data/valid.en.txt \
    --output_prefix data/preprocessed/valid.en \
    --vocab_path vocab.en.txt \
    --workers 8 \
    --append_eos

python -m mammoth.scripts.preprocess_indexed \
    --input data/valid.de.txt \
    --output_prefix data/preprocessed/valid.de \
    --vocab_path vocab.de.txt \
    --workers 8 \
    --append_eos

# 3. Train with indexed datasets
mammoth train -config indexed_train.yaml
```

## Comparison with Text Datasets

| Feature | Text Dataset | Indexed Dataset |
|---------|-------------|-----------------|
| Tokenization | Every epoch | Once (preprocessing) |
| I/O | Read text files | Memory-mapped binary |
| Memory | Text + tokens | Efficient mmap |
| Startup | Fast | Slightly slower (loading index) |
| Training speed | Slower (tokenization overhead) | Faster |
| Flexibility | Easy to modify | Need to reprocess |

## When to Use Indexed Datasets

**Use indexed datasets when:**
- Training for many epochs
- Using large vocabularies (e.g., HF tokenizers with 50k+ tokens)
- Data loading is a bottleneck
- Running on systems with fast storage

**Use text datasets when:**
- Rapidly iterating on data
- Data changes frequently
- Using simple tokenization (whitespace)
- Training for very few epochs

## Troubleshooting

**"Indexed dataset not found" error:**
- Ensure you ran preprocessing for both train and validation data
- Check that paths in config match preprocessing output_prefix
- Verify both `.bin` and `.idx` files exist

**Out of memory:**
- Indexed datasets use memory mapping, not loading data into RAM
- Check if index file is corrupted (try reprocessing)
- Reduce batch size if model memory is the issue

**Slow preprocessing:**
- Increase `--workers` (typically 2x CPU cores)
- Use local storage instead of network filesystems
- Preprocess different files in parallel

**Different results vs text dataset:**
- Verify vocabulary files match
- Check if special tokens (BOS/EOS) are handled consistently
- Compare with `--append_eos` flag

**"Transforms warning" or unexpected behavior:**
- Indexed datasets do NOT support transforms during training
- Remove all transforms from task configuration
- Apply preprocessing (filtering, prefix, etc.) in the preprocessing script instead
- See "Important: Transforms Are Not Supported" section above

## Implementation Details

The indexed dataset implementation consists of three main components:

1. **`mammoth/inputters/indexed_dataset.py`**: Binary format I/O
   - `IndexedDatasetBuilder`: Writes binary files during preprocessing
   - `IndexedDataset`: Reads binary files using memory mapping

2. **`mammoth/scripts/preprocess_indexed.py`**: Preprocessing script
   - `Encoder`: Tokenization logic for worker processes
   - `Preprocessor`: Manages multiprocessing pool and statistics

3. **`mammoth/inputters/indexed_corpus.py`**: Training integration
   - `IndexedCorpus`: IterableDataset for training
   - **Note:** Transforms are NOT applied (data is already preprocessed)

All components follow the Megatron-LM design patterns for consistency with other large-scale training frameworks.
