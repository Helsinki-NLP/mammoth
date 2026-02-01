"""
Indexed dataset for pre-tokenized data.

This module provides efficient storage and retrieval of pre-tokenized sequences
using memory-mapped binary files. Inspired by Megatron-LM's indexed_dataset.

File format:
- .bin file: Raw token IDs stored as numpy array
- .idx file: Index with offsets and lengths for each sequence

This allows:
- Fast random access without loading entire dataset
- Efficient memory usage via memory mapping
- No tokenization overhead during training
"""

import numpy as np
import struct
import os


# Data type codes
DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.int16,
    4: np.int32,
    5: np.int64,
    6: np.float32,
    7: np.float64,
    8: np.uint16,
}

# Reverse mapping
CODE_TO_DTYPE = {v: k for k, v in DTYPES.items()}


def dtype_code(dtype):
    """Get code for a numpy dtype."""
    if dtype in CODE_TO_DTYPE:
        return CODE_TO_DTYPE[dtype]
    raise ValueError(f"Unsupported dtype: {dtype}")


def optimal_dtype(vocab_size):
    """
    Return the smallest dtype that can represent vocab_size.

    Args:
        vocab_size: Size of vocabulary

    Returns:
        numpy dtype
    """
    if vocab_size < 2**8:
        return np.uint8
    elif vocab_size < 2**16:
        return np.uint16
    elif vocab_size < 2**31:
        return np.int32
    else:
        return np.int64


class IndexedDatasetBuilder:
    """
    Builder for creating indexed datasets.

    Usage:
        builder = IndexedDatasetBuilder(output_bin_file, dtype=np.int32)
        builder.add_document(token_ids, sentence_lens)
        ...
        builder.finalize(output_idx_file)
    """

    def __init__(self, bin_path, dtype=np.int32):
        """
        Initialize builder.

        Args:
            bin_path: Path to output .bin file
            dtype: Data type for token IDs (default: int32)
        """
        self.bin_path = bin_path
        self.dtype = dtype
        self.dtype_code = dtype_code(dtype)

        # Track document boundaries and sentence lengths
        self.doc_idx = [0]  # Offsets for each document
        self.sentence_lens = []  # Lengths of sentences in each document

        # Open binary file for writing
        self.bin_file = open(bin_path, 'wb')
        self.total_tokens = 0

    def add_document(self, token_ids, sent_lens=None):
        """
        Add a document to the dataset.

        Args:
            token_ids: List/array of token IDs for the document
            sent_lens: Optional list of sentence lengths within the document
        """
        if len(token_ids) == 0:
            return

        # Convert to numpy array and write
        tokens = np.array(token_ids, dtype=self.dtype)
        self.bin_file.write(tokens.tobytes())

        # Update metadata
        self.total_tokens += len(token_ids)
        self.doc_idx.append(self.total_tokens)

        # Store sentence lengths if provided
        if sent_lens is not None:
            self.sentence_lens.append(sent_lens)
        else:
            # Single sentence per document
            self.sentence_lens.append([len(token_ids)])

    def finalize(self, idx_path):
        """
        Finalize the dataset and write index file.

        Args:
            idx_path: Path to output .idx file
        """
        self.bin_file.close()

        # Write index file
        with open(idx_path, 'wb') as f:
            # Magic string
            f.write(b'MMTHIDX\x00\x00')

            # Version
            f.write(struct.pack('<Q', 1))

            # Data type code
            f.write(struct.pack('<B', self.dtype_code))

            # Number of documents
            f.write(struct.pack('<Q', len(self.doc_idx) - 1))

            # Total number of tokens
            f.write(struct.pack('<Q', self.total_tokens))

            # Document offsets
            for offset in self.doc_idx:
                f.write(struct.pack('<Q', offset))

            # Sentence lengths for each document
            for sent_lens in self.sentence_lens:
                # Number of sentences in this document
                f.write(struct.pack('<Q', len(sent_lens)))
                # Length of each sentence
                for length in sent_lens:
                    f.write(struct.pack('<Q', length))


class IndexedDataset:
    """
    Read-only indexed dataset backed by memory-mapped binary files.

    Usage:
        dataset = IndexedDataset(path_prefix)
        tokens = dataset[idx]  # Get token IDs for document idx
    """

    def __init__(self, path_prefix):
        """
        Initialize dataset.

        Args:
            path_prefix: Path prefix (without .bin/.idx extension)
        """
        self.path_prefix = path_prefix
        self.idx_path = path_prefix + '.idx'
        self.bin_path = path_prefix + '.bin'

        # Read index
        self._read_index()

        # Memory map the binary file
        self.bin_data = np.memmap(
            self.bin_path,
            mode='r',
            dtype=self.dtype,
        )

    def _read_index(self):
        """Read and parse the index file."""
        with open(self.idx_path, 'rb') as f:
            # Read magic
            magic = f.read(9)
            if magic != b'MMTHIDX\x00\x00':
                raise ValueError(f"Invalid index file: {self.idx_path}")

            # Read version
            version = struct.unpack('<Q', f.read(8))[0]
            if version != 1:
                raise ValueError(f"Unsupported index version: {version}")

            # Read dtype code
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self.dtype = DTYPES[dtype_code]

            # Read counts
            self.num_docs = struct.unpack('<Q', f.read(8))[0]
            self.total_tokens = struct.unpack('<Q', f.read(8))[0]

            # Read document offsets
            self.doc_idx = []
            for _ in range(self.num_docs + 1):
                offset = struct.unpack('<Q', f.read(8))[0]
                self.doc_idx.append(offset)

            # Read sentence lengths
            self.sentence_lens = []
            for _ in range(self.num_docs):
                num_sentences = struct.unpack('<Q', f.read(8))[0]
                sent_lens = []
                for _ in range(num_sentences):
                    length = struct.unpack('<Q', f.read(8))[0]
                    sent_lens.append(length)
                self.sentence_lens.append(sent_lens)

    def __len__(self):
        """Return number of documents."""
        return self.num_docs

    def __getitem__(self, idx):
        """
        Get token IDs for a document.

        Args:
            idx: Document index

        Returns:
            numpy array of token IDs
        """
        if idx < 0 or idx >= self.num_docs:
            raise IndexError(f"Index {idx} out of range [0, {self.num_docs})")

        start = self.doc_idx[idx]
        end = self.doc_idx[idx + 1]
        return self.bin_data[start:end]

    def get_sentence_lengths(self, idx):
        """
        Get sentence lengths for a document.

        Args:
            idx: Document index

        Returns:
            List of sentence lengths
        """
        return self.sentence_lens[idx]

    def __del__(self):
        """Cleanup memory map."""
        if hasattr(self, 'bin_data'):
            del self.bin_data


def exists(path_prefix):
    """
    Check if indexed dataset exists.

    Args:
        path_prefix: Path prefix (without extension)

    Returns:
        True if both .bin and .idx files exist
    """
    return os.path.exists(path_prefix + '.bin') and os.path.exists(path_prefix + '.idx')


class _PreprocessEncoder:
    """
    Encoder class for tokenizing text in worker processes.

    This class is used for multiprocessing-based tokenization.
    """

    def __init__(self, vocab_path, lang, decoder_start_with_eos):
        self.vocab_path = vocab_path
        self.lang = lang
        self.decoder_start_with_eos = decoder_start_with_eos

    def initializer(self):
        """Initialize tokenizer in worker process."""
        from mammoth.inputters.vocab import get_vocab

        use_hf = self.vocab_path.endswith('.json')
        _PreprocessEncoder.vocab = get_vocab(
            path=self.vocab_path,
            lang=self.lang,
            size=None,
            use_hf_tokenizer=use_hf,
            decoder_start_with_eos=self.decoder_start_with_eos,
        )

        if hasattr(_PreprocessEncoder.vocab, 'tokenizer'):
            _PreprocessEncoder.tokenizer = _PreprocessEncoder.vocab.tokenizer
        else:
            _PreprocessEncoder.tokenizer = None

    def encode(self, line):
        """Encode a single line of text."""
        line = line.strip()
        if not line:
            return [], len(line)

        text = line

        if _PreprocessEncoder.tokenizer is not None:
            # HuggingFace tokenizer
            token_ids = _PreprocessEncoder.tokenizer.encode(text).ids
        else:
            # Traditional tokenizer (word-level)
            words = text.split()
            unk_id = _PreprocessEncoder.vocab.specials.get('<unk>', 0)
            token_ids = []
            for word in words:
                token_id = _PreprocessEncoder.vocab.stoi.get(word, unk_id)
                token_ids.append(token_id)

        return token_ids, len(line)


def preprocess_text_to_indexed(
    input_path,
    output_prefix,
    vocab_path,
    lang='',
    decoder_start_with_eos=False,
    workers=4,
    log_interval=1000
):
    """
    Preprocess text file to indexed dataset format.

    This is the programmatic API for preprocessing. The CLI wrapper
    is in mammoth/scripts/preprocess_indexed.py.

    Args:
        input_path: Path to input text file
        output_prefix: Path prefix for output files (without .bin/.idx)
        vocab_path: Path to vocabulary file
        lang: Language tag
        decoder_start_with_eos: Whether to use BART-style EOS
        workers: Number of worker processes
        log_interval: Log progress every N documents

    Returns:
        True if successful, False otherwise
    """
    import time
    import multiprocessing
    from mammoth.inputters.vocab import get_vocab
    from mammoth.utils.logging import logger

    try:
        # Validate inputs
        if not os.path.exists(input_path):
            logger.error(f"Input file not found: {input_path}")
            return False

        if not os.path.exists(vocab_path):
            logger.error(f"Vocabulary file not found: {vocab_path}")
            return False

        # Open input file
        logger.info(f"Opening input file: {input_path}")
        fin = open(input_path, 'r', encoding='utf-8')

        # Create encoder and multiprocessing pool
        encoder = _PreprocessEncoder(vocab_path, lang, decoder_start_with_eos)
        logger.info(f"Creating worker pool with {workers} workers")
        pool = multiprocessing.Pool(workers, initializer=encoder.initializer)

        # Process documents in parallel
        logger.info("Starting tokenization...")
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        # Load vocab to get size for optimal dtype
        use_hf = vocab_path.endswith('.json')
        vocab = get_vocab(
            path=vocab_path,
            lang=lang,
            size=None,
            use_hf_tokenizer=use_hf,
            decoder_start_with_eos=decoder_start_with_eos,
        )

        dtype = optimal_dtype(len(vocab))
        logger.info(f"Using dtype {dtype} for vocab size {len(vocab)}")

        output_bin = output_prefix + '.bin'
        output_idx = output_prefix + '.idx'

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_bin)) or '.', exist_ok=True)

        builder = IndexedDatasetBuilder(output_bin, dtype=dtype)

        # Process all documents
        start_time = time.time()
        total_bytes = 0
        total_docs = 0
        total_tokens = 0

        for i, (token_ids, bytes_processed) in enumerate(encoded_docs, start=1):
            if len(token_ids) == 0:
                continue

            total_bytes += bytes_processed
            total_docs += 1
            total_tokens += len(token_ids)

            # Add document to builder
            builder.add_document(token_ids)

            # Print statistics
            if i % log_interval == 0:
                elapsed = time.time() - start_time
                docs_per_sec = total_docs / elapsed if elapsed > 0 else 0
                mb_per_sec = (total_bytes / elapsed / 1024 / 1024) if elapsed > 0 else 0
                logger.info(
                    f"Processed {total_docs} documents "
                    f"({docs_per_sec:.1f} docs/s, {mb_per_sec:.1f} MB/s)"
                )

        # Finalize dataset
        logger.info("Finalizing dataset...")
        builder.finalize(output_idx)

        # Close files
        fin.close()
        pool.close()
        pool.join()

        # Final statistics
        elapsed = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Pretokenization complete!")
        logger.info(f"  Total documents: {total_docs}")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Average tokens/doc: {total_tokens/total_docs:.1f}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")
        logger.info(f"  Throughput: {total_docs/elapsed:.1f} docs/s")
        logger.info(f"  Output files:")
        logger.info(f"    {output_bin}")
        logger.info(f"    {output_idx}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Pretokenization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Clean up partial files
        try:
            output_bin = output_prefix + '.bin'
            output_idx = output_prefix + '.idx'
            if os.path.exists(output_bin):
                os.remove(output_bin)
            if os.path.exists(output_idx):
                os.remove(output_idx)
            logger.info("Cleaned up partial output files")
        except:
            pass

        return False
