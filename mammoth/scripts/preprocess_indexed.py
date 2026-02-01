#!/usr/bin/env python
"""
Preprocess text data into indexed dataset format.

This script tokenizes text data and creates memory-mapped binary files
for efficient data loading during training. Uses multiprocessing for
fast preprocessing of large datasets.

Usage:
    python -m mammoth.scripts.preprocess_indexed \\
        --input data.txt \\
        --output_prefix data/preprocessed \\
        --vocab vocab.txt \\
        --workers 8

Inspired by Megatron-LM's preprocess_data.py
"""

import argparse
import multiprocessing
import os
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mammoth.inputters.indexed_dataset import IndexedDatasetBuilder, optimal_dtype
from mammoth.inputters.vocab import get_vocab
from mammoth.utils.logging import init_logger, logger


class Encoder:
    """
    Encoder class for tokenizing text in worker processes.

    This class is used as a container for global data in worker processes.
    The initializer() method sets up class-level attributes that are shared
    across all workers.
    """

    def __init__(self, args):
        """
        Initialize encoder with arguments.

        Args:
            args: Command-line arguments
        """
        self.args = args

    def initializer(self):
        """
        Initialize tokenizer in worker process.

        This is called once per worker process to set up the tokenizer.
        Using class-level attributes avoids pickling the tokenizer for
        every function call.
        """
        # Load vocabulary/tokenizer
        use_hf = self.args.vocab_path.endswith('.json')
        Encoder.vocab = get_vocab(
            path=self.args.vocab_path,
            lang=self.args.lang,
            size=None,
            use_hf_tokenizer=use_hf,
            decoder_start_with_eos=self.args.decoder_start_with_eos,
        )

        # For HF tokenizers, we can use the tokenizer directly
        if hasattr(Encoder.vocab, 'tokenizer'):
            Encoder.tokenizer = Encoder.vocab.tokenizer
        else:
            Encoder.tokenizer = None

    def encode(self, line):
        """
        Encode a single line of text.

        Args:
            line: Text line to encode

        Returns:
            Tuple of (token_ids, num_bytes_processed)
        """
        line = line.strip()
        if not line:
            return [], len(line)

        # Use plain text input
        text = line

        # Tokenize
        if Encoder.tokenizer is not None:
            # HuggingFace tokenizer
            token_ids = Encoder.tokenizer.encode(text).ids
        else:
            # Traditional tokenizer (word-level)
            # This is a simple whitespace tokenizer - adapt as needed
            words = text.split()
            # Get UNK token ID for handling out-of-vocabulary words
            unk_id = Encoder.vocab.specials.get('<unk>', 0)
            token_ids = []
            for word in words:
                # Use get() with default UNK token for words not in vocabulary
                token_id = Encoder.vocab.stoi.get(word, unk_id)
                token_ids.append(token_id)

        return token_ids, len(line)


class Preprocessor:
    """
    Preprocessor class for managing multiprocessing workers.
    """

    def __init__(self, args, workers):
        """
        Initialize preprocessor.

        Args:
            args: Command-line arguments
            workers: Number of worker processes
        """
        self.args = args
        self.workers = workers

    def print_stats(self, count, start_time, total_bytes):
        """
        Print processing statistics.

        Args:
            count: Number of documents processed
            start_time: Processing start time
            total_bytes: Total bytes processed
        """
        # if count % self.args.log_interval == 0:
        #     elapsed = time.time() - start_time
        #     docs_per_sec = count / elapsed if elapsed > 0 else 0
        #     mb_per_sec = (total_bytes / elapsed / 1024 / 1024) if elapsed > 0 else 0
        #     logger.info(
        #         f"Processed {count} documents "
        #         f"({docs_per_sec:.1f} docs/s, {mb_per_sec:.1f} MB/s)"
        #     )

    def process_file(self):
        """Process input file and create indexed dataset."""
        logger.info(f"Opening input file: {self.args.input}")

        # Open input file
        fin = open(self.args.input, 'r', encoding='utf-8')

        # Create encoder and multiprocessing pool
        encoder = Encoder(self.args)
        logger.info(f"Creating worker pool with {self.workers} workers")
        pool = multiprocessing.Pool(
            self.workers,
            initializer=encoder.initializer
        )

        # Process documents in parallel
        # imap returns results in order, chunk size of 32 for efficiency
        logger.info("Starting tokenization...")
        encoded_docs = pool.imap(encoder.encode, fin, 32)

        # Create dataset builder
        # Load vocab to get size for optimal dtype
        init_logger()
        use_hf = self.args.vocab_path.endswith('.json')
        vocab = get_vocab(
            path=self.args.vocab_path,
            lang=self.args.lang,
            size=None,
            use_hf_tokenizer=use_hf,
            decoder_start_with_eos=self.args.decoder_start_with_eos,
        )

        dtype = optimal_dtype(len(vocab))
        logger.info(f"Using dtype {dtype} for vocab size {len(vocab)}")

        output_bin = self.args.output_prefix + '.bin'
        output_idx = self.args.output_prefix + '.idx'

        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_bin)), exist_ok=True)

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
            self.print_stats(i, start_time, total_bytes)

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
        logger.info(f"Processing complete!")
        logger.info(f"  Total documents: {total_docs}")
        logger.info(f"  Total tokens: {total_tokens}")
        logger.info(f"  Average tokens/doc: {total_tokens/total_docs:.1f}")
        logger.info(f"  Time elapsed: {elapsed:.1f}s")
        logger.info(f"  Throughput: {total_docs/elapsed:.1f} docs/s")
        logger.info(f"  Output files:")
        logger.info(f"    {output_bin}")
        logger.info(f"    {output_idx}")
        logger.info("=" * 60)


def get_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Preprocess text data into indexed dataset format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/output
    group = parser.add_argument_group('Input/Output')
    group.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input text file (one document per line)',
    )
    group.add_argument(
        '--output_prefix',
        type=str,
        required=True,
        help='Path prefix for output files (will create .bin and .idx)',
    )

    # Vocabulary
    group = parser.add_argument_group('Vocabulary')
    group.add_argument(
        '--vocab_path',
        type=str,
        required=True,
        help='Path to vocabulary file (.txt) or HF tokenizer (.json)',
    )
    group.add_argument(
        '--lang',
        type=str,
        default='',
        help='Language tag for logging',
    )
    group.add_argument(
        '--decoder_start_with_eos',
        action='store_true',
        help='BART-style: decoder sequences start with EOS token',
    )

    # Processing
    group = parser.add_argument_group('Processing')
    group.add_argument(
        '--workers',
        type=int,
        required=True,
        help='Number of worker processes for parallel tokenization',
    )
    group.add_argument(
        '--log_interval',
        type=int,
        default=1000,
        help='Log progress every N documents',
    )

    args = parser.parse_args()

    # Validation
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    if not os.path.exists(args.vocab_path):
        parser.error(f"Vocabulary file not found: {args.vocab_path}")

    return args


def main():
    """Main entry point."""
    args = get_args()

    # Initialize logging
    init_logger()

    logger.info("=" * 60)
    logger.info("MAMMOTH Indexed Dataset Preprocessor")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output prefix: {args.output_prefix}")
    logger.info(f"Vocabulary: {args.vocab_path}")
    logger.info(f"Workers: {args.workers}")
    logger.info("=" * 60)

    # Create preprocessor and process file
    preprocessor = Preprocessor(args, args.workers)
    preprocessor.process_file()


if __name__ == '__main__':
    main()
