#!/usr/bin/env python3
"""
Length-based filtering for parallel corpora before pretokenization.

This script filters parallel text files based on tokenized sequence length,
removing examples that are too short or too long for the model's positional
encoding limitations.

Since indexed datasets don't support on-the-fly transforms, this filtering
must be done during preprocessing rather than training.
"""

import argparse
import gzip
from pathlib import Path
from typing import Optional, Tuple

from tokenizers import Tokenizer


def open_file(filepath: str, mode: str = 'rt'):
    """Open text or gzip file for reading/writing."""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, mode, encoding='utf-8')
    else:
        return open(filepath, mode, encoding='utf-8')


def filter_parallel_corpus(
    src_input: str,
    tgt_input: str,
    src_output: str,
    tgt_output: str,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Optional[Tokenizer] = None,
    min_length: int = 1,
    max_length: int = 512,
    log_interval: int = 10000,
):
    """
    Filter parallel corpus by tokenized sequence length.

    Args:
        src_input: Source input file path (.txt or .gz)
        tgt_input: Target input file path (.txt or .gz)
        src_output: Source output file path
        tgt_output: Target output file path
        src_tokenizer: HuggingFace tokenizer for source
        tgt_tokenizer: HuggingFace tokenizer for target (if None, uses src_tokenizer)
        min_length: Minimum sequence length (inclusive)
        max_length: Maximum sequence length (inclusive)
        log_interval: Print progress every N lines
    """
    # Use shared tokenizer if only one provided
    if tgt_tokenizer is None:
        tgt_tokenizer = src_tokenizer
        print(f"Using shared tokenizer for both source and target")
    else:
        print(f"Using separate tokenizers for source and target")

    print(f"Source input: {src_input}")
    print(f"Target input: {tgt_input}")
    print(f"Source output: {src_output}")
    print(f"Target output: {tgt_output}")
    print(f"Length filter: [{min_length}, {max_length}]")
    print()

    total_lines = 0
    kept_lines = 0
    too_short = 0
    too_long = 0

    with open_file(src_input, 'rt') as src_in, \
         open_file(tgt_input, 'rt') as tgt_in, \
         open(src_output, 'w', encoding='utf-8') as src_out, \
         open(tgt_output, 'w', encoding='utf-8') as tgt_out:

        for src_line, tgt_line in zip(src_in, tgt_in):
            total_lines += 1

            # Tokenize to get lengths
            src_text = src_line.strip()
            tgt_text = tgt_line.strip()

            if not src_text or not tgt_text:
                # Skip empty lines
                too_short += 1
                continue

            # Get token counts
            src_encoding = src_tokenizer.encode(src_text)
            tgt_encoding = tgt_tokenizer.encode(tgt_text)

            src_len = len(src_encoding.ids)
            tgt_len = len(tgt_encoding.ids)

            # Apply filter
            if src_len < min_length or tgt_len < min_length:
                too_short += 1
                continue

            if src_len > max_length or tgt_len > max_length - 2:
                too_long += 1
                continue

            # Keep this example
            src_out.write(src_line)  # Keep original line (with newline)
            tgt_out.write(tgt_line)
            kept_lines += 1

            # Log progress
            if total_lines % log_interval == 0:
                kept_pct = 100.0 * kept_lines / total_lines
                print(f"Processed {total_lines:,} lines | "
                      f"Kept: {kept_lines:,} ({kept_pct:.1f}%) | "
                      f"Too short: {too_short:,} | "
                      f"Too long: {too_long:,}")

    # Final statistics
    print()
    print("=" * 70)
    print("Filtering complete!")
    print(f"Total lines processed: {total_lines:,}")
    print(f"Lines kept: {kept_lines:,} ({100.0 * kept_lines / total_lines:.2f}%)")
    print(f"Lines too short (< {min_length}): {too_short:,} ({100.0 * too_short / total_lines:.2f}%)")
    print(f"Lines too long (> {max_length}): {too_long:,} ({100.0 * too_long / total_lines:.2f}%)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Filter parallel corpus by tokenized sequence length',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using shared tokenizer
  python filter_by_length.py \\
    --src_input train.en.txt \\
    --tgt_input train.de.txt \\
    --src_output train.en.filtered.txt \\
    --tgt_output train.de.filtered.txt \\
    --tokenizer tokenizer.json \\
    --min_length 1 \\
    --max_length 512

  # Using separate tokenizers
  python filter_by_length.py \\
    --src_input train.en.txt.gz \\
    --tgt_input train.ar.txt.gz \\
    --src_output train.en.filtered.txt \\
    --tgt_output train.ar.filtered.txt \\
    --src_tokenizer tokenizer_en.json \\
    --tgt_tokenizer tokenizer_ar.json \\
    --min_length 5 \\
    --max_length 256
        """
    )

    # Input/output files
    parser.add_argument(
        '--src_input',
        type=str,
        required=True,
        help='Source input file (.txt or .gz)'
    )
    parser.add_argument(
        '--tgt_input',
        type=str,
        required=True,
        help='Target input file (.txt or .gz)'
    )
    parser.add_argument(
        '--src_output',
        type=str,
        required=True,
        help='Source output file (filtered)'
    )
    parser.add_argument(
        '--tgt_output',
        type=str,
        required=True,
        help='Target output file (filtered)'
    )

    # Tokenizers
    parser.add_argument(
        '--tokenizer',
        type=str,
        help='HuggingFace tokenizer file (.json) for both source and target (shared vocab)'
    )
    parser.add_argument(
        '--src_tokenizer',
        type=str,
        help='HuggingFace tokenizer file (.json) for source (separate vocabs)'
    )
    parser.add_argument(
        '--tgt_tokenizer',
        type=str,
        help='HuggingFace tokenizer file (.json) for target (separate vocabs)'
    )

    # Filtering parameters
    parser.add_argument(
        '--min_length',
        type=int,
        default=1,
        help='Minimum sequence length (inclusive, default: 1)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length (inclusive, default: 512)'
    )
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10000,
        help='Print progress every N lines (default: 10000)'
    )

    args = parser.parse_args()

    # Validate tokenizer arguments
    if args.tokenizer:
        # Shared tokenizer
        if args.src_tokenizer or args.tgt_tokenizer:
            parser.error("Cannot specify both --tokenizer and --src_tokenizer/--tgt_tokenizer")
        src_tokenizer = Tokenizer.from_file(args.tokenizer)
        tgt_tokenizer = None  # Will use src_tokenizer
    else:
        # Separate tokenizers
        if not args.src_tokenizer:
            parser.error("Must specify either --tokenizer or --src_tokenizer")
        src_tokenizer = Tokenizer.from_file(args.src_tokenizer)
        tgt_tokenizer = Tokenizer.from_file(args.tgt_tokenizer) if args.tgt_tokenizer else None

    # Validate length arguments
    if args.min_length < 0:
        parser.error("--min_length must be >= 0")
    if args.max_length < args.min_length:
        parser.error("--max_length must be >= --min_length")

    # Create output directories if needed
    Path(args.src_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.tgt_output).parent.mkdir(parents=True, exist_ok=True)

    # Run filtering
    filter_parallel_corpus(
        src_input=args.src_input,
        tgt_input=args.tgt_input,
        src_output=args.src_output,
        tgt_output=args.tgt_output,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        min_length=args.min_length,
        max_length=args.max_length,
        log_interval=args.log_interval,
    )


if __name__ == '__main__':
    main()
