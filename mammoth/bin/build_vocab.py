import argparse
import itertools
import json
import os
from tokenizers import Regex, Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from mammoth.inputters.corpus_io import open_corpus_lines

SPECIAL_TOKENS = ["</s>", "<pad>", "<s>", "<unk>", "<mask>"]
# One token per byte value, used by BPE byte fallback for characters not in the vocab.
BYTE_TOKENS = [f"<0x{i:02X}>" for i in range(256)]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a tokenizer using HuggingFace tokenizers library"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to the training file(s): plain text (one example per line) or .jsonl / .jsonl.gz (only the \"text\" key is used). A .gz suffix means gzipped. Can provide multiple files."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./",
        help="Directory to save the trained tokenizer. Output to the current directory by default."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size for the tokenizer"
    )
    return parser.parse_args()


def _new_tokenizer():
    """MARIAN-style BPE (▁ word markers) that also handles paragraphs and rare characters.

    - Newline runs are split off before Metaspace, so they become their own tokens
      ("\n", "\n\n", ...) and are never glued to words; sentence-level data simply
      never uses them. \r\n and \r are normalized to \n first.
    - prepend_scheme="first" adds ▁ only at the start of the text, not after each
      newline, so decoding reproduces the text exactly.
    - The decoder turns ▁ back into spaces and byte tokens back into characters.
    """
    tokenizer = Tokenizer(BPE(unk_token="<unk>", byte_fallback=True))
    tokenizer.normalizer = normalizers.Replace(Regex(r"\r\n?"), "\n")
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\n+"), behavior="isolated"),
        pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="first"),
    ])
    tokenizer.decoder = decoders.Sequence([
        decoders.Replace("▁", " "),
        decoders.ByteFallback(),
        decoders.Fuse(),
        decoders.Strip(" ", 1, 0),
    ])
    return tokenizer


def _add_byte_tokens(tokenizer):
    """Append BYTE_TOKENS to the BPE vocab as regular (non-special) tokens.

    Passing them to the trainer as special tokens would make
    decode(skip_special_tokens=True), which MAMMOTH uses, silently drop them.
    """
    data = json.loads(tokenizer.to_str())
    vocab = data["model"]["vocab"]
    next_id = max(vocab.values()) + 1
    for tok in BYTE_TOKENS:
        if tok not in vocab:
            vocab[tok] = next_id
            next_id += 1
    return Tokenizer.from_str(json.dumps(data))


def train_tokenizer_from_texts(texts, vocab_size, show_progress=False):
    """Train the tokenizer on an iterable of example strings.

    `vocab_size` is the final size, including special and byte tokens.
    """
    reserved = len(SPECIAL_TOKENS) + len(BYTE_TOKENS)
    if vocab_size <= reserved:
        raise ValueError(f"vocab_size must be larger than {reserved} (special + byte tokens), got {vocab_size}")

    tokenizer = _new_tokenizer()
    trainer = BpeTrainer(
        vocab_size=vocab_size - len(BYTE_TOKENS),
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=["\n"],  # newline token exists even if the data has no paragraphs
        show_progress=show_progress
    )
    tokenizer.train_from_iterator(texts, trainer)
    return _add_byte_tokens(tokenizer)


def train_tokenizer(input_files, vocab_size, show_progress=False):
    """Train on corpus files, read the same way MAMMOTH reads training data.

    Plain text (.txt, .gz, ...) is read one example per line; .jsonl / .jsonl.gz
    contribute only their "text" values. Invalid UTF-8 is replaced with U+FFFD.
    """
    texts = itertools.chain.from_iterable(
        open_corpus_lines(path, errors='replace') for path in input_files
    )
    return train_tokenizer_from_texts(texts, vocab_size, show_progress=show_progress)


def main():
    # Parse command line arguments
    args = parse_args()
    INPUT_FILES = args.input_file
    OUTPUT_DIR = args.output_dir
    VOCAB_SIZE = args.vocab_size

    # Validate input files exist
    for file_path in INPUT_FILES:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file not found: {file_path}")

    # Train the tokenizer
    print(f"Training MARIAN-style tokenizer on {len(INPUT_FILES)} file(s)...")
    for i, file_path in enumerate(INPUT_FILES):
        print(f"  [{i+1}/{len(INPUT_FILES)}] {file_path}")
    tokenizer = train_tokenizer(INPUT_FILES, VOCAB_SIZE, show_progress=True)

    # NOTE: We DO NOT add a post-processor for BOS/EOS tokens here
    # MAMMOTH handles special token insertion during data loading (dataset.py:143-145)
    # The tokenizer's job is ONLY subword tokenization, not sequence formatting
    # This gives MAMMOTH flexibility to experiment with different special token strategies

    # Enable padding (needed for batching)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

    # Save the trained tokenizer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer.json")
    tokenizer.save(tokenizer_path)
    print(f"Tokenizer saved to {tokenizer_path}")

    # Also save in HuggingFace format for easy loading later
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    # Save HuggingFace tokenizer (creates tokenizer_config.json, special_tokens_map.json, etc.)
    hf_tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"HuggingFace tokenizer saved to {OUTPUT_DIR}")

    # Test the tokenizer
    print("\n" + "="*70)
    print("TESTING MARIAN-STYLE TOKENIZER")
    print("="*70)

    test_text = "Hello, world! This is a test."
    encoded = tokenizer.encode(test_text)

    print(f"\nOriginal text: '{test_text}'")
    print(f"Tokens: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")

    # Test decoding to verify punctuation handling
    decoded_text = tokenizer.decode(encoded.ids)
    print(f"Decoded text: '{decoded_text}'")
    print(f"Decoding matches original: {decoded_text == test_text}")

    # Demonstrate how ▁ markers work
    print("\n" + "-"*70)
    print("UNDERSTANDING ▁ MARKERS:")
    print("-"*70)
    for i, token in enumerate(encoded.tokens):
        has_marker = token.startswith("▁")
        marker_info = "← Word boundary" if has_marker else "← Continuation"
        print(f"  [{i}] '{token}' {marker_info}")

    print("\n" + "-"*70)
    print("PUNCTUATION SPACING TEST:")
    print("-"*70)
    # Test that comma and period don't get spaces before them
    test_cases = [
        "Hello, world!",
        "I like apples, oranges, and bananas.",
        "Dr. Smith said: \"Hello!\""
    ]

    for test in test_cases:
        encoded = tokenizer.encode(test)
        decoded = tokenizer.decode(encoded.ids)
        match = "✓" if decoded == test else "✗"
        print(f"{match} Original: '{test}'")
        print(f"   Decoded:  '{decoded}'")
        print()


if __name__ == "__main__":
    main()
