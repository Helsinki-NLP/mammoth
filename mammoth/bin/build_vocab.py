import os
import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder


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
        help="Path(s) to the training text file(s). Can provide multiple files."
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


# Parse command line arguments
args = parse_args()
INPUT_FILE = args.input_file
OUTPUT_DIR = args.output_dir
VOCAB_SIZE = args.vocab_size

# Validate input files exist
for file_path in INPUT_FILE:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")

# Validate UTF-8 encoding
def validate_and_clean_if_needed(file_path, sample_size=10_000_000):
    """Validate UTF-8 and clean if necessary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Sample the file (first 10MB) to check encoding
            f.read(sample_size)
        print(f"✓ File is valid UTF-8: {file_path}")
        return file_path
    except UnicodeDecodeError as e:
        print(f"⚠️  Invalid UTF-8 detected at position {e.start}: {e.reason}")
        print(f"   Creating cleaned version...")

        # Create cleaned file
        cleaned_path = file_path + ".utf8_cleaned"
        with open(file_path, 'rb') as f_in:
            raw_data = f_in.read()

        # Replace invalid sequences with � (U+FFFD replacement character)
        cleaned_text = raw_data.decode('utf-8', errors='replace')

        with open(cleaned_path, 'w', encoding='utf-8') as f_out:
            f_out.write(cleaned_text)

        print(f"✓ Cleaned file saved to: {cleaned_path}")
        return cleaned_path

# Validate and potentially clean all input files
INPUT_FILES = []
for file_path in INPUT_FILE:
    validated_file = validate_and_clean_if_needed(file_path)
    INPUT_FILES.append(validated_file)

# Initialize tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="<unk>",byte_fallback=True))

# Configure trainer with correct special tokens (no duplicates)
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["</s>", "<pad>", "<s>", "<unk>", "<mask>"],
    show_progress=True
)

# Set pre-tokenizer
tokenizer.pre_tokenizer = Metaspace(replacement="▁")

# Train the tokenizer
print(f"Training MARIAN-style tokenizer on {len(INPUT_FILES)} file(s)...")
for i, file_path in enumerate(INPUT_FILES):
    print(f"  [{i+1}/{len(INPUT_FILES)}] {file_path}")
tokenizer.train(INPUT_FILES, trainer)

# Set decoder (MARIAN-style: properly handles ▁ markers during decoding)
tokenizer.decoder = MetaspaceDecoder(replacement="▁")

# NOTE: We DO NOT add a post-processor for BOS/EOS tokens here
# MAMMOTH handles special token insertion during data loading (dataset.py:143-145)
# The tokenizer's job is ONLY subword tokenization, not sequence formatting
# This gives MAMMOTH flexibility to experiment with different special token strategies

# Enable padding (needed for batching)
tokenizer.enable_padding(pad_id=tokenizer.token_to_id("<pad>"), pad_token="<pad>")

# Save the trained tokenizer
os.makedirs(OUTPUT_DIR, exist_ok=True)
tokenizer.save(os.path.join(OUTPUT_DIR, "tokenizer.json"))
print(f"Tokenizer saved to {OUTPUT_DIR}tokenizer.json")

# Also save in HuggingFace format for easy loading later
from transformers import PreTrainedTokenizerFast

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