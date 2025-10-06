import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Metaspace
from tokenizers.decoders import Metaspace as MetaspaceDecoder

# Configuration
INPUT_FILE = "combined_bilingual.txt"
OUTPUT_DIR = "tokenizer_output"
VOCAB_SIZE = 64000

# Validate input file exists
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Training file not found: {INPUT_FILE}")

# Initialize tokenizer with BPE model
tokenizer = Tokenizer(BPE(unk_token="<unk>"))

# Configure trainer with correct special tokens (no duplicates)
trainer = BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["</s>", "<pad>", "<s>", "<unk>", "<mask>"],
    show_progress=True
)

# Set pre-tokenizer (MARIAN-style: uses ▁ for word boundaries)
tokenizer.pre_tokenizer = Metaspace(replacement="▁")

# Train the tokenizer
print(f"Training MARIAN-style tokenizer on {INPUT_FILE}...")
tokenizer.train([INPUT_FILE], trainer)

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
print(f"Tokenizer saved to {OUTPUT_DIR}/tokenizer.json")

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