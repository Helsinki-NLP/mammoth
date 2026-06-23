"""Greedy inference with a Mammoth model exported to TFLite via litert-torch.

The TFLite model is teacher-forced (full forward pass every step), so decoding
is O(n²) in output length.  That is expected for this export format.

Inputs baked in at conversion time:
    src_tokens     int32  [1, seq_len]   source token IDs, padded
    decoder_tokens int32  [1, tgt_len]   decoder token IDs so far, padded
    src_mask       bool   [1, seq_len]   True = real token, False = padding

Output:
    logits         float32 [1, tgt_len, vocab_size]

Usage:
    python mammoth/litert/infer.py \\
        --model   /path/to/eng_spa.tflite \\
        --src-tok /path/to/eng-tokenizer.json \\
        --tgt-tok /path/to/spa-tokenizer.json \\
        --text    "The cat sat on the mat."

    # or read sentences from a file, one per line:
    python mammoth/litert/infer.py \\
        --model   /path/to/eng_spa.tflite \\
        --src-tok /path/to/eng-tokenizer.json \\
        --tgt-tok /path/to/spa-tokenizer.json \\
        --input-file sentences.txt
"""

import argparse
import sys

import numpy as np


def _load_tokenizer(path: str):
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: 'tokenizers' package not found. Install with: pip install tokenizers", file=sys.stderr)
        sys.exit(1)
    return Tokenizer.from_file(path)


def _encode(tokenizer, text: str, seq_len: int):
    """Encode text to a padded int32 array and a bool mask.

    Returns:
        token_ids: np.int32 [1, seq_len]
        mask:      np.bool_ [1, seq_len]  True = real token
    """
    enc = tokenizer.encode(text)
    ids = enc.ids  # list[int], includes BOS/EOS added by the tokenizer template

    # Truncate to seq_len (keep BOS at start, EOS at end)
    if len(ids) > seq_len:
        # Keep as many tokens as fit; preserve EOS if present
        ids = ids[: seq_len - 1] + [ids[-1]]

    pad_id = tokenizer.token_to_id("<pad>") or 0
    n = len(ids)
    padded = np.full((1, seq_len), pad_id, dtype=np.int32)
    padded[0, :n] = ids
    mask = np.zeros((1, seq_len), dtype=np.bool_)
    mask[0, :n] = True
    return padded, mask


def _greedy_decode(runner, src_tokens, src_mask, bos_id: int, eos_id: int, pad_id: int, tgt_len: int):
    """Run autoregressive greedy decoding.

    Each step feeds all decoder tokens so far and reads the next-token logit
    from the last filled position.
    """
    decoder_tokens = np.full((1, tgt_len), pad_id, dtype=np.int32)
    decoder_tokens[0, 0] = bos_id

    generated = []
    for step in range(tgt_len - 1):
        outputs = runner(args_0=src_tokens, args_1=decoder_tokens, args_2=src_mask)
        # outputs is a dict; single output tensor is keyed 'output_0'
        logits = outputs["output_0"] if isinstance(outputs, dict) else outputs
        next_token = int(np.argmax(logits[0, step, :]))
        generated.append(next_token)
        decoder_tokens[0, step + 1] = next_token
        if next_token == eos_id:
            break

    # Strip the trailing EOS from the output
    if generated and generated[-1] == eos_id:
        generated = generated[:-1]
    return generated


def translate(model_path: str, src_tok_path: str, tgt_tok_path: str, text: str, seq_len: int = 128, tgt_len: int = 128) -> str:
    from ai_edge_litert import interpreter as litert_interp

    src_tok = _load_tokenizer(src_tok_path)
    tgt_tok = _load_tokenizer(tgt_tok_path)

    interp = litert_interp.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    runner = interp.get_signature_runner("serving_default")

    bos_id = tgt_tok.token_to_id("<s>") or 0
    eos_id = tgt_tok.token_to_id("</s>") or 1
    pad_id = tgt_tok.token_to_id("<pad>") or 0

    src_tokens, src_mask = _encode(src_tok, text, seq_len)
    token_ids = _greedy_decode(runner, src_tokens, src_mask, bos_id, eos_id, pad_id, tgt_len)
    return tgt_tok.decode(token_ids)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="Path to .tflite model file")
    parser.add_argument("--src-tok", required=True, help="Path to source tokenizer .json")
    parser.add_argument("--tgt-tok", required=True, help="Path to target tokenizer .json")

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", help="Single sentence to translate")
    input_group.add_argument("--input-file", help="File with one sentence per line (- for stdin)")

    parser.add_argument("--seq-len", type=int, default=128, help="Encoder sequence length used at conversion (default: 128)")
    parser.add_argument("--tgt-len", type=int, default=128, help="Decoder sequence length used at conversion (default: 128)")
    args = parser.parse_args()

    if args.text is None and args.input_file is None:
        parser.error("Provide --text or --input-file")

    if args.text:
        sentences = [args.text]
    else:
        fh = sys.stdin if args.input_file == "-" else open(args.input_file)
        sentences = [line.rstrip("\n") for line in fh if line.strip()]

    from ai_edge_litert import interpreter as litert_interp

    src_tok = _load_tokenizer(args.src_tok)
    tgt_tok = _load_tokenizer(args.tgt_tok)

    interp = litert_interp.Interpreter(model_path=args.model)
    interp.allocate_tensors()
    runner = interp.get_signature_runner("serving_default")

    bos_id = tgt_tok.token_to_id("<s>") or 0
    eos_id = tgt_tok.token_to_id("</s>") or 1
    pad_id = tgt_tok.token_to_id("<pad>") or 0

    for sentence in sentences:
        src_tokens, src_mask = _encode(src_tok, sentence, args.seq_len)
        token_ids = _greedy_decode(runner, src_tokens, src_mask, bos_id, eos_id, pad_id, args.tgt_len)
        print(tgt_tok.decode(token_ids))


if __name__ == "__main__":
    main()