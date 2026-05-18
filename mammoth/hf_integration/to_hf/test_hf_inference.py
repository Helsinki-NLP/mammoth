#!/usr/bin/env python3
"""
Test inference of a Mammoth model converted to HuggingFace format.

Usage:
    python test_hf_inference.py \
        --model-dir path/to/converted_hf_model \
        [--sentences "Hola mundo." "Buenos días."] \
        [--beam-size 4] \
        [--max-new-tokens 128] \
        [--device cpu]
"""

import argparse
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


DEFAULT_SENTENCES = [
    "Hola, ¿cómo estás?",
    "El gato está sobre la mesa.",
    "La reunión empezará a las diez de la mañana.",
]


def run_inference(model_dir: str, sentences: list[str], beam_size: int,
                  max_new_tokens: int, device: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    src_tok_dir = os.path.join(model_dir, config.src_tokenizer_dir)
    tgt_tok_dir = os.path.join(model_dir, config.tgt_tokenizer_dir)

    print(f"Loading src tokenizer from {src_tok_dir} ...")
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(src_tok_dir)
    print(f"  src vocab size: {len(src_tokenizer)}")

    print(f"Loading tgt tokenizer from {tgt_tok_dir} ...")
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(tgt_tok_dir)
    print(f"  tgt vocab size: {len(tgt_tokenizer)}")

    print(f"Loading model from {model_dir} ...")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    model = model.to(device)
    model.eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}  |  device: {device}")

    print(f"\nRunning inference (beam_size={beam_size}, max_new_tokens={max_new_tokens})\n")
    print("=" * 60)

    for src in sentences:
        inputs = src_tokenizer(src, return_tensors="pt").to(device)
        inputs.pop("token_type_ids", None)  # not used by Mammoth encoder

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                num_beams=beam_size,
                max_new_tokens=max_new_tokens,
                early_stopping=True,
                use_cache=False,
            )

        translation = tgt_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"SRC : {src}")
        print(f"HYP : {translation}")
        print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Test HF inference of a converted Mammoth model")
    parser.add_argument("--model-dir", required=True,
                        help="Directory produced by convert_mammoth_to_hf.py")
    parser.add_argument("--sentences", nargs="+", default=None,
                        help="Source sentences to translate (defaults to built-in Spanish examples)")
    parser.add_argument("--beam-size", type=int, default=4,
                        help="Beam search width (default: 4; use 1 for greedy)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Maximum tokens to generate (default: 128)")
    parser.add_argument("--device", default="cpu",
                        help="Device: 'cpu', 'cuda', 'mps' (default: cpu)")
    args = parser.parse_args()

    sentences = args.sentences if args.sentences else DEFAULT_SENTENCES

    try:
        run_inference(args.model_dir, sentences, args.beam_size, args.max_new_tokens, args.device)
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
