"""
Inference template for a Mammoth model converted to HuggingFace format.

Mammoth uses separate source and target vocabularies, so two tokenizers are
stored under the model directory (src_tokenizer/ and tgt_tokenizer/).  The
subdirectory names are recorded in config.json under `src_tokenizer_dir` /
`tgt_tokenizer_dir`.

No `mammoth` package required — the x-transformers fork is vendored inside
the model directory.

Install:

    pip install transformers torch einops einx loguru packaging
    # optional, for faster attention on supported GPUs:
    pip install flash-attn

Minimal snippet (paste-and-go):

    import os
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast

    model_id = "your-org/your-mammoth-model"          # or a local path
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(model_id, config.src_tokenizer_dir))
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(model_id, config.tgt_tokenizer_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)

    inputs = src_tokenizer(["Hola, ¿cómo estás?"], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)                # not consumed by Mammoth encoder
    output_ids = model.generate(**inputs, num_beams=4, max_new_tokens=128)
    print(tgt_tokenizer.batch_decode(output_ids, skip_special_tokens=True))

Why two tokenizers? Mammoth trains with a separate vocabulary per language.
Source token IDs index into the source vocab; decoder output IDs index into
the target vocab. Using the source tokenizer to decode output produces garbage.

Why `trust_remote_code=True`? Mammoth ships its config and model classes as
sibling .py files in the model directory rather than upstream in
`transformers`. The flag tells HF to import them.

Why no `use_cache=...`? The converted config sets `use_cache=False` by default,
because this wrapper does not yet implement KV caching. Generation is correct
but O(T^2) in decode length. If you load an older checkpoint whose config does
not pin this, pass `use_cache=False` explicitly into `generate(...)`.

Run as a script:

    python example_inference.py
    python example_inference.py --model-dir ./my_model --device cuda --num-beams 4
"""

import argparse
import os

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


DEFAULT_MODEL_DIR = "./converted_model"
DEFAULT_SENTENCES = [
    "Helsinki is the capital of Finland.",
    "Beijing is the capital of China",
]


def load(model_dir: str, device: str):
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(model_dir, config.src_tokenizer_dir))
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(model_dir, config.tgt_tokenizer_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device).eval()
    return src_tokenizer, tgt_tokenizer, model


@torch.inference_mode()
def translate(src_tokenizer, tgt_tokenizer, model, sentences: list[str],
              num_beams: int, max_new_tokens: int) -> list[str]:
    inputs = src_tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    inputs.pop("token_type_ids", None)

    gen_kwargs = dict(num_beams=num_beams, max_new_tokens=max_new_tokens)
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
    if getattr(model.config, "use_cache", True):
        gen_kwargs["use_cache"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)
    return tgt_tokenizer.batch_decode(output_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="Path to the converted model directory or a HF Hub repo id")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sentences", nargs="+", default=None,
                        help="Sentences to translate (default: built-in Spanish examples)")
    parser.add_argument("--input-file", default=None,
                        help="Path to a text file with one source sentence per line")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Number of sentences per batch (default: 32)")
    parser.add_argument("--output-file", default=None,
                        help="Path to write translations (one per line); defaults to stdout")
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file) as f:
            sentences = [line.rstrip("\n") for line in f if line.strip()]
    else:
        sentences = args.sentences or DEFAULT_SENTENCES

    src_tokenizer, tgt_tokenizer, model = load(args.model_dir, args.device)

    out = open(args.output_file, "w") if args.output_file else None
    try:
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i : i + args.batch_size]
            translations = translate(
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                model=model,
                sentences=batch,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            for src, hyp in zip(batch, translations):
                if out:
                    out.write(hyp + "\n")
                else:
                    print(f"SRC: {src}")
                    print(f"HYP: {hyp}")
                    print("-" * 60)
    finally:
        if out:
            out.close()


if __name__ == "__main__":
    main()
