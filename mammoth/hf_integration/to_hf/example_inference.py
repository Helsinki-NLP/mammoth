"""
Inference template for a Mammoth model converted to HuggingFace format.

Follows the standard `transformers` model-card pattern: AutoTokenizer +
AutoModelForSeq2SeqLM + model.generate. No `mammoth` package required —
the x-transformers fork is vendored inside the model directory.

Install:

    pip install transformers torch einops einx loguru packaging
    # optional, for faster attention on supported GPUs:
    pip install flash-attn

Minimal snippet (paste-and-go):

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    model_id = "your-org/your-mammoth-model"          # or a local path
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True)

    inputs = tokenizer(["Hola, ¿cómo estás?"], return_tensors="pt", padding=True)
    inputs.pop("token_type_ids", None)                # not consumed by Mammoth encoder
    output_ids = model.generate(**inputs, num_beams=4, max_new_tokens=128)
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True))

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

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DEFAULT_MODEL_DIR = "./converted_model"
DEFAULT_SENTENCES = [
    "Hola, ¿cómo estás?",
    "El gato está sobre la mesa.",
    "La reunión empezará a las diez de la mañana.",
]


def load(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device).eval()
    return tokenizer, model


@torch.inference_mode()
def translate(tokenizer, model, sentences: list[str], num_beams: int,
              max_new_tokens: int) -> list[str]:
    inputs = tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    inputs.pop("token_type_ids", None)

    gen_kwargs = dict(num_beams=num_beams, max_new_tokens=max_new_tokens)
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
    # Belt-and-braces for older checkpoints whose config.json predates the
    # use_cache=False default; harmless if the config already pins it.
    if getattr(model.config, "use_cache", True):
        gen_kwargs["use_cache"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)


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

    tokenizer, model = load(args.model_dir, args.device)

    out = open(args.output_file, "w") if args.output_file else None
    try:
        for i in range(0, len(sentences), args.batch_size):
            batch = sentences[i : i + args.batch_size]
            translations = translate(
                tokenizer=tokenizer,
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
