"""
Inference template for a Mammoth model converted to HuggingFace format.

Supports two modes:

1. **Single-task model** (converted with --src/--tgt or a single-task checkpoint):

       python example_inference.py --model-dir ./my_model

2. **Multi-task bundle** (converted with --single-artifact):
   Use --task to pick which translation direction to run:

       python example_inference.py --model-dir ./bundled_model --task eng-spa

Mammoth uses separate source and target vocabularies, so two tokenizers are
stored under the model directory (src_tokenizer/ and tgt_tokenizer/).  The
subdirectory names are recorded in config.json under `src_tokenizer_dir` /
`tgt_tokenizer_dir`.

No `mammoth` package required for single-task models. Multi-task bundles need
the `mammoth` package (for `MammothHub`) — or use the vendored `mammoth_hub.py`
copied into the artifact, which supports both local paths and HF Hub repo ids.

Install:

    pip install transformers torch einops einx loguru packaging
    # optional, for faster attention on supported GPUs:
    pip install flash-attn

Minimal snippet — single-task (paste-and-go):

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

Minimal snippet — multi-task bundle:

    from mammoth_hub import MammothHub  # vendored in the repo — no mammoth package needed

    model = MammothHub.from_pretrained("your-org/your-bundled-model", task="eng-spa")
    # model is a standard MammothForConditionalGeneration — use tokenizers from bundle
    src_tokenizer = model.src_tokenizer   # (if available)
    tgt_tokenizer = model.tgt_tokenizer   # (if available)

    inputs = src_tokenizer(["Hello!"], return_tensors="pt", padding=True)
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

    # Single-task:
    python example_inference.py --model-dir ./my_model
    python example_inference.py --model-dir ./my_model --device cuda --num-beams 4

    # Multi-task bundle:
    python example_inference.py --model-dir ./bundled_model --task eng-spa
    python example_inference.py --model-dir ./bundled_model --task eng-fra --device cuda
"""

import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


DEFAULT_MODEL_DIR = "./converted_model"
DEFAULT_SENTENCES = [
    "Helsinki is the capital of Finland.",
    "Beijing is the capital of China",
]


def _is_bundle(model_dir: str) -> bool:
    """Check if the model directory is a multi-task bundle."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        return False
    with open(config_path) as f:
        cfg = json.load(f)
    return cfg.get("model_type") == "mammoth_hub"


def load_single(model_dir: str, device: str):
    """Load a single-task converted model."""
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(model_dir, config.src_tokenizer_dir))
    tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
        os.path.join(model_dir, config.tgt_tokenizer_dir))
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device).eval()
    return src_tokenizer, tgt_tokenizer, model


def load_bundle(model_dir: str, task: str, device: str):
    """Load a specific task from a multi-task bundled model."""
    try:
        from mammoth_hub import MammothHub  # vendored in the artifact
    except ImportError:
        from mammoth.hf_integration.to_hf.mammoth_hub import MammothHub  # dev fallback

    model = MammothHub.from_pretrained(model_dir, task=task, device=device)
    return model.src_tokenizer, model.tgt_tokenizer, model


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
    parser.add_argument("--task", default=None,
                        help="Task name for multi-task bundles (e.g. eng-spa). "
                             "Ignored for single-task models.")
    parser.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--sentences", nargs="+", default=None,
                        help="Sentences to translate (default: built-in examples)")
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

    # Auto-detect bundle vs single-task
    if _is_bundle(args.model_dir):
        if not args.task:
            # List available tasks
            with open(os.path.join(args.model_dir, "config.json")) as f:
                tasks = list(json.load(f).get("tasks", {}).keys())
            parser.error(
                f"This is a multi-task bundle. Please specify --task. "
                f"Available tasks: {tasks}"
            )
        src_tokenizer, tgt_tokenizer, model = load_bundle(
            args.model_dir, args.task, args.device)
    else:
        src_tokenizer, tgt_tokenizer, model = load_single(
            args.model_dir, args.device)

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
