"""
Inference script for a Mammoth model converted to HuggingFace format.

This file is shipped inside every converted artifact, so it can be fetched
without the mammoth repo:

    hf download <repo_id> inference.py --local-dir .

Install:

    pip install transformers torch huggingface_hub safetensors

Usage (--model-dir takes a HF Hub repo id or a local directory):

    # Single-task model:
    python inference.py --model-dir ./my_model
    python inference.py --model-dir ./my_model --device cuda --num-beams 4

    # Download a Hub repo into a plain local directory (instead of the hidden HF
    # cache), then reuse that directory on later runs:
    python inference.py --model-dir org/my-model --local-dir ./my_model
    python inference.py --model-dir ./my_model

    # Multi-task bundle (--single-artifact):
    python inference.py --model-dir ./bundled_model --task eng-spa

    # Modular repo (--modular): trained task, or any combination of trained paths
    python inference.py --model-dir ./modular_model --task eng-spa
    python inference.py --model-dir ./modular_model --src deu --tgt fra
    python inference.py --model-dir ./modular_model --src deu --tgt fra \
        --encoder deu all --decoder fra all

Use from Python (inference.py must be in the current directory or on sys.path):

    from inference import load_model, translate

    src_tok, tgt_tok, model = load_model("org/my-model")                    # standalone
    src_tok, tgt_tok, model = load_model("org/my-bundle", task="eng-spa")   # multi-task
    src_tok, tgt_tok, model = load_model("org/my-modular", src="deu", tgt="fra",
                                         encoder=["deu", "all"], decoder=["fra", "all"])
    print(translate(src_tok, tgt_tok, model, ["Hello!"], num_beams=4, max_new_tokens=128))

load_model returns a standard MammothForConditionalGeneration, so model.generate()
works as usual if you need more control.

Why two tokenizers? Mammoth trains with a separate vocabulary per language.
Source token IDs index into the source vocab; decoder output IDs index into
the target vocab. Using the source tokenizer to decode output produces garbage.

Why `trust_remote_code=True`? Mammoth ships its config and model classes as
sibling .py files in the model directory rather than upstream in
`transformers`. The flag tells HF to import them.
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, PreTrainedTokenizerFast


DEFAULT_MODEL_DIR = "./converted_model"
# config.json model_type of multi-task repos loaded through MammothHub:
# per-task bundle (--single-artifact) and component repo (--modular).
BUNDLE_MODEL_TYPES = ("mammoth_hub", "mammoth_modular")
DEFAULT_SENTENCES = [
    "Helsinki is the capital of Finland.",
    "Beijing is the capital of China",
]


def _is_bundle(model_dir: str) -> bool:
    """Check if the model directory/repo is a multi-task bundle."""
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        return cfg.get("model_type") in BUNDLE_MODEL_TYPES
    # HF Hub repo id: fetch config.json remotely
    if not os.path.exists(model_dir):
        try:
            from huggingface_hub import hf_hub_download
            _cfg_path = hf_hub_download(model_dir, "config.json")
            with open(_cfg_path) as f:
                cfg = json.load(f)
            return cfg.get("model_type") in BUNDLE_MODEL_TYPES
        except Exception:
            return False
    return False


def load_single(model_dir: str, device: str, local_dir: str | None = None):
    """Load a single-task converted model."""
    if local_dir and not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(repo_id=model_dir, local_dir=local_dir)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    if os.path.isdir(model_dir):
        src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(model_dir, config.src_tokenizer_dir))
        tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            os.path.join(model_dir, config.tgt_tokenizer_dir))
    else:
        # HF Hub repo id: use subfolder= instead of os.path.join
        src_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_dir, subfolder=config.src_tokenizer_dir)
        tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_dir, subfolder=config.tgt_tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    model.to(device).eval()
    return src_tokenizer, tgt_tokenizer, model


def _load_mammoth_hub(model_id: str):
    """Import MammothHub, bootstrapping from the Hub if neither local copy is available."""
    try:
        from mammoth_hub import MammothHub  # vendored in the artifact or on sys.path
        return MammothHub
    except ImportError:
        pass
    try:
        from mammoth.hf_integration.to_hf.mammoth_hub import MammothHub  # dev install
        return MammothHub
    except ImportError:
        pass
    # Clean environment: fetch mammoth_hub.py directly from the Hub repo
    from huggingface_hub import hf_hub_download
    import importlib.util
    _path = hf_hub_download(model_id, "mammoth_hub.py")
    _spec = importlib.util.spec_from_file_location("mammoth_hub", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    return _mod.MammothHub


def load_bundle(model_dir: str, task: str | None, device: str,
                local_dir: str | None = None, **components):
    """Load a model from a multi-task repo: a trained task, or (modular repos)
    a src/tgt/encoder/decoder combination."""
    MammothHub = _load_mammoth_hub(model_dir)
    if local_dir:
        components["local_dir"] = local_dir
    model = MammothHub.from_pretrained(model_dir, task=task, device=device, **components)
    return model.src_tokenizer, model.tgt_tokenizer, model


def load_model(model_dir: str, task: str | None = None, src: str | None = None,
               tgt: str | None = None, encoder: list[str] | None = None,
               decoder: list[str] | None = None, device: str = "cpu",
               local_dir: str | None = None):
    """Load any Mammoth HF repo (standalone, single-artifact or modular).

    Returns (src_tokenizer, tgt_tokenizer, model). Multi-task repos need `task`,
    or `src` and `tgt` (plus optional `encoder`/`decoder`) for modular repos.
    `local_dir`: download Hub files into this directory instead of the HF cache
    (multi-task repos: only the files the selected model needs). Pass it as
    `model_dir` on later runs to load offline.
    """
    if not _is_bundle(model_dir):
        return load_single(model_dir, device, local_dir=local_dir)
    if not task and not (src and tgt):
        raise ValueError(
            "This is a multi-task repo: specify a task (e.g. eng-spa), "
            "or a source and target language for a modular repo.")
    return load_bundle(model_dir, task, device, local_dir=local_dir, src=src, tgt=tgt,
                       encoder=encoder, decoder=decoder)


@torch.inference_mode()
def translate(src_tokenizer, tgt_tokenizer, model, sentences: list[str],
              num_beams: int = 4, max_new_tokens: int = 128) -> list[str]:
    inputs = src_tokenizer(sentences, return_tensors="pt", padding=True).to(model.device)
    inputs.pop("token_type_ids", None)

    gen_kwargs = dict(num_beams=num_beams, max_new_tokens=max_new_tokens)
    if num_beams > 1:
        gen_kwargs["early_stopping"] = True
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
    parser.add_argument("--src", default=None,
                        help="Modular repos: source language (alternative to --task)")
    parser.add_argument("--tgt", default=None,
                        help="Modular repos: target language (alternative to --task)")
    parser.add_argument("--encoder", nargs="+", default=None,
                        help="Modular repos: encoder xcoder_id per layer stack, e.g. eng all")
    parser.add_argument("--decoder", nargs="+", default=None,
                        help="Modular repos: decoder xcoder_id per layer stack, e.g. spa all")
    parser.add_argument("--local-dir", default=None,
                        help="When --model-dir is a HF Hub repo id: download the files "
                             "into this directory instead of the HF cache, so they are easy "
                             "to find and can be reused with --model-dir <local-dir>. "
                             "Multi-task repos: only the selected task's files are downloaded.")
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

    try:
        src_tokenizer, tgt_tokenizer, model = load_model(
            args.model_dir, task=args.task, src=args.src, tgt=args.tgt,
            encoder=args.encoder, decoder=args.decoder, device=args.device,
            local_dir=args.local_dir)
    except ValueError as e:
        parser.error(str(e))
    if args.local_dir and not os.path.isdir(args.model_dir):
        print(f"Model files saved to {args.local_dir}; "
              f"next time run with --model-dir {args.local_dir}", file=sys.stderr)

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
