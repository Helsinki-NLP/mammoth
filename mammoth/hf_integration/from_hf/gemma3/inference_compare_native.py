#!/usr/bin/env python3
"""
Compare the Mammoth (native-backend, fake-encoder) Gemma3 conversion
(convert_gemma3_native.py) against the original HuggingFace Gemma3 model.

Unlike inference_compare.py (for the true decoder-only converter), this
model has a real (but numerically inert) encoder: convert_gemma3_native.py
zero-initializes cross_attn.to_out, so any src input produces the exact same
decoder output as Gemma3 alone. Forward calls here pass a fixed random `src`
tensor for that reason -- its content never affects the output.

Two modes:
  * No --mammoth-checkpoint: runs convert_gemma3_native.convert() in-memory
    (fresh weights copied from the HF model each time, nothing persisted).
  * --mammoth-checkpoint <prefix>: loads an ON-DISK checkpoint previously
    written by `convert_gemma3_native.py <hf_model_path> <prefix>`, so you
    can also validate that a saved checkpoint round-trips correctly.

Usage:
    python inference_compare_native.py hf_models/gemma3_270m
    python inference_compare_native.py hf_models/gemma3_270m \\
        --mammoth-checkpoint converted/mammoth_gemma_native/model
"""
import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from mammoth.model_builder import build_model  # noqa: E402
from mammoth.utils.model_saver import load_frame_checkpoint, load_parameters_from_checkpoint  # noqa: E402

from convert_gemma3_native import (  # noqa: E402
    DEFAULT_DECODER_GROUP,
    DEFAULT_ENCODER_GROUP,
    DEFAULT_LANG,
    build_task_queue_manager,
    convert,
)


DEFAULT_SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Mammoth is a multilingual machine translation framework.",
    "2 + 2 =",
]


def load_mammoth_model(hf_model_path, mammoth_checkpoint, task_id, encoder_group, decoder_group,
                        src_lang, tgt_lang, enc_layers, enc_model_dim, weight=1.0):
    if mammoth_checkpoint is None:
        # In-memory conversion: same path exercised by
        # tests/test_gemma3_native_conversion.py, nothing written to disk.
        mammoth_model, _, tqm = convert(
            hf_model_path, save_path=None, enc_layers=enc_layers, enc_model_dim=enc_model_dim,
            task_id=task_id, encoder_group=encoder_group, decoder_group=decoder_group,
            src_lang=src_lang, tgt_lang=tgt_lang, weight=weight,
        )
        mammoth_model.eval()
        return mammoth_model, tqm

    frame, frame_checkpoint_path = load_frame_checkpoint(mammoth_checkpoint)
    if frame is None:
        raise FileNotFoundError(f"No checkpoint frame found at prefix {mammoth_checkpoint!r}")

    model_opts = frame["opts"]
    vocabs_dict = frame["vocab"]

    if task_id is None:
        task_id = f"{src_lang}-{tgt_lang}"
    tqm = build_task_queue_manager(
        model_opts, vocabs_dict, task_id, encoder_group, decoder_group, src_lang, tgt_lang, weight=weight,
    )

    mammoth_model = build_model(model_opts, model_opts, vocabs_dict, task_queue_manager=tqm, single_task=None)
    load_parameters_from_checkpoint(
        frame_checkpoint_path,
        mammoth_model,
        optim=None,
        task_queue_manager=tqm,
        reset_optim=True,
        yes_i_messed_with_the_checkpoint=False,
    )
    mammoth_model.eval()
    return mammoth_model, tqm


@torch.no_grad()
def greedy_generate_mammoth(mammoth_model, tqm, src, src_mask, input_ids, max_new_tokens, eos_token_id):
    metadata = tqm.get_my_tasks()[0].get_serializable_metadata()
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits, _ = mammoth_model(src, generated, src_mask, metadata=metadata)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break
    return generated


def compare(hf_model_path, mammoth_checkpoint, sentences, max_new_tokens,
            task_id, encoder_group, decoder_group, src_lang, tgt_lang,
            enc_layers, enc_model_dim):
    tok = AutoTokenizer.from_pretrained(hf_model_path, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, torch_dtype=torch.float32, local_files_only=True,
    ).eval()

    mammoth_model, tqm = load_mammoth_model(
        hf_model_path, mammoth_checkpoint, task_id, encoder_group, decoder_group,
        src_lang, tgt_lang, enc_layers, enc_model_dim,
    )
    metadata = tqm.get_my_tasks()[0].get_serializable_metadata()

    # Fake-encoder input: content is irrelevant (cross_attn.to_out is zero),
    # fixed here just so repeated forward calls in the generation loop are
    # deterministic and cheap to construct.
    src = torch.randint(0, 100, (1, 4))
    src_mask = torch.ones(1, 4).bool()

    for text in sentences:
        print("=" * 80)
        print(f"PROMPT: {text!r}")
        ids = tok(text, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            hf_logits = hf_model(input_ids=ids).logits
            mammoth_logits, _ = mammoth_model(src, ids, src_mask, metadata=metadata)

        max_abs_diff = (hf_logits - mammoth_logits).abs().max().item()
        top1_match = (hf_logits.argmax(dim=-1) == mammoth_logits.argmax(dim=-1)).float().mean().item()
        print(f"  teacher-forced logits: max_abs_diff={max_abs_diff:.6g}  top1_match_rate={top1_match:.4f}")

        hf_out = hf_model.generate(
            input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
        )
        mammoth_out = greedy_generate_mammoth(
            mammoth_model, tqm, src, src_mask, ids, max_new_tokens, eos_token_id=tok.eos_token_id,
        )

        hf_new_ids = hf_out[0, ids.shape[1]:].tolist()
        mammoth_new_ids = mammoth_out[0, ids.shape[1]:].tolist()
        ids_match = hf_new_ids == mammoth_new_ids

        print(f"  HF greedy:      {tok.decode(hf_new_ids, skip_special_tokens=True)!r}")
        print(f"  Mammoth greedy: {tok.decode(mammoth_new_ids, skip_special_tokens=True)!r}")
        print(f"  generated token ids match: {ids_match}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("hf_model_path", help="Path to the original HF Gemma3 model (e.g. hf_models/gemma3_270m)")
    parser.add_argument("--mammoth-checkpoint", default=None,
                         help="Checkpoint prefix from convert_gemma3_native.py. "
                              "If omitted, converts in-memory instead (no checkpoint needed).")
    parser.add_argument("--sentences", nargs="+", default=None, help="Prompts to compare (default: built-in examples)")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--enc-layers", type=int, default=1, help="Depth of the fake encoder (in-memory mode only)")
    parser.add_argument("--enc-model-dim", type=int, default=64, help="Hidden dim of the fake encoder (in-memory mode only)")
    parser.add_argument("--task-id", default=None, help="Must match the task-id used at conversion time (default: '<src-lang>-<tgt-lang>')")
    parser.add_argument("--encoder-group", default=DEFAULT_ENCODER_GROUP)
    parser.add_argument("--decoder-group", default=DEFAULT_DECODER_GROUP)
    parser.add_argument("--src-lang", default=DEFAULT_LANG)
    parser.add_argument("--tgt-lang", default=DEFAULT_LANG)
    args = parser.parse_args()

    compare(
        args.hf_model_path,
        args.mammoth_checkpoint,
        sentences=args.sentences or DEFAULT_SENTENCES,
        max_new_tokens=args.max_new_tokens,
        task_id=args.task_id,
        encoder_group=args.encoder_group,
        decoder_group=args.decoder_group,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        enc_layers=args.enc_layers,
        enc_model_dim=args.enc_model_dim,
    )


if __name__ == "__main__":
    main()
