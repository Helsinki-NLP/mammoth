#!/usr/bin/env python3
"""
Compare a converted Mammoth (native-backend, decoder-only) Gemma3 checkpoint
against the original HuggingFace Gemma3 model it was converted from.

This loads the ON-DISK checkpoint written by convert_gemma3_decoder_only.py
(frame + per-component shards, e.g. `converted/mammoth_gemma_step_0_*.pt`) --
not an in-memory re-conversion -- so it also validates that the saved
checkpoint round-trips correctly.

Reuses build_task_queue_manager()/build_model_opts() from
convert_gemma3_decoder_only.py so the task/decoder-group/lang wiring exactly
matches what was used at conversion time; pass --task-id/--decoder-group/
--lang if you converted with non-default values.

Usage:
    python inference_compare.py hf_models/gemma3_270m converted/mammoth_gemma
    python inference_compare.py hf_models/gemma3_270m converted/mammoth_gemma \\
        --sentences "The quick brown fox" "2 + 2 =" --max-new-tokens 30
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

from convert_gemma3_decoder_only import (  # noqa: E402
    DEFAULT_DECODER_GROUP,
    DEFAULT_LANG,
    build_task_queue_manager,
)


DEFAULT_SENTENCES = [
    "What is the capital of Helsinki? What is the weather like?.",
    "Mammoth is a multilingual machine translation framework.",
    "2 + 2 =",
]


def load_mammoth_model(mammoth_checkpoint, task_id, decoder_group, lang, weight=1.0):
    """Rebuild the Mammoth model architecture and load weights from an
    on-disk checkpoint prefix (e.g. 'converted/mammoth_gemma')."""
    frame, frame_checkpoint_path = load_frame_checkpoint(mammoth_checkpoint)
    if frame is None:
        raise FileNotFoundError(f"No checkpoint frame found at prefix {mammoth_checkpoint!r}")

    model_opts = frame["opts"]
    vocabs_dict = frame["vocab"]

    if task_id is None:
        task_id = f"{lang}-lm"
    tqm = build_task_queue_manager(model_opts, vocabs_dict, task_id, decoder_group, lang, weight=weight)

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
def greedy_generate_mammoth(mammoth_model, tqm, input_ids, max_new_tokens, eos_token_id):
    metadata = tqm.get_my_tasks()[0].get_serializable_metadata()
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits, _ = mammoth_model(None, generated, None, metadata=metadata)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=1)
        if eos_token_id is not None and next_id.item() == eos_token_id:
            break
    return generated


def compare(hf_model_path, mammoth_checkpoint, sentences, max_new_tokens,
            task_id, decoder_group, lang):
    tok = AutoTokenizer.from_pretrained(hf_model_path, local_files_only=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path, dtype=torch.bfloat16, local_files_only=True,
    ).eval()

    mammoth_model, tqm = load_mammoth_model(mammoth_checkpoint, task_id, decoder_group, lang)
    metadata = tqm.get_my_tasks()[0].get_serializable_metadata()

    for text in sentences:
        print("=" * 80)
        print(f"PROMPT: {text!r}")
        ids = tok(text, return_tensors="pt")["input_ids"]

        with torch.no_grad():
            hf_logits = hf_model(input_ids=ids).logits
            mammoth_logits, _ = mammoth_model(None, ids, None, metadata=metadata)

        max_abs_diff = (hf_logits - mammoth_logits).abs().max().item()
        top1_match = (hf_logits.argmax(dim=-1) == mammoth_logits.argmax(dim=-1)).float().mean().item()
        print(f"  teacher-forced logits: max_abs_diff={max_abs_diff:.6g}  top1_match_rate={top1_match:.4f}")

        hf_out = hf_model.generate(
            input_ids=ids, max_new_tokens=max_new_tokens, do_sample=False,
        )
        mammoth_out = greedy_generate_mammoth(
            mammoth_model, tqm, ids, max_new_tokens, eos_token_id=tok.eos_token_id,
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
    parser.add_argument("mammoth_checkpoint", help="Checkpoint prefix from convert_gemma3_decoder_only.py (e.g. converted/mammoth_gemma)")
    parser.add_argument("--sentences", nargs="+", default=None, help="Prompts to compare (default: built-in examples)")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--task-id", default=None, help="Must match the task-id used at conversion time (default: '<lang>-lm')")
    parser.add_argument("--decoder-group", default=DEFAULT_DECODER_GROUP)
    parser.add_argument("--lang", default=DEFAULT_LANG)
    args = parser.parse_args()

    compare(
        args.hf_model_path,
        args.mammoth_checkpoint,
        sentences=args.sentences or DEFAULT_SENTENCES,
        max_new_tokens=args.max_new_tokens,
        task_id=args.task_id,
        decoder_group=args.decoder_group,
        lang=args.lang,
    )


if __name__ == "__main__":
    main()
