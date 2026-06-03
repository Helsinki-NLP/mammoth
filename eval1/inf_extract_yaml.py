#!/usr/bin/env python3
# inf_extract_yaml.py
#
# Purpose
# -------
# Extract one single-task inference configuration per Mammoth task from an
# existing multi-task training configuration.
#
# This script converts a training YAML into a set of smaller YAML files, one
# per task, suitable for inference-time use. It removes training-only options,
# keeps only the selected task, and restricts src_vocab / tgt_vocab to the
# source and target languages required by that task.
#
# Same-language autoencoder-style tasks are skipped.
#
# Typical use
# -----------
#   python inf_extract_yaml.py train.yaml out_dir
#
# Example:
#   python inf_extract_yaml.py train.yaml inf_yamls
#
# Main effects
# ------------
# For each task in train.yaml:
#   - copy the training config
#   - drop training-only top-level keys
#   - drop dataset path keys from the task block
#   - keep only that one task in the "tasks" section
#   - keep only the relevant source and target vocab entries
#   - write the resulting YAML to:
#         out_dir/<task_name>.yaml
#
# Inputs
# ------
# Positional arguments:
#   train_yaml
#       Path to the original Mammoth training YAML
#   out_dir
#       Directory where per-task inference YAMLs are written
#
# Optional arguments:
#   --beam-size
#   --batch-size
#   --batch-type
#   --gpu
#   --world-size
#   --gpu-ranks
#
# Reads
# -----
#   - the input training YAML
#
# Writes
# ------
#   - one YAML file per task into out_dir
#
# Output
# ------
#   - per-task inference YAML files
#   - progress messages on stdout
#
# Notes
# -----
# The task-level field "src_tgt" is preferred when inferring the source and
# target languages. If it is missing, the script falls back to parsing the
# task name prefix such as mt_, sentmt_, or docmt_.


import argparse
import copy
from pathlib import Path

import yaml

DROP_TOP_LEVEL_KEYS = [
    "early_stopping",
    "valid_timeout",
    "valid_decode_timeout",
    "valid_max_length",
    "tensorboard",
    "tensorboard_log_dir",
    "report_tflops",
    "denoising_objective",
    "mask_ratio",
    "mask_length",
    "poisson_lambda",
    "replace_length",
    "reset_optim",
    "src_seq_length_min",
    "tgt_seq_length_min",
    "src_seq_length_max",
    "tgt_seq_length_max",
    "train_steps",
    "accum_count",
    "lookahead_minibatches",
    "normalization",
    "queue_size",
    "valid_batch_size",
    "optim",
    "learning_rate",
    "adam_beta1",
    "adam_beta2",
    "weight_decay",
    "max_grad_norm",
    "label_smoothing",
    "warmup_steps",
    "decay_method",
    "learning_rate_decay",
    "start_decay_steps",
    "average_decay",
    "n_nodes",
    "task_distribution_strategy",
    "valid_steps",
    "valid_metrics",
    "save_checkpoint_steps",
    "keep_checkpoint",
    "log_model_structure",
    "report_every",
    "report_training_accuracy",
    "save_model",
    "save_strategy",
]

DROP_TASK_KEYS = [
    "path_src",
    "path_tgt",
    "path_valid_src",
    "path_valid_tgt",
]


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def infer_src_tgt_from_task(task_name: str, task_cfg: dict) -> tuple[str, str]:
    src_tgt = task_cfg.get("src_tgt")
    if src_tgt:
        src_lang, tgt_lang = src_tgt.split("-")
        return src_lang, tgt_lang
    for prefix in ("mt_", "sentmt_", "docmt_"):
        if task_name.startswith(prefix):
            src_tgt = task_name[len(prefix):]
            src_lang, tgt_lang = src_tgt.split("-")
            return src_lang, tgt_lang
    raise ValueError(f"Cannot infer src/tgt languages for task {task_name!r}")

def build_inference_config(
    train_cfg: dict,
    task_name: str,
    beam_size: int,
    batch_size: int,
    batch_type: str,
    gpu: int,
    world_size: int,
    gpu_ranks: list[int],
) -> dict:

    if "tasks" not in train_cfg:
        raise KeyError("train config has no 'tasks' section")
    if task_name not in train_cfg["tasks"]:
        raise KeyError(f"task {task_name!r} not found in training config")

    task_cfg = copy.deepcopy(train_cfg["tasks"][task_name])
    src_lang, tgt_lang = infer_src_tgt_from_task(task_name, task_cfg)
    if src_lang == tgt_lang:
        return None

    infer_cfg = copy.deepcopy(train_cfg)

    for key in DROP_TOP_LEVEL_KEYS:
        infer_cfg.pop(key, None)

    for key in DROP_TASK_KEYS:
        task_cfg.pop(key, None)

    task_cfg["transforms"] = [
        t for t in task_cfg.get("transforms", [])
        if t != "filtertoolong"
    ]
    task_cfg["transforms"] = []
    task_cfg["node_gpu"] = "0:0"

    infer_cfg["tasks"] = {task_name: task_cfg}

    src_vocab = infer_cfg.get("src_vocab", {})
    tgt_vocab = infer_cfg.get("tgt_vocab", {})

    if src_lang not in src_vocab:
        raise KeyError(f"source language {src_lang!r} missing from src_vocab")
    if tgt_lang not in tgt_vocab:
        raise KeyError(f"target language {tgt_lang!r} missing from tgt_vocab")

    infer_cfg["src_vocab"] = {src_lang: src_vocab[src_lang]}
    infer_cfg["tgt_vocab"] = {tgt_lang: tgt_vocab[tgt_lang]}

    infer_cfg["beam_size"] = beam_size
    infer_cfg["batch_size"] = batch_size
    infer_cfg["batch_type"] = batch_type
    infer_cfg["gpu"] = gpu
    infer_cfg["world_size"] = world_size
    infer_cfg["gpu_ranks"] = gpu_ranks

    return infer_cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one inference YAML per Mammoth task from a training YAML."
    )
    parser.add_argument("train_yaml", help="Path to train.yaml")
    parser.add_argument("out_dir", help="Output directory for per-task YAMLs")

    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-type", default="sents", choices=["sents", "tokens"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--gpu-ranks",
        type=int,
        nargs="+",
        default=[0],
        help="GPU ranks list, e.g. --gpu-ranks 0 or --gpu-ranks 0 1",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_cfg = load_yaml(args.train_yaml)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "tasks" not in train_cfg:
        raise KeyError("train config has no 'tasks' section")

    for task_name in train_cfg["tasks"]:
        infer_cfg = build_inference_config(
            train_cfg=train_cfg,
            task_name=task_name,
            beam_size=args.beam_size,
            batch_size=args.batch_size,
            batch_type=args.batch_type,
            gpu=args.gpu,
            world_size=args.world_size,
            gpu_ranks=args.gpu_ranks,
        )
        if infer_cfg == None:
            print(f"Skipping autoencoder task {task_name}")
        else:
            out_path = out_dir / f"{task_name}.yaml"
            dump_yaml(infer_cfg, out_path)
            print(f"Wrote {out_path}")
            

if __name__ == "__main__":
    main()

    
