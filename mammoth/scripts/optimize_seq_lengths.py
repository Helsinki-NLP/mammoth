#!/usr/bin/env python3
"""Profile token length distributions across training tasks and produce
a report with recommended values for src_seq_length_max, tgt_seq_length_max,
max_length, and valid_max_length.

Usage:
    python -m mammoth.scripts.optimize_seq_lengths issues/83nodes/train.yaml

    # Custom percentile and sample limit
    python -m mammoth.scripts.optimize_seq_lengths train.yaml --percentile 95 --max-samples 100000
"""

import argparse
import gzip
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from tokenizers import Tokenizer


def parse_training_yaml(yaml_path: str) -> dict:
    """Parse a mammoth training YAML and return structured config dict."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def extract_lang_from_path(file_path: str) -> str:
    """Extract language code from a data file path.

    Examples:
        /data/eng-spa.eng.gz      -> 'eng'
        /data/train.srp_Cyrl      -> 'srp_Cyrl'
        /data/train.123.fin.gz    -> 'fin'
        /data/flores200/fra_Latn.dev -> 'fra_Latn'
    """
    name = Path(file_path).name
    for ext in (".gz", ".dev", ".test"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name.rsplit(".", 1)[-1]


def resolve_tokenizer(lang: str, vocab_dict: dict) -> str | None:
    """Look up tokenizer path for a language from src_vocab or tgt_vocab.

    Tries the full lang code first (e.g. 'fra_Latn'), then falls back to
    the base code before '_' (e.g. 'fra'), to handle flores200-style filenames.
    """
    if lang in vocab_dict:
        return vocab_dict[lang]
    base = lang.split("_")[0]
    return vocab_dict.get(base)


def profile_token_lengths(
    src_path: str,
    tgt_path: str,
    tok_src_path: str,
    tok_tgt_path: str | None = None,
    max_samples: int | None = None,
    tokenizer_cache: dict | None = None,
) -> dict:
    """Tokenize src/tgt files and return raw length arrays.

    Args:
        src_path: path to source file (.txt or .gz)
        tgt_path: path to target file (.txt or .gz)
        tok_src_path: path to source tokenizer.json
        tok_tgt_path: path to target tokenizer.json (defaults to src)
        max_samples: limit number of sentence pairs
        tokenizer_cache: optional dict of {path: Tokenizer} to reuse

    Returns:
        {"src_lengths": [int, ...], "tgt_lengths": [int, ...]}
    """
    if tok_tgt_path is None:
        tok_tgt_path = tok_src_path

    if tokenizer_cache is not None:
        if tok_src_path not in tokenizer_cache:
            tokenizer_cache[tok_src_path] = Tokenizer.from_file(tok_src_path)
        if tok_tgt_path not in tokenizer_cache:
            tokenizer_cache[tok_tgt_path] = Tokenizer.from_file(tok_tgt_path)
        tok_src = tokenizer_cache[tok_src_path]
        tok_tgt = tokenizer_cache[tok_tgt_path]
    else:
        tok_src = Tokenizer.from_file(tok_src_path)
        tok_tgt = Tokenizer.from_file(tok_tgt_path)

    src_open = gzip.open if src_path.endswith(".gz") else open
    tgt_open = gzip.open if tgt_path.endswith(".gz") else open

    src_lengths = []
    tgt_lengths = []

    with src_open(src_path, "rt") as f_src, tgt_open(tgt_path, "rt") as f_tgt:
        for i, (src_line, tgt_line) in enumerate(zip(f_src, f_tgt)):
            if max_samples and i >= max_samples:
                break
            src_lengths.append(len(tok_src.encode(src_line.strip()).ids))
            tgt_lengths.append(len(tok_tgt.encode(tgt_line.strip()).ids))

    return {"src_lengths": src_lengths, "tgt_lengths": tgt_lengths}


def _round_up_to_multiple(value: int, multiple: int = 64) -> int:
    """Round up to nearest multiple for GPU tensor alignment."""
    return math.ceil(value / multiple) * multiple


def compute_optimal_settings(
    task_profiles: dict[str, dict],
    percentile: int = 99,
) -> dict:
    """Compute optimal sequence length settings from per-task profiles.

    Takes the GLOBAL percentile across all tasks' max(src_len, tgt_len),
    then rounds up to the nearest 64 for GPU alignment.

    Args:
        task_profiles: {"task_name": {"src_lengths": [...], "tgt_lengths": [...]}}
        percentile: which percentile to use (default 99 = keep 99% of data)

    Returns:
        dict with src_seq_length_max, tgt_seq_length_max, max_length,
        valid_max_length, pct_data_kept, per_task_stats
    """
    all_max_of_both = []
    all_src = []
    all_tgt = []
    per_task_stats = {}

    for task_name, profile in task_profiles.items():
        src = np.array(profile["src_lengths"])
        tgt = np.array(profile["tgt_lengths"])
        max_of_both = np.maximum(src, tgt)

        all_src.extend(src.tolist())
        all_tgt.extend(tgt.tolist())
        all_max_of_both.extend(max_of_both.tolist())

        per_task_stats[task_name] = {
            "n_samples": len(src),
            "src_mean": float(np.mean(src)),
            "src_p50": float(np.percentile(src, 50)),
            "src_p99": float(np.percentile(src, 99)),
            "src_max": int(np.max(src)),
            "tgt_mean": float(np.mean(tgt)),
            "tgt_p50": float(np.percentile(tgt, 50)),
            "tgt_p99": float(np.percentile(tgt, 99)),
            "tgt_max": int(np.max(tgt)),
            "max_of_both_p99": float(np.percentile(max_of_both, 99)),
        }

    all_max_of_both = np.array(all_max_of_both)
    all_src = np.array(all_src)
    all_tgt = np.array(all_tgt)

    global_p = float(np.percentile(all_max_of_both, percentile))
    src_p = float(np.percentile(all_src, percentile))
    tgt_p = float(np.percentile(all_tgt, percentile))

    src_max = max(_round_up_to_multiple(int(math.ceil(src_p)), 64), 64)
    tgt_max = max(_round_up_to_multiple(int(math.ceil(tgt_p)), 64), 64)
    rounded_max = max(_round_up_to_multiple(int(math.ceil(global_p)), 64), 64)
    max_length = max(src_max, tgt_max, rounded_max)

    pct_kept = float(np.sum(all_max_of_both <= max_length) / len(all_max_of_both) * 100)

    return {
        "src_seq_length_max": src_max,
        "tgt_seq_length_max": tgt_max,
        "max_length": max_length,
        "valid_max_length": max_length,
        "pct_data_kept": pct_kept,
        "per_task_stats": per_task_stats,
    }


def compute_valid_max_length(
    valid_profiles: dict[str, dict],
    percentile: int = 99,
) -> dict:
    """Compute valid_max_length from validation set profiles.

    Args:
        valid_profiles: {"task_name": {"src_lengths": [...], "tgt_lengths": [...]}}
        percentile: which percentile to use

    Returns:
        dict with valid_max_length, pct_data_kept, per_task_stats
    """
    all_max_of_both = []
    per_task_stats = {}

    for task_name, profile in valid_profiles.items():
        src = np.array(profile["src_lengths"])
        tgt = np.array(profile["tgt_lengths"])
        max_of_both = np.maximum(src, tgt)
        all_max_of_both.extend(max_of_both.tolist())

        per_task_stats[task_name] = {
            "n_samples": len(src),
            "src_p50": float(np.percentile(src, 50)),
            "src_p99": float(np.percentile(src, 99)),
            "src_max": int(np.max(src)),
            "tgt_p50": float(np.percentile(tgt, 50)),
            "tgt_p99": float(np.percentile(tgt, 99)),
            "tgt_max": int(np.max(tgt)),
            "max_of_both_p99": float(np.percentile(max_of_both, 99)),
        }

    all_max_of_both = np.array(all_max_of_both)
    global_p = float(np.percentile(all_max_of_both, percentile))
    valid_max = max(_round_up_to_multiple(int(math.ceil(global_p)), 64), 64)
    pct_kept = float(np.sum(all_max_of_both <= valid_max) / len(all_max_of_both) * 100)

    return {
        "valid_max_length": valid_max,
        "pct_data_kept": pct_kept,
        "per_task_stats": per_task_stats,
    }


def print_report(config: dict, recommendations: dict, valid_recommendations: dict | None = None) -> None:
    """Print a human-readable report with per-task stats and recommendations."""
    per_task = recommendations["per_task_stats"]

    print("\n" + "=" * 72)
    print("TOKEN LENGTH PROFILING REPORT")
    print("=" * 72)

    print("\n--- Current settings ---")
    print(f"  src_seq_length_max: {config.get('src_seq_length_max', 'N/A')}")
    print(f"  tgt_seq_length_max: {config.get('tgt_seq_length_max', 'N/A')}")
    print(f"  max_length:          {config.get('max_length', 'N/A')}")
    print(f"  valid_max_length:    {config.get('valid_max_length', 'N/A')}")

    print("\n--- Training set per-task statistics ---")
    print(f"  {'Task':<35} {'N':>7} {'src P50':>8} {'src P99':>8} "
          f"{'tgt P50':>8} {'tgt P99':>8} {'max P99':>8}")
    print("  " + "-" * 90)

    for task_name, stats in per_task.items():
        print(f"  {task_name:<35} {stats['n_samples']:>7} "
              f"{stats['src_p50']:>8.0f} {stats['src_p99']:>8.0f} "
              f"{stats['tgt_p50']:>8.0f} {stats['tgt_p99']:>8.0f} "
              f"{stats['max_of_both_p99']:>8.0f}")

    if valid_recommendations:
        print("\n--- Validation set per-task statistics ---")
        print(f"  {'Task':<35} {'N':>7} {'src P50':>8} {'src P99':>8} "
              f"{'tgt P50':>8} {'tgt P99':>8} {'max P99':>8}")
        print("  " + "-" * 90)
        for task_name, stats in valid_recommendations["per_task_stats"].items():
            print(f"  {task_name:<35} {stats['n_samples']:>7} "
                  f"{stats['src_p50']:>8.0f} {stats['src_p99']:>8.0f} "
                  f"{stats['tgt_p50']:>8.0f} {stats['tgt_p99']:>8.0f} "
                  f"{stats['max_of_both_p99']:>8.0f}")

    valid_max_recommendation = (
        valid_recommendations["valid_max_length"]
        if valid_recommendations
        else recommendations["valid_max_length"]
    )

    print("\n--- Recommended settings (rounded to nearest 64) ---")
    print(f"  src_seq_length_max: {config.get('src_seq_length_max', 'N/A'):>6}"
          f"  ->  {recommendations['src_seq_length_max']}")
    print(f"  tgt_seq_length_max: {config.get('tgt_seq_length_max', 'N/A'):>6}"
          f"  ->  {recommendations['tgt_seq_length_max']}")
    print(f"  max_length:          {config.get('max_length', 'N/A'):>6}"
          f"  ->  {recommendations['max_length']}")
    print(f"  valid_max_length:    {config.get('valid_max_length', 'N/A'):>6}"
          f"  ->  {valid_max_recommendation}"
          + (" (from validation set)" if valid_recommendations else " (derived from training set)"))
    print(f"\n  Training data retained: {recommendations['pct_data_kept']:.2f}%")
    if valid_recommendations:
        print(f"  Validation data retained: {valid_recommendations['pct_data_kept']:.2f}%")

    old_max = config.get("max_length", 1024)
    new_max = recommendations["max_length"]
    if old_max > 0:
        waste_reduction = (1 - new_max / old_max) * 100
        print(f"  Padding waste reduction: ~{waste_reduction:.1f}% "
              f"(max_length {old_max} -> {new_max})")

    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Profile token lengths and produce optimization report.",
    )
    parser.add_argument("yaml_path", help="Path to mammoth training YAML")
    parser.add_argument(
        "--percentile", type=int, default=99,
        help="Percentile for optimal length (default: 99)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max sentence pairs to profile per task (default: all)",
    )
    args = parser.parse_args()

    config = parse_training_yaml(args.yaml_path)

    if "tasks" not in config:
        print("Error: No 'tasks' section found in YAML.", file=sys.stderr)
        sys.exit(1)

    src_vocab = config.get("src_vocab", {})
    tgt_vocab = config.get("tgt_vocab", {})

    tokenizer_cache = {}
    task_profiles = {}
    skipped = []

    valid_profiles = {}
    skipped_valid = []

    for task_name, task in config["tasks"].items():
        src_path = task.get("path_src", "")
        tgt_path = task.get("path_tgt", "")

        src_lang = extract_lang_from_path(src_path)
        tgt_lang = extract_lang_from_path(tgt_path)

        tok_src = resolve_tokenizer(src_lang, src_vocab)
        tok_tgt = resolve_tokenizer(tgt_lang, tgt_vocab)

        if not tok_src or not tok_tgt:
            skipped.append(
                f"{task_name}: missing tokenizer "
                f"(src={src_lang}:{tok_src}, tgt={tgt_lang}:{tok_tgt})"
            )
            continue

        print(f"Profiling task: {task_name} ({src_lang}->{tgt_lang})...", flush=True)

        profile = profile_token_lengths(
            src_path, tgt_path, tok_src, tok_tgt,
            max_samples=args.max_samples,
            tokenizer_cache=tokenizer_cache,
        )
        task_profiles[task_name] = profile

        valid_src = task.get("path_valid_src", "")
        valid_tgt = task.get("path_valid_tgt", "")
        if valid_src and valid_tgt:
            valid_src_lang = extract_lang_from_path(valid_src)
            valid_tgt_lang = extract_lang_from_path(valid_tgt)
            tok_valid_src = resolve_tokenizer(valid_src_lang, src_vocab)
            tok_valid_tgt = resolve_tokenizer(valid_tgt_lang, tgt_vocab)
            if not tok_valid_src or not tok_valid_tgt:
                skipped_valid.append(
                    f"{task_name}: missing tokenizer "
                    f"(src={valid_src_lang}:{tok_valid_src}, tgt={valid_tgt_lang}:{tok_valid_tgt})"
                )
            else:
                print(f"  Profiling validation: {task_name}...", flush=True)
                valid_profiles[task_name] = profile_token_lengths(
                    valid_src, valid_tgt, tok_valid_src, tok_valid_tgt,
                    tokenizer_cache=tokenizer_cache,
                )

    if skipped:
        print(f"\nSkipped {len(skipped)} training task(s):")
        for s in skipped:
            print(f"  - {s}")

    if skipped_valid:
        print(f"\nSkipped {len(skipped_valid)} validation task(s):")
        for s in skipped_valid:
            print(f"  - {s}")

    if not task_profiles:
        print("Error: No tasks could be profiled.", file=sys.stderr)
        sys.exit(1)

    recommendations = compute_optimal_settings(task_profiles, percentile=args.percentile)
    valid_recommendations = (
        compute_valid_max_length(valid_profiles, percentile=args.percentile)
        if valid_profiles
        else None
    )
    print_report(config, recommendations, valid_recommendations)


if __name__ == "__main__":
    main()
