#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path


def count_files(path: Path, pattern: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob(pattern))


def count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", errors="replace") as f:
        return sum(1 for _ in f)

def count_regex_lines(path: Path, pattern: str) -> int:
    if not path.is_file():
        return 0
    rx = re.compile(pattern)
    with path.open("r", errors="replace") as f:
        return sum(1 for line in f if rx.search(line))

    
def slurm_time_to_hhmm(value: str) -> str:
    # Supports: D-HH:MM:SS, HH:MM:SS, MM:SS, HH:MM
    days = 0
    if "-" in value:
        day_s, value = value.split("-", 1)
        days = int(day_s)

    parts = value.split(":")
    if len(parts) == 3:
        hh, mm, _ss = map(int, parts)
    elif len(parts) == 2:
        # In squeue elapsed time this is MM:SS.
        # In sbatch --time it is usually HH:MM.
        hh, mm = map(int, parts)
    else:
        return value

    hh += days * 24
    return f"{hh:02d}:{mm:02d}"


def sbatch_time_to_hhmm(value: str) -> str:
    days = 0
    if "-" in value:
        day_s, value = value.split("-", 1)
        days = int(day_s)

    parts = value.split(":")
    if len(parts) >= 2:
        hh = int(parts[0]) + days * 24
        mm = int(parts[1])
        return f"{hh:02d}:{mm:02d}"

    return value


def extract_option(line: str, name: str) -> str:
    m = re.search(rf"--{re.escape(name)}=([^ ]+)", line)
    return m.group(1) if m else ""


def infer_info(model_dir: Path) -> str:
    sbatch = model_dir / "inf_out" / "inf.sbatch"
    if not sbatch.is_file() or sbatch.stat().st_size == 0:
        return ""

    line = sbatch.read_text(errors="replace").strip()

    t = extract_option(line, "time")
    nodes = extract_option(line, "nodes")
    gpn = extract_option(line, "gpus-per-node")
    gpus = extract_option(line, "gpus")
    ntasks = extract_option(line, "ntasks")

    if nodes and gpn:
        total_gpus = str(int(nodes) * int(gpn))
    elif gpus:
        total_gpus = gpus
    elif ntasks:
        total_gpus = ntasks
    else:
        total_gpus = "?"

    return f"{sbatch_time_to_hhmm(t)}x{total_gpus}" if t else "[x]"


def current_runtime(model_dir: Path) -> str:
    done_file = model_dir / "inference.done"
    flag = model_dir / "inference.submitted"

    if done_file.exists():
        return "DONE"

    if not flag.is_file():
        return ""

    jobid = flag.read_text(errors="replace").strip()
    if not jobid:
        return ""

    try:
        result = subprocess.run(
            ["squeue", "-h", "-j", jobid, "-o", "%M"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return ""

    raw = result.stdout.strip().splitlines()
    if not raw:
        try:
            flag.unlink()
        except OSError:
            pass
        return ""

    runtime_raw = raw[0].strip()

    if re.match(r"^\d+-\d+:\d+", runtime_raw):
        return slurm_time_to_hhmm(runtime_raw)
    if re.match(r"^\d+:\d+:\d+$", runtime_raw):
        hh, mm, _ss = map(int, runtime_raw.split(":"))
        return f"{hh:02d}:{mm:02d}"
    if re.match(r"^\d+:\d+$", runtime_raw):
        mm, _ss = map(int, runtime_raw.split(":"))
        return f"00:{mm:02d}"

    return runtime_raw


def pair_count_str(zero_count: int, main_count: int) -> str:
    return f"{zero_count}+{main_count}"

def row(alias: str, model_dir_s: str) -> tuple:
    model_dir = Path(model_dir_s)

    yaml_ok = count_files(model_dir / "inf_out", "*.yaml")
    calls = count_lines(model_dir / "inf_out" / "calls.out")

    plan_file = model_dir / "inf_out" / "plan.out"
    plan_hyp = count_regex_lines(plan_file, r"\.hyp\b")
    plan_0shyp = count_regex_lines(plan_file, r"\.0shyp\b")
    plan = pair_count_str(plan_0shyp, plan_hyp)
    
    zhyp_count = count_files(model_dir / "inf_out", "*.0shyp")
    hyp_count = count_files(model_dir / "inf_out", "*.hyp")
    hyp = pair_count_str(zhyp_count, hyp_count)
    
    sacre0_count = count_files(model_dir / "inf_scores", "*.0ssacre")
    sacre_count = count_files(model_dir / "inf_scores", "*.sacre")
    sacre = pair_count_str(sacre0_count, sacre_count)

    comet0_count = count_files(model_dir / "inf_scores", "*.0scomet")
    comet_count = count_files(model_dir / "inf_scores", "*.comet")
    comet = pair_count_str(comet0_count, comet_count)

    return (
        alias,
        yaml_ok,
        plan,
        calls,
        infer_info(model_dir),
        current_runtime(model_dir),
        hyp,
        sacre,
        comet,
    )

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="ALIAS=DIR",
        help="Model alias and model directory. May be repeated.",
    )
    args = parser.parse_args()

    models: list[tuple[str, str]] = []
    for item in args.model:
        if "=" not in item:
            raise SystemExit(f"Bad --model value: {item!r}; expected ALIAS=DIR")
        alias, model_dir = item.split("=", 1)
        models.append((alias, model_dir))

    fmt = "{:<22} {:<5} {:<12} {:<6} {:<10} {:<5} {:<9} {:<9} {:<9}"
    print(fmt.format("model", "yamls", "0s+spv tasks", "calls", "HH:MMxGPUs", "elaps", "hyp", "sacre", "comet"))
    print(fmt.format("-" * 22, "-" * 5, "-" * 12, "-" * 6, "-" * 10, "-" * 5, "-" * 9, "-" * 9, "-" * 9))

    for alias, model_dir in models:
        print(fmt.format(*row(alias, model_dir)))

    print(fmt.format("-" * 22, "-" * 5, "-" * 12, "-" * 6, "-" * 10, "-" * 5, "-" * 9, "-" * 9, "-" * 9))

if __name__ == "__main__":
    main()

    
