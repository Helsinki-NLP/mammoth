# config_config2/reduce.py
from __future__ import annotations

import os
from typing import List, Dict, Any, Set, Tuple

from .utils import load_yaml, save_yaml, UserConfigError, logger  # use your existing utils


def _write_yaml_file(path: str, obj) -> None:
    """
    Robust writer: tries utils.save_yaml(path, obj); if your project defines
    save_yaml(obj) with a different signature, fall back to plain yaml dump.
    """
    try:
        from .utils import save_yaml as _save_yaml
    except Exception:
        _save_yaml = None

    if _save_yaml is not None:
        try:
            # Preferred modern signature: (path, data)
            _save_yaml(path, obj)
            return
        except TypeError:
            # Fallback: maybe utils.save_yaml(obj) writes to a known place or returns text
            try:
                _save_yaml(obj)  # if this writes through internal path/logic
                return
            except Exception:
                pass

    # Final fallback: direct dump
    try:
        import yaml
        try:
            from yaml import CDumper as _YDumper
        except Exception:
            from yaml import Dumper as _YDumper
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(obj, f, Dumper=_YDumper, sort_keys=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Failed to write YAML to {path}: {e}") from e

def _ordered_commands(candidates: List[str]) -> List[str]:
    idx = {c: i for i, c in enumerate(PIPELINE_ORDER)}
    return sorted(candidates, key=lambda c: idx.get(c, len(PIPELINE_ORDER)))

def _detect_commands_from_output(doc: Dict[str, Any]) -> List[str]:
    """Pick commands whose 'writes' appear in the document, ordered by pipeline."""
    found = []
    for cmd, io in COMMAND_IO.items():
        writes = set(io.get("writes") or [])
        if not writes:
            continue
        # if the doc contains ANY write key for this command, consider it used
        if any(_has_key(doc, k) for k in writes):
            found.append(cmd)
    return _ordered_commands(found)

def _has_key(doc: Dict[str, Any], dotted: str) -> bool:
    cur = doc
    parts = dotted.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            return False
        cur = cur[p]
    return True

def _get_value(doc: Dict[str, Any], dotted: str):
    cur = doc
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur

def _ensure(root: Dict[str, Any], dotted: str):
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        cur.setdefault(p, {})
        cur = cur[p]
    return cur, parts[-1]

def _prune_writes_from_reduced(reduced: Dict[str, Any], cmds: List[str]):
    """Remove generated output keys from reduced input YAML."""
    writes_all = set()
    for c in cmds:
        writes_all.update(COMMAND_IO.get(c, {}).get("writes") or [])
    for key in sorted(writes_all, key=lambda s: (-s.count("."), s)):  # delete deep first
        parent, leaf = _ensure_parent(reduced, key)
        if parent and leaf in parent:
            del parent[leaf]
    # clean empty maps
    _prune_empty_maps(reduced)

def _ensure_parent(root: Dict[str, Any], dotted: str):
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            return None, parts[-1]
        cur = cur[p]
    return cur, parts[-1]

def _prune_empty_maps(obj):
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            _prune_empty_maps(obj[k])
            if isinstance(obj[k], dict) and not obj[k]:
                del obj[k]
    elif isinstance(obj, list):
        for x in obj:
            _prune_empty_maps(x)

def _explain_coverage(doc: Dict[str, Any], cmds: List[str]) -> Tuple[Set[str], Set[str]]:
    """Return (explained_keys, unexplained_top_keys)."""
    writes = set()
    for c in cmds:
        writes.update(COMMAND_IO.get(c, {}).get("writes") or [])
    # explained: any of those write keys present
    explained = {k for k in writes if _has_key(doc, k)}

    # unexplained: top-level keys in doc that are not in any writes or always-input clusters
    # We treat these as "inputs/durable": src_vocab, tgt_vocab, tasks, adapters, configs, config_config, world_size, node_gpu, gpu_ranks, etc.
    # But some of these *are* writes; that’s fine—this set is just a sanity surface.
    top = set(doc.keys())
    # If you want to be stricter, subtract any top-level key that is a prefix of any write key.
    return explained, (top - set(k.split(".")[0] for k in writes))

def reduce_command(opts):
    """
    --in_config OUT.yaml          (required) produced YAML to analyze
    --out_reduced REDUCED.yaml    (required) minimal input YAML to write
    --pipeline CMD [CMD ...]      (optional) override auto-detection; order is respected
    --emit-script FILE.sh         (optional) write a replay script
    --dry-run                     (optional) do not write files; just show the plan
    --explain                     (optional) print coverage report
    --template-mode {inputs,all,raw} (optional) how minimal to make the reduced YAML (default: inputs)
    """
    doc, _ = load_yaml(opts.in_config)

    if getattr(opts, "pipeline", None):
        cmds = list(opts.pipeline)
    else:
        cmds = _detect_commands_from_output(doc)

    if not cmds:
        raise UserConfigError("Could not detect any pipeline steps in the YAML. "
                              "Pass --pipeline CMD ... to specify them.")

    # Build a minimal template (only required inputs by default)
    mode = getattr(opts, "template_mode", "inputs")
    reduced = build_yaml_template_for_commands(cmds, mode=mode)

    # Fill template keys from the produced doc where possible
    for dotted in _flatten_keys(reduced):
        v = _get_value(doc, dotted)
        if v is not None:
            parent, leaf = _ensure(reduced, dotted)
            parent[leaf] = v

    # Remove all generated outputs from the reduced YAML
    _prune_writes_from_reduced(reduced, cmds)

    # Optional coverage explanation
    if getattr(opts, "explain", False):
        explained, unexplained_top = _explain_coverage(doc, cmds)
        print("--- reduce: pipeline steps ---")
        for c in cmds: print("  •", c)
        print("\n--- reduce: explained output keys (present & written by steps) ---")
        for k in sorted(explained): print("  -", k)
        print("\n--- reduce: top-level keys in OUT.yaml not directly explained by writes ---")
        for k in sorted(unexplained_top): print("  -", k)
        print()

    # Write reduced YAML
    if not getattr(opts, "dry_run", False):
        _write_yaml_file(opts.out_reduced, reduced)

    # Prepare a replay script suggestion
    if getattr(opts, "emit_script", None):
        script = _make_replay_script(cmds, opts.out_reduced, os.path.splitext(opts.out_reduced)[0])
        if not getattr(opts, "dry_run", False):
            with open(opts.emit_script, "w", encoding="utf-8") as f:
                f.write(script)
        else:
            print("--- suggested replay script ---")
            print(script)

def _flatten_keys(d: Dict[str, Any], prefix: str = "") -> List[str]:
    out = []
    for k, v in d.items():
        dotted = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_flatten_keys(v, dotted))
        else:
            out.append(dotted)
    return out

def _make_replay_script(cmds: List[str], in_yaml: str, stem: str) -> str:
    """
    Create a simple shell script that chains the steps, producing step1.yaml, step2.yaml, ... final.yaml
    """
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    prev = in_yaml
    for i, cmd in enumerate(cmds, 1):
        outp = f"{stem}.step{i:02d}.yaml" if i < len(cmds) else f"{stem}.replayed.yaml"
        lines.append(f"python -m config_config2 {cmd} --in_config {prev} --out_config {outp}")
        prev = outp
    lines.append("")
    lines.append(f'echo "Replayed output written to {prev}"')
    return "\n".join(lines)

def register(subparsers):
    p = subparsers.add_parser(
        "reduce",
        help="Infer minimal input YAML from a produced YAML and suggest a replay script.",
        description=(
            "Given a produced YAML, infer the minimal input (removing self-generated keys), "
            "detect pipeline steps, and output a reproducer script."
        ),
    )
    p.add_argument("--in_config", metavar="OUT.yaml", required=True,
                   help="The produced YAML you want to reverse into a minimal input.")
    p.add_argument("--out_reduced", metavar="REDUCED.yaml", required=True,
                   help="Where to write the reduced (minimal) input YAML.")
    p.add_argument("--pipeline", nargs="+", metavar="CMD",
                   help="Override auto-detection: explicit list of commands in replay order.")
    p.add_argument("--emit_script", metavar="FILE.sh",
                   help="Write a runnable script that replays the pipeline with the reduced YAML.")
    p.add_argument("--dry_run", action="store_true",
                   help="Show the plan but do not write files.")
    p.add_argument("--reduce-explain", action="store_true",
                   help="Print which keys are explained by the steps and which are not.")
    p.add_argument("--template_mode", choices=["inputs", "all", "raw"], default="inputs",
                   help="How minimal to make the reduced YAML. 'inputs' keeps only needed inputs.")
    p.set_defaults(handler=reduce_command, _parser=p)
    p.set_defaults(_mutates_yaml=True)

    register_command_io("reduce", {
        "reads": [
            # strictly speaking it reads the whole doc, but for docs:
            "src_vocab", "tgt_vocab", "tasks", "config_config", "adapters", "configs",
            "world_size", "node_gpu", "gpu_ranks",
        ],
        "writes": [
            # it writes the reduced YAML file; inside YAML it doesn't add new keys,
            # it *removes* generated keys. So we can leave this empty or note the output file.
        ],
        "summary": "Reverse-engineer a produced YAML into minimal inputs and suggest a replay script.",
    })


    
