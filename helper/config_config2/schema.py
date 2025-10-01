# config_config2/schema.py

from __future__ import annotations
from typing import Dict, Any, List

# Paths use dotted keys relative to the root YAML.
# For config_config.* keys, prefix them with "config_config."
SCHEMA: Dict[str, Dict[str, Any]] = {
    # ---- top-level sections ----
    "src_vocab": {
        "type": "mapping[str -> path_template]",
        "required": True,
        "description": "Source language vocab templates. Used by complete_language_pairs.",
        "example": {"en": "/data/vocabs/en.vocab", "fi": "/data/vocabs/fi.vocab"},
    },
    "tgt_vocab": {
        "type": "mapping[str -> path_template]",
        "required": True,
        "description": "Target language vocab templates. Used by complete_language_pairs.",
        "example": {"en": "/data/vocabs/en.vocab", "fi": "/data/vocabs/fi.vocab"},
    },
    "tasks": {
        "type": "mapping[str -> task]",
        "required": False,
        "description": "Generated/curated tasks. Many commands read and write here.",
        "example": {"en-fi": {"src": "...", "tgt": "..."}},
    },

    # ---- config_config.* keys (global knobs) ----
    "config_config.temperature": {
        "type": "float",
        "required": False,
        "description": "Weight shaping exponent for corpora_schedule (size^temperature).",
        "example": 1.0,
    },
    "config_config.use_weight": {
        "type": "bool",
        "required": False,
        "description": "Enable weighted corpus sampling in corpora_schedule.",
        "example": True,
    },
    "config_config.ae_weight": {
        "type": "float",
        "required": False,
        "description": "Multiplier for AE tasks in corpora_schedule.",
        "example": 0.5,
    },
    "config_config.use_introduce_at_training_step": {
        "type": "bool",
        "required": False,
        "description": "Enable curriculum ('introduce_at_training_step') in corpora_schedule.",
        "example": False,
    },
    "config_config.split_large_language_pairs": {
        "type": "float",
        "required": False,
        "description": "Split corpora with weight > threshold into stride/offset shards.",
        "example": 0.15,
    },
    "config_config.distance_matrix": {
        "type": "path(csv)",
        "required": False,
        "description": "Language distance matrix for cluster_languages.",
        "example": "/data/lang_dist.csv",
    },
    "config_config.n_groups": {
        "type": "int",
        "required": False,
        "description": "Number of clusters for cluster_languages (unless 'groups' is provided).",
        "example": 16,
    },
    "config_config.groups": {
        "type": "mapping[str -> group_id]",
        "required": False,
        "description": "Predefined language->group mapping; skips clustering if provided.",
        "example": {"en": 0, "fi": 3},
    },
    "config_config.enc_sharing_groups": {
        "type": "list[str]",
        "required": False,
        "description": "Per-encoder-layer sharing tokens (LANGUAGE/GROUP/FULL/with SRC_/TGT_).",
        "example": ["LANGUAGE", "GROUP", "FULL"],
    },
    "config_config.dec_sharing_groups": {
        "type": "list[str]",
        "required": False,
        "description": "Per-decoder-layer sharing tokens (…same tokens…).",
        "example": ["TGT_LANGUAGE", "GROUP", "FULL"],
    },
    "config_config.n_gpus_per_node": {
        "type": "int",
        "required": True,
        "description": "GPUs per node for allocate_devices.",
        "example": 8,
    },
    "config_config.n_nodes": {
        "type": "int",
        "required": False,
        "description": "Nodes to use (optional if n_slots_per_gpu is given).",
        "example": 4,
    },
    "config_config.n_slots_per_gpu": {
        "type": "int",
        "required": False,
        "description": "Task slots per GPU (optional if n_nodes is given).",
        "example": 2,
    },
    "config_config.transforms": {
        "type": "list[str]",
        "required": False,
        "description": "Default transforms for translation tasks.",
        "example": ["normalize", "prefix"],
    },
    "config_config.ae_transforms": {
        "type": "list[str]",
        "required": False,
        "description": "Default transforms for autoencoder tasks.",
        "example": ["normalize"],
    },
    "config_config.use_src_lang_token": {
        "type": "bool",
        "required": False,
        "description": "If true, require 'prefix' transform somewhere.",
        "example": True,
    },
    # …extend as needed…
}

# Which keys each command reads/writes (for help + change explanation)
COMMAND_IO: Dict[str, Dict[str, Any]] = {
    "complete_language_pairs": {
        "reads": ["src_vocab", "tgt_vocab", "config_config.src_path", "config_config.tgt_path",
                  "config_config.ae_path", "config_config.valid_src_path", "config_config.valid_tgt_path",
                  "config_config.autoencoder", "config_config.autoencoder_validation"],
        "writes": ["tasks"],
        "summary": "Scans vocab/templates and builds tasks; optional autoencoder/validation tasks.",
    },
    "corpora_schedule": {
        "reads": ["tasks", "config_config.temperature", "config_config.use_weight",
                  "config_config.ae_weight", "config_config.use_introduce_at_training_step",
                  "config_config.split_large_language_pairs"],
        "writes": ["tasks"],  # updates weights, stride/offset, introduce_at_training_step
        "summary": "Computes weights and optional curriculum; may split large corpora.",
    },
    "cluster_languages": {
        "reads": ["config_config.distance_matrix", "config_config.n_groups", "config_config.groups"],
        "writes": ["config_config.groups"],
        "summary": "Builds language→group mapping, unless already provided.",
    },
    "sharing_groups": {
        "reads": ["config_config.groups", "config_config.enc_sharing_groups",
                  "config_config.dec_sharing_groups", "enc_layers", "dec_layers"],
        "writes": ["config_config.encoder_sharing", "config_config.decoder_sharing"],
        "summary": "Resolves per-layer sharing lists for encoder/decoder.",
    },
    "allocate_devices": {
        "reads": ["tasks", "config_config.groups", "config_config.n_gpus_per_node",
                  "config_config.n_nodes", "config_config.n_slots_per_gpu",
                  "config_config.time_budget_s", "config_config.log_name"],
        "writes": ["tasks", "world_size", "node_gpu", "gpu_ranks"],
        "summary": "Assigns tasks to node:GPU slots; sets world_size/ranks.",
    },
    "set_transforms": {
        "reads": ["tasks", "config_config.transforms", "config_config.ae_transforms",
                  "config_config.use_src_lang_token"],
        "writes": ["tasks"],
        "summary": "Attaches transforms to tasks; enforces prefix when required.",
    },
    "adapter_config": {
        "reads": ["adapters", "config_config.groups"],
        "writes": ["adapters", "tasks"],
        "summary": "Expands adapter id-spaces (LANGUAGE/GROUP/FULL) and maps tasks to adapters.",
    },
    "translation_configs": {
        "reads": ["tasks", "config_config.zero_shot"],
        "writes": ["tasks", "configs"],  # or whatever you create
        "summary": "Generates (optionally zero-shot) translation configs.",
    },
    "remove_temporary_keys": {
        "reads": ["config_config"],
        "writes": ["(removes config_config)"],
        "summary": "Strips transient config_config before final save.",
    },
    # …extras, etc…
}

# still in schema.py
import shutil, textwrap, re

def _term_cols() -> int:
    try:
        return shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:
        return 80

def _wrap(text: str, indent: int = 2) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    width = max(40, _term_cols() - indent)
    return textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)

def print_schema(keys: list[str] | None = None) -> None:
    """Print YAML schema for all or selected keys."""
    print("YAML schema (selected keys):" if keys else "YAML schema (all known keys):")
    items = sorted((k, v) for k, v in SCHEMA.items() if keys is None or k in keys)
    if not items:
        print("(no matching keys)")
        return
    for k, meta in items:
        print(f"\n{k}")
        print(_wrap(f"type: {meta.get('type','?')}", indent=4))
        if "required" in meta:
            print(_wrap(f"required: {meta['required']}", indent=4))
        if "description" in meta:
            print(_wrap(meta["description"], indent=4))
        if "example" in meta:
            ex = meta["example"]
            print(_wrap(f"example: {ex}", indent=4))

def print_command_yaml_help(cmd: str) -> None:
    """Explain, for one command, which YAML keys it reads/writes (with schema details when known)."""
    io = COMMAND_IO.get(cmd)
    if not io:
        print(f"No YAML documentation for '{cmd}'.")
        return
    print(f"{cmd} — what it reads/writes")
    if io.get("summary"):
        print(_wrap(io["summary"], indent=2))
        print()
    reads = io.get("reads", [])
    writes = io.get("writes", [])
    if reads:
        print("Reads:")
        for k in reads:
            meta = SCHEMA.get(k, {})
            desc = meta.get("description", "")
            line = f"{k}" + (f" — {desc}" if desc else "")
            print(_wrap(line, indent=2))
    if writes:
        print("\nWrites:")
        for k in writes:
            meta = SCHEMA.get(k, {})
            desc = meta.get("description", "")
            line = f"{k}" + (f" — {desc}" if desc else "")
            print(_wrap(line, indent=2))
