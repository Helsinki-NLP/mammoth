# config_config2/yaml_help.py

from __future__ import annotations
from typing import Dict, Any, List, Set
from copy import deepcopy
import shutil, textwrap, re
from .utils import UserConfigError

from typing import List, Dict, Any
from .utils import UserConfigError, COMMAND_IO, register_command_io, register_command_template_extras

def register(subparsers):
    p = subparsers.add_parser(
        "yaml_help",
        help="Describe YAML keys used by commands.",
        description=("Explain YAML inputs/outputs by command or by specific keys. "
                     "Use '--cmd CMD [...]' for focused help, '--keys [KEY ...]' to describe keys "
                     "(if no KEY is given, shows all keys), or '--all' for all commands.")
    )
    p.add_argument("--keys", nargs="*", metavar="KEY",
                   help="Describe specific YAML keys. If used with no KEYs, lists all keys.")
    p.add_argument("--cmd", nargs="*", metavar="COMMAND",
                   help="Show YAML help for one or more commands. If usef with no CMD, for all commands")
    p.set_defaults(handler=yaml_help_command,
                   _parser=p,
                   _early_exit=True,          # <-- ADD THIS
                   _mutates_yaml=False)       # <-- and be explicit that it doesn't write YAML

    register_command_io("yaml_help", {
        "reads": [],
        "writes": [],
        "summary": "Describe YAML used by commands (by command, keys, or all).",
    })
    

def _term_cols() -> int:
    try:
        return shutil.get_terminal_size(fallback=(80, 24)).columns
    except Exception:
        return 80

def bullet_fill(text: str, left: int = 2, bullet: str = "- ", extra: int = 2) -> str:
    """Wrap with a hanging indent that aligns continuation after the bullet."""
    text = " ".join((text or "").split())
    initial = " " * left + bullet
    subsequent = " " * (left + len(bullet) + extra)
    width = max(40, _term_cols() - len(initial))
    return textwrap.fill(
        text,
        width=width,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )

SCHEMA: Dict[str, Dict[str, Any]] = {
    "config_config.complete_language_pairs.src_path":  {"type": "path", "required": False, "description": "Source train path template", "example": "/data/{src_lang}-{tgt_lang}.src"},
    "config_config.complete_language_pairs.tgt_path":  {"type": "path", "required": False, "description": "Target train path template", "example": "/data/{tgt_lang}-{src_lang}.tgt"},
    "config_config.complete_language_pairs.valid_src_path": {"type": "path", "required": False, "description": "Validation source template", "example": "/data/valid/{src_lang}-{tgt_lang}.src"},
    "config_config.complete_language_pairs.valid_tgt_path": {"type": "path", "required": False, "description": "Validation target template", "example": "/data/valid/{tgt_lang}-{src_lang}.tgt"},
    "config_config.complete_language_pairs.autoencoder": {"type": "bool", "required": False, "description": "Whether AE tasks are expected", "example": True},
    "config_config.complete_language_pairs.autoencoder_validation": {"type": "bool", "required": False, "description": "Whether AE validation is expected", "example": True},
    "config_config.complete_language_pairs.ae_path":   {"type": "path", "required": False, "description": "Monolingual AE template", "example": "/data/mono/{lang}.txt"},
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
    "config_config.enc_layers": {
        "type": "int",
        "required": True,
        "description": "Number of encoder layers in the model. Used by 'sharing_groups' to expand per-layer sharing lists.",
        "example": 6,
    },
    "config_config.dec_layers": {
        "type": "int",
        "required": True,
        "description": "Number of decoder layers in the model. Used by 'sharing_groups' to expand per-layer sharing lists.",
        "example": 6,
    },
    "config_config.enc_sharing_groups": {
        "type": "list[str]",
        "required": False,
        "description": "High-level pattern for encoder sharing (short list). Expanded to per-layer 'encoder_sharing'.",
        "example": ["LANGUAGE", "GROUP", "FULL"],
    },
    "config_config.dec_sharing_groups": {
        "type": "list[str]",
        "required": False,
        "description": "High-level pattern for decoder sharing (short list). Expanded to per-layer 'decoder_sharing'.",
        "example": ["TGT_LANGUAGE", "GROUP", "FULL"],
    },
    "config_config.encoder_sharing": {
        "type": "list[str]",
        "required": False,
        "description": "Expanded per-layer encoder sharing (one entry per encoder layer). Produced by 'sharing_groups'.",
        "example": ["LANGUAGE", "GROUP", "FULL", "FULL", "FULL", "FULL"],
    },
    "config_config.decoder_sharing": {
        "type": "list[str]",
        "required": False,
        "description": "Expanded per-layer decoder sharing (one entry per decoder layer). Produced by 'sharing_groups'.",
        "example": ["TGT_LANGUAGE", "GROUP", "FULL", "FULL", "FULL", "FULL"],
    },
    "config_config.src_path": {
        "type": "path(template with {src},{tgt})",
        "required": False,
        "description": "Path template for bilingual *source* files. Use {src} and {tgt} placeholders; they will be replaced with language codes.",
        "example": "/data/corpora/{src}-{tgt}.src",
    },
    "config_config.tgt_path": {
        "type": "path(template with {src},{tgt})",
        "required": False,
        "description": "Path template for bilingual *target* files. Use {src} and {tgt} placeholders.",
        "example": "/data/corpora/{src}-{tgt}.tgt",
    },
    "config_config.ae_path": {
        "type": "path(template with {lang})",
        "required": False,
        "description": "Path template for *monolingual* autoencoder data. Use {lang} placeholder.",
        "example": "/data/mono/{lang}.txt",
    },
    "config_config.valid_src_path": {
        "type": "path(template with {src},{tgt})",
        "required": False,
        "description": "Optional path template for bilingual *validation source* files. Used only if both source and target validation files exist.",
        "example": "/data/valid/{src}-{tgt}.src",
    },
    "config_config.valid_tgt_path": {
        "type": "path(template with {src},{tgt})",
        "required": False,
        "description": "Optional path template for bilingual *validation target* files. Used only if both source and target validation files exist.",
        "example": "/data/valid/{src}-{tgt}.tgt",
    },
    "config_config.autoencoder": {
        "type": "bool",
        "required": False,
        "description": "If 'True', create autoencoder tasks for each language using 'ae_path' when files exist.",
        "example": True,
    },
    "config_config.autoencoder_validation": {
        "type": "bool",
        "required": False,
        "description": "If 'True', also attach validation data for autoencoder tasks (implementation-dependent).",
        "example": False,
    },
    "configs": {"type": "mapping[str -> dict]", "required": False,
                "description": "Emitted training/eval configuration blocks."},
    "world_size": {"type": "int", "required": False, "description": "Total process count for DDP."},
    "node_gpu": {"type": "mapping[node -> list[gpu]]", "required": False,
                 "description": "Per-node GPU indices used for placement."},
    "gpu_ranks": {"type": "list[int]", "required": False,
                  "description": "Global ranks corresponding to (node,gpu) slots."},
    "adapters": {"type": "mapping[str -> adapter]", "required": False,
                 "description": "Adapter definitions and assignment to tasks."},
}
# Some generic example defaults we’ll use when SCHEMA lacks an example:
_GENERIC_EXAMPLES = {
    "src_vocab": {"en": "/example/vocabs/en.vocab", "fi": "/example/vocabs/fi.vocab"},
    "tgt_vocab": {"en": "/example/vocabs/en.vocab", "fi": "/example/vocabs/fi.vocab"},
    "tasks": {},

    # Paths with placeholders that your pipeline understands:
    "config_config.src_path": "/example/corpora/{src}-{tgt}.src",
    "config_config.tgt_path": "/example/corpora/{src}-{tgt}.tgt",
    "config_config.ae_path":  "/example/mono/{lang}.txt",
    "config_config.valid_src_path": "/example/valid/{src}-{tgt}.src",
    "config_config.valid_tgt_path": "/example/valid/{src}-{tgt}.tgt",

    # Scheduling knobs:
    "config_config.temperature": 1.0,
    "config_config.use_weight": True,
    "config_config.ae_weight": 0.5,
    "config_config.use_introduce_at_training_step": False,
    "config_config.split_large_language_pairs": 0.15,

    # Layers & sharing (safe defaults if not specified elsewhere):
    "config_config.enc_layers": 6,
    "config_config.dec_layers": 6,
    "config_config.enc_sharing_groups": ["LANGUAGE", "GROUP", "FULL"],
    "config_config.dec_sharing_groups": ["TGT_LANGUAGE", "GROUP", "FULL"],

    # Clustering & groups:
    "config_config.n_groups": 2,
    "config_config.groups": {"en": 0, "fi": 1},

    # Devices:
    "config_config.n_gpus_per_node": 4,
    "config_config.n_nodes": 2,              # or use n_slots_per_gpu below
    "config_config.n_slots_per_gpu": 1,

    # Transforms:
    "config_config.transforms": ["normalize", "prefix"],
    "config_config.ae_transforms": ["normalize"],
    "config_config.use_src_lang_token": True,

    # Zero-shot optional:
    "config_config.zero_shot": [],
}

def print_keys(keys: list[str]) -> None:
    """Print YAML schema for selected keys."""
    if not keys:
        print("  (no YAML keys)\n")
        return
    missing = [k for k in keys if k not in SCHEMA]
    if missing:
        raise UserConfigError("yaml_help: unknown key(s): " + ", ".join(missing))
    items = sorted((k, v) for k, v in SCHEMA.items() if k in keys)
    if not items:
        print("(no matching keys)")
        return
    for key, meta in items:
        typ = meta.get("type", "(unknown type)")
        req = "required" if meta.get("required") else "optional"
        desc = meta.get("description", "(no description)")
        ex = meta.get("example", None)
        reads = []
        writes = []
        for c in sorted(COMMAND_IO.keys()):
            io = COMMAND_IO.get(c, {})
            if key in io.get("reads",[]):
                reads.append(c)
            if key in io.get("writes",[]):
                writes.append(c)
        if not reads and not writes:
            continue
        if reads:
            read = ",".join(reads)
        else:
            read = "MAMMOTH"
        if writes:
            write = ",".join(writes)
        else:
            write = "USER"
        
        print(f"\n{key} ({req})")
        if "description" in meta:
            print(bullet_fill(desc, left=4, bullet="", extra=2))
        print(bullet_fill(f"information flow: {write} ===> {read}", left=4, bullet="- ", extra=2))
        if "example" in meta:
            print(bullet_fill(f"example: {ex}", left=4, bullet="- ", extra=2))
        print(bullet_fill(f"type: {typ}", left=4, bullet="- ", extra=2))
        
    print()

def example_for_key(key: str):
    """Return an example value for a given YAML dotted key."""
    meta = SCHEMA.get(key)
    if meta and "example" in meta:
        return deepcopy(meta["example"])
    # fallback to our generic examples
    if key in _GENERIC_EXAMPLES:
        return deepcopy(_GENERIC_EXAMPLES[key])
    # last resort: simple placeholders by type
    t = (meta or {}).get("type", "")
    if t.startswith("mapping"):
        return {}
    if t.startswith("list"):
        return []
    if t in ("int", "float", "number"):
        return 0
    if t == "bool":
        return False
    if "path" in t:
        return "/example/path"
    return None

# Canonical pipeline order (used for sorting if user passes out-of-order)
PIPELINE_ORDER: List[str] = [
    "complete_language_pairs",
    "corpora_schedule",
    "cluster_languages",
    "sharing_groups",
    "set_transforms",
    "allocate_devices",
    "remove_temporary_keys"
]

def _order_commands(cmds: List[str]) -> List[str]:
    # Keep user order for listed cmds, but move any known ones to their canonical positions
    # Strategy: stable sort by index in PIPELINE_ORDER when present
    idx = {c: i for i, c in enumerate(PIPELINE_ORDER)}
    return sorted(cmds, key=lambda c: idx.get(c, len(PIPELINE_ORDER)))

def _ensure(root: dict, dotted: str):
    """Ensure nested structure for a dotted key and return the final parent dict + last key."""
    parts = dotted.split(".")
    cur = root
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    return cur, parts[-1]

def build_yaml_template_for_commands(
    cmds: List[str],
    mode: str = "inputs",  # "inputs" | "all" | "raw"
) -> Dict[str, Any]:
    """
    Build a minimal YAML dict for multiple commands.
    modes:
      - inputs: include only keys the user must provide (reads minus prior writes) + smart extras
      - all:    include union of reads + extras (no pruning)
      - raw:    include only exact reads (strict)
    """
    from .schema import COMMAND_IO, COMMAND_TEMPLATE_EXTRAS  # ensure available

    cmds = _order_commands(cmds)
    root: Dict[str, Any] = {}
    needed: Set[str] = set()
    produced: Set[str] = set()

    # Helper to add keys with examples into root
    def _add_keys(keys: List[str]):
        nonlocal root
        # Ensure parent containers
        if any(k.startswith("config_config.") for k in keys):
            root.setdefault("config_config", {})
        if "tasks" in keys:
            root.setdefault("tasks", {})
        for k in keys:
            parent, leaf = _ensure(root, k)
            val = example_for_key(k)
            if val is None:
                val = "<fill me>"
            parent[leaf] = deepcopy(val)

    if mode == "all":
        union_reads: Set[str] = set()
        for c in cmds:
            union_reads.update(COMMAND_IO.get(c, {}).get("reads", []))
            union_reads.update(COMMAND_TEMPLATE_EXTRAS.get(c, []))
        _add_keys(sorted(union_reads))
        return root

    # 1) choose keys based on IO + mode
    #    reads = union of reads from cmds (filter by mode)
    # 2) for each key, fetch schema from effective_schema_for_commands(cmds)
    # 3) build the skeleton with "<fill me>" or examples
    # 4) merge extras (COMMAND_TEMPLATE_EXTRAS[cmd]) for each cmd

    for c in cmds:
        io = COMMAND_IO.get(c, {})
        reads = list(io.get("reads", []))
        writes = list(io.get("writes", []))
        extras = list(COMMAND_TEMPLATE_EXTRAS.get(c, []))

        if mode == "raw":
            # Strict: only add reads that are not already produced by *previous* commands
            for key in reads:
                if key not in produced:
                    needed.add(key)
        else:
            # "inputs": reads not already produced, plus extras
            for key in reads:
                extras = COMMAND_TEMPLATE_EXTRAS.get(c, {})
                
                if key not in produced:
                    needed.add(key)
            needed.update(extras)

        # Anything this command writes is now "produced"
        produced.update(writes)

    _add_keys(sorted(needed))
    return root

def print_command_summary(cmd: str) -> None:
    """Explain, for one command, which YAML keys it reads/writes (with schema details when known)."""
    io = COMMAND_IO.get(cmd, {})
    if not io:
        raise UserConfigError(f"No YAML documentation for '{cmd}'.")
        return
    summary = io.get("summary", "(no summary)")
    print(f"-------------------- {cmd} ----------------- ")
    print(bullet_fill(summary, left=0, bullet="", extra=0))
    print()   

def print_command_keys(cmds: List[str]) -> None:
    read_keys = []
    write_keys = []
    seen = set()   
    for c in cmds:
        io = COMMAND_IO.get(c, {})
        reads  = io.get("reads", [])
        writes = io.get("writes", [])
        for k in reads:
            if k not in seen:
                read_keys.append(k)
        for k in writes:
            if k not in seen:
                write_keys.append(k)
        for k in reads+writes:
            if k not in seen:
                seen.add(k)
    print("Reads:")
    print_keys(read_keys)
    print("Writes:")
    print_keys(write_keys)
    
def _get_command_io() -> Dict[str, Dict[str, Any]]:
    return COMMAND_IO

def yaml_help_command(opts) -> None:
    """
    Modes:
      - yaml_help --keys [KEY ...]           (specific keys; no KEYs => all keys)
      - yaml_help --cmd  [CMD ...]           (keys for specific commands; no CMD => for every command)
      - yaml_help --keys --cmd CMD [...]     (keys per a set of commands; no CMD => for all commands)
    Priority: when --keys is present, we are in KEYS MODE.
    """
    want_keys = getattr(opts, "keys", None)   # None=absent; []=present no args; [..]=explicit keys
    want_cmds = getattr(opts, "cmd", None)
    COMMAND_IO = _get_command_io()  # validate against schema, not argparse
    if not want_cmds:
        if want_keys is not None:
            # explicit key list ⇒ describe just those keys
            if len(want_keys) > 0:
                print_keys(want_keys)
                return "printed help"
            # all keyes 
            from .schema import SCHEMA
            print_keys(sorted(SCHEMA.keys()))
            return "printed help"
    else:
        if want_cmds is not None:
            unknown = [c for c in want_cmds if c not in COMMAND_IO]
            if unknown:
                raise UserConfigError("yaml_help: unknown command(s): " + ", ".join(unknown))        
        else:
            want_cmds = sorted(COMMAND_IO.keys())
            
        if want_keys is None:
            for c in want_cmds:
                print_command_summary(c)  
                print_command_keys([c])     
            return "printed help"
        else:            
            print_command_keys(want_cmds)
            for c in want_cmds:
                print_command_summary(c)
            return "printed help"

        
