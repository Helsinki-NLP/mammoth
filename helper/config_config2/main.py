# opt.py
# A friendly, grouped command-line interface for the config tool.
# - Regular help:   python -m config_tool.cli -h
# - Detailed help:  python -m config_tool.cli --all-help
#
# This file only wires arguments and calls into the step functions.
# All business logic lives in the sibling modules.

# import textwrap
#def _wrap(s: str) -> str:
#    """Dedent and strip for neat multi-line help texts."""
#    return textwrap.dedent(s).strip()


# ---------- Module loader with neater errors ----------

import sys
import traceback
import importlib
from typing import List, Tuple, Dict
from . import utils

# config_config2/main.py
import argparse
import importlib
import sys
import traceback

from .utils import init_logging, save_yaml, UserConfigError

def _note(msg: str) -> None:
    print(f"[info] {msg}", file=sys.stderr)

def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)

def _err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)

def _rule(char: str = "─", width: int = 70) -> str:
    return char * width

def build_parser(debug_load: bool=False) -> argparse.ArgumentParser:
    import importlib, traceback, sys
    FEATURE_MODULES = (# You can comment out what you want - these are now independent
        "pairs",       # 1) Data discovery & expansion: complete_language_pairs
        "schedule",    # 2) Scheduling & curriculum:    corpora_schedule
        "clustering",  # 3) Language grouping:          cluster_languages
        "sharing",     # 4) Architecture sharing:       sharing_groups
        "devices",     # 5) Device placement:           allocate_devices
        "transforms",  # 6) Transforms:                 set_transforms
        "adapters",    # 7) Adapters:                   adapter_config
        "translations",# 8) Translation utilities:      translation_configs,
        #         remove_temporary_keys, config_all
        "extras",      # 9) Extras & maintenance:       extra_cpu,
        #       extra_fully_shared_hack, extra_copy_gpu_assignment
    )
    parser = argparse.ArgumentParser(
        prog="config_config2",
        description="Configuration pipeline driver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--debug-load", action="store_true",
                        help="Show full tracebacks for module import/register errors.")
    parser.add_argument("--list-commands", action="store_true",
                        help="List available subcommands and exit.")
    parser.add_argument("--all-help", action="store_true",
                        help="Show top-level help AND the help for every subcommand, then exit.")   
    parser.add_argument("--cmd-help", metavar="NAME",
                        help="Show help for a single subcommand NAME, then exit.")
#    parser.add_argument("--yaml-help", action="store_true",
#                        help="Use '<command> --yaml-help' for focused help, or --yaml-keys-help for specific keys.")
    parser.add_argument("--yaml-all-help", action="store_true",
                        help="Describe the YAML structure (all keys).")
    parser.add_argument("--yaml-keys-help", metavar="KEY", nargs="+",
                        help="Describe specific YAML keys (e.g. config_config.temperature tasks).")
    parser.add_argument("--explain", metavar="COMMAND",
                        help="Explain what a command reads/writes in the YAML.")
    parser.add_argument("--show-diff", action="store_true",
                        help="After running a command, print a YAML diff of changes.")
    
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    pkg = __package__ or "config_config2"
    for name in FEATURE_MODULES:
        fq = f"{pkg}.{name}"
        try:
            mod = importlib.import_module(fq)
        except Exception as e:
            print(f"[warn] import failed: {fq}: {e}", file=sys.stderr)
            if debug_load:
                traceback.print_exc()
            continue
        reg = getattr(mod, "register", None)
        if not reg:
            print(f"[warn] {fq} has no register(subparsers)", file=sys.stderr)
            continue
        try:
            reg(subparsers)
        except Exception as e:
            print(f"[warn] register() failed in {fq}: {e}", file=sys.stderr)
            if debug_load:
                traceback.print_exc()
            continue

    return parser


import argparse

def print_subcommand_help(parser: argparse.ArgumentParser, cmd_name: str) -> bool:
    """
    Print the argparse help for a single registered subcommand.
    Returns True if printed, False if the command doesn't exist.
    """
    # find the subparsers action
    spa = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
    if spa is None:
        print("(no subcommands are registered)")
        return False

    sub = spa.choices.get(cmd_name)
    if sub is None:
        return False

    print(sub.format_help())
    return True

def print_all_help(parser: argparse.ArgumentParser, *, stream=None) -> None:
    """Print top-level help + each subcommand's help."""
    if stream is None:
        stream = sys.stdout

    # 1) Top-level help
    stream.write(parser.format_help())
    stream.write("\n")

    # 2) Each subcommand
    # Find the subparsers action
    subparsers_action = next(
        (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
        None
    )
    if not subparsers_action:
        return

    # Print in alphabetical order
    for name in sorted(subparsers_action.choices):
        sub = subparsers_action.choices[name]
        stream.write("=" * 78 + "\n")
        stream.write(f"{name} — subcommand help\n")
        stream.write("=" * 78 + "\n")
        stream.write(sub.format_help())
        stream.write("\n")



import argparse, shutil, textwrap, sys, re

def _term_columns(fallback: int = 80) -> int:
    try:
        return shutil.get_terminal_size(fallback=(fallback, 24)).columns
    except Exception:
        return fallback

def _normalize_spaces(s: str | None) -> str:
    # collapse internal whitespace to single spaces, trim ends
    return re.sub(r"\s+", " ", (s or "").strip())

def _wrap_summary_for_label(text: str, label_width: int, min_content: int = 20, max_cols: int | None = None) -> list[str]:
    cols = _term_columns() if max_cols is None else max_cols
    content_width = max(min_content, cols - label_width)
    # wrap *only* to the content width (not full width)
    return textwrap.wrap(
        _normalize_spaces(text),
        width=content_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

def list_commands_wrapped(parser: argparse.ArgumentParser) -> None:
    # find the subparsers action
    spa = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
    if spa is None or not spa.choices:
        print("No commands are registered.")
        return

    # build a mapping name -> one-line help (from add_parser(..., help="..."))
    help_map = {}
    for ca in getattr(spa, "_choices_actions", []):  # stable enough in practice
        help_map[getattr(ca, "choice", None)] = getattr(ca, "help", "") or ""

    names = sorted(spa.choices)
    longest = max(len(n) for n in names)
    label_width = 4 + longest + 2        # "  - " + name + "  "

    print("Available commands:")
    for name in names:
        sub = spa.choices[name]          # ArgumentParser for this subcommand
        # prefer description's first paragraph; fall back to short help
        summary = (sub.description or help_map.get(name, "") or "")
        # wrap to the *content* width (terminal_cols - label_width)
        wrapped = _wrap_summary_for_label(summary, label_width=label_width)

        label = f"  - {name:<{longest}}  "
        if wrapped:
            # first line
            print(label + wrapped[0])
            # continuation lines
            pad = " " * label_width
            for line in wrapped[1:]:
                print(pad + line)
        else:
            # no text available
            print(label + "(no description)")

    # give a hint about YAML keys too
    from .schema import COMMAND_IO
    io = COMMAND_IO.get(name, {})
    reads = io.get("reads", [])
    if reads:
        preview = ", ".join(reads[:2]) + ("…" if len(reads) > 2 else "")
        print((" " * label_width) + f"[reads: {preview}]")

import argparse, sys, shutil, textwrap, re
from .schema import COMMAND_IO, SCHEMA  # make sure these exist in schema.py

def _cols(fallback=80):
    try:
        return shutil.get_terminal_size(fallback=(fallback, 24)).columns
    except Exception:
        return fallback

def _wrap(text: str, indent: int = 2, width: int | None = None) -> str:
    width = width or max(40, _cols() - indent)
    text = re.sub(r"\s+", " ", (text or "").strip())
    return textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * indent)

def _rule(title: str | None = None, ch: str = "─"):
    w = _cols()
    if not title:
        return ch * w
    t = f" {title} "
    k = max(0, w - len(t))
    left = k // 2
    right = k - left
    return (ch * left) + t + (ch * right)

def print_cmd_yaml_help(parser: argparse.ArgumentParser, cmd_name: str) -> bool:
    """Print a full, focused help page for one command (argparse help + YAML IO + key schema)."""
    # 1) find the subparser
    spa = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
    if not spa:
        print(f"No subcommands are registered.", file=sys.stderr)
        return False
    sub = spa.choices.get(cmd_name)
    if not sub:
        print(f"No such subcommand: {cmd_name}", file=sys.stderr)
        return False

    # 2) argparse help for that command
    print(_rule(f"{cmd_name} — command help"))
    print(sub.format_help().rstrip())
    print()

    # 3) YAML IO summary for that command
    io = COMMAND_IO.get(cmd_name, {})
    reads = io.get("reads", []) or []
    writes = io.get("writes", []) or []
    summary = io.get("summary", "")

    print(_rule(f"{cmd_name} — YAML I/O"))
    if summary:
        print(_wrap(summary, indent=2))
        print()

    if reads:
        print("Reads:")
        for k in reads:
            desc = (SCHEMA.get(k, {}) or {}).get("description", "")
            line = f"{k}" + (f" — {desc}" if desc else "")
            print(_wrap(line, indent=2))
    else:
        print("Reads:\n  (none)")

    print()
    if writes:
        print("Writes:")
        for k in writes:
            desc = (SCHEMA.get(k, {}) or {}).get("description", "")
            line = f"{k}" + (f" — {desc}" if desc else "")
            print(_wrap(line, indent=2))
    else:
        print("Writes:\n  (none)")

    # 4) Detailed schema for keys this command touches
    keys = list(dict.fromkeys(reads + writes))  # preserve order, unique
    if keys:
        print()
        print(_rule("Schema for referenced keys"))
        for k in keys:
            meta = SCHEMA.get(k, {})
            print(f"\n{k}")
            t = meta.get("type"); req = meta.get("required"); desc = meta.get("description"); ex = meta.get("example")
            if t is not None:    print(_wrap(f"type: {t}", indent=4))
            if req is not None:  print(_wrap(f"required: {req}", indent=4))
            if desc:             print(_wrap(desc, indent=4))
            if ex is not None:   print(_wrap(f"example: {ex}", indent=4))

    print()
    print(_rule())
    return True
    
def main():
    init_logging()

    # 1) Pre-parse only top-level diagnostic flag(s) without subparsers.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--debug-load", action="store_true")
    pre.add_argument("--dump-registry", action="store_true")   # for debugging arguments
    pre.add_argument("--all-help", action="store_true")        # for showing all helps
    pre.add_argument("--cmd-help", metavar="NAME")             # help on command
    pre.add_argument("--cmd-yaml-help", metavar="NAME")        # yaml help on command
    pre.add_argument("--list-commands", action="store_true")   # listing commands only
    pre.add_argument("--yaml-all-help", action="store_true")
    pre.add_argument("--yaml-keys-help", nargs="+")
    pre.add_argument("--explain", metavar="COMMAND")
    pre.add_argument("--show-diff", action="store_true")       # not an early-exit
    pre_args, _ = pre.parse_known_args(sys.argv[1:])

    # 2) Build a FULLY REGISTERED parser (commands added here).
    parser = build_parser(debug_load=pre_args.debug_load)
    
    # 3) Show all helps (exit early)
    if pre_args.all_help:
        print_all_help(parser)
        return

    # 4) Implement --list-commands (early exit).
    if pre_args.list_commands:
        list_commands_wrapped(parser)
        return
    
    # 5) Help on command
    if pre_args.cmd_help:
        if not print_subcommand_help(parser, pre_args.cmd_help):
            print(f"No such subcommand: {pre_args.cmd_help}", file=sys.stderr)
            # Show a compact TOC to help the user
            list_commands_wrapped(parser)
            sys.exit(2)
        return

    # 6) Print the registry (early exit)
    if pre_args.dump_registry:
        # list subcommands
        for act in parser._actions:
            if isinstance(act, argparse._SubParsersAction):
                names = sorted(act.choices.keys())
                print("Registered commands:")
                for n in names:
                    print(f"  - {n}")
                break
        return

    # 7) YAML all help (early exit)
    if pre_args.yaml_all_help:
        from .schema import print_schema
        print_schema()
        return
    
    if pre_args.cmd_yaml_help:
        if not print_cmd_yaml_help(parser, pre_args.cmd_yaml_help):
            # Helpful fallback: show a compact list
            list_commands_wrapped(parser) if 'list_commands_wrapped' in globals() else None
            sys.exit(2)
        return

    if pre_args.yaml_keys_help:
        from .schema import print_schema
        print_schema(pre_args.yaml_keys_help)
        return

    # Command IO explanation (early exit)
    if pre_args.explain:
        from .schema import print_command_yaml_help
        print_command_yaml_help(pre_args.explain)
        return

    # 6) Parse ONCE. At this point subcommands exist.
    opts = parser.parse_args(sys.argv[1:])
    from copy import deepcopy
    if pre_args.show_diff:
        if getattr(opts, "in_config", None):
            before = deepcopy(opts.in_config[0])
        else:
            before = None
    
    # 7) Dispatch.
    handler = getattr(opts, "handler", None)
    if handler is None:
        parser.print_help()
        sys.exit(2)

    try:
        handler(opts)
        save_yaml(opts)   # if your handlers mutate opts.in_config[0]
    except UserConfigError as e:
        print("\n" + "-"*70, file=sys.stderr)
        print(str(e), file=sys.stderr)
        print("-"*70 + "\n", file=sys.stderr)
        sys.exit(2)

    # after
    if pre_args.show_diff and before is not None:
        from .utils import print_yaml_diff
        print_yaml_diff(before, opts.in_config[0])

    
if __name__ == "__main__":
    main()
