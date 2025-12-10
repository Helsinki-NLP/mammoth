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
import argparse
import shutil, textwrap
import re
import time

# from typing import List, Tuple, Dict

from . import utils
from . import reduce


#import importlib
#import sys
#import traceback

from .utils import init_logging, save_yaml, UserConfigError, logger
from . import yaml

#def _note(msg: str) -> None:
#    print(f"[info] {msg}", file=sys.stderr)
#
#def _warn(msg: str) -> None:
#    print(f"[warn] {msg}", file=sys.stderr)
#
#def _err(msg: str) -> None:
#    print(f"[error] {msg}", file=sys.stderr)
#
#def _rule(char: str = "─", width: int = 70) -> str:
#    return char * width

def build_parser(debug_load: bool=False) -> argparse.ArgumentParser:
    import importlib, traceback, sys
    FEATURE_MODULES = (# You can comment out what you want - these are now independent
        "pairs",       # 1) Data discovery & expansion: complete_language_pairs
#       "schedule",    # 2) Scheduling & curriculum:    corpora_schedule
#       "clustering",  # 3) Language grouping:          cluster_languages
        "sharing",     # 4) Architecture sharing:       sharing_groups
        "devices",     # 5) Device placement:           allocate_devices
#        "transforms",  # 6) Transforms:                 set_transforms
#        "adapters",    # 7) Adapters:                   adapter_config
#        "translations",# 8) Translation utilities:      translation_configs,
#                       #         remove_temporary_keys, config_all
#        "extras",      # 9) Extras & maintenance:       extra_cpu,
#                       #       extra_fully_shared_hack, extra_copy_gpu_assignment
#        "all",
#        "reduce",
        "yaml",
        "template",
        "clone",
    )
    parser = argparse.ArgumentParser(
        prog="config_config2",
        add_help=False,
        allow_abbrev=False,
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



def _normalize_spaces(s: str | None) -> str:
    # collapse internal whitespace to single spaces, trim ends
    return re.sub(r"\s+", " ", (s or "").strip())

def _term_columns(fallback: int = 80) -> int:
    try:
        return shutil.get_terminal_size(fallback=(fallback, 24)).columns
    except Exception:
        return fallback

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

def list_commands_wrapped(parser: argparse.ArgumentParser, command_io) -> None:
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
        io = command_io.get(name, {})
    
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

        reads = io.get("reads") or []
        if reads:
            preview = ", ".join(reads[:4]) + (", …" if len(reads) > 4 else "")
            print((" " * label_width) + f"[reads: {preview}]")
        print()



def get_subparser(parser: argparse.ArgumentParser, name: str) -> argparse.ArgumentParser | None:
    """Return the subparser for 'name' without printing anything, 
    or None if missing.
    """
    spa = next((a for a in parser._actions if isinstance(a, argparse._SubParsersAction)), None)
    if not spa:
        return None
    return spa.choices.get(name)


class TopHelpAction(argparse.Action):
    """
    Custom -h/--help that supports:
      - no value           -> top-level help
      - value == 'all'     -> help for all subcommands
      - value == COMMAND   -> help for that subcommand
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # find the subparsers action
        sp_action = next(
            (a for a in parser._actions if isinstance(a, argparse._SubParsersAction)),
            None
        )

        def _print_top():
            parser.print_help()
            parser.exit()

        def _print_all():
            if not sp_action or not sp_action.choices:
                _print_top()
            # stable order
            names = sorted(sp_action.choices.keys())
            width_rule = "=" * 78
            for i, name in enumerate(names, 1):
                subp = sp_action.choices[name]
                print(width_rule)
                print(f"{i}. {name}\n")
                subp.print_help()
                print()
            parser.exit()

        def _print_one(cmd):
            if sp_action and cmd in sp_action.choices:
                sp_action.choices[cmd].print_help()
                parser.exit()
            parser.error(f"unknown COMMAND for --help: {cmd}")

        # Dispatch
        if values is None:            # bare --help
            _print_top()
        elif values == "all":         # --help all
            _print_all()
        else:                         # --help COMMAND
            _print_one(values)

def main():
    init_logging()

    # 1) Pre-parse only top-level diagnostic flag(s) without subparsers.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--debug-load", action="store_true")
    pre.add_argument("--dump-registry", action="store_true")   # for debugging arguments
    pre.add_argument("--list-commands", action="store_true")   # listing commands only
    
    pre.add_argument("--explain", metavar="COMMAND")
    pre.add_argument("--show-diff", action="store_true")       # not an early-exit

    pre_args, _ = pre.parse_known_args(sys.argv[1:])

    # 2) Build a FULLY REGISTERED parser (commands added here).
    parser = build_parser(debug_load=pre_args.debug_load)
    parser.add_argument("-h", "--help", nargs="?", action=TopHelpAction,
                        help="Show help. Use '--help COMMAND' for a specific command, or '--help all' for every command.")
    
    if pre_args.list_commands:
        from .utils import COMMAND_IO
        list_commands_wrapped(parser, COMMAND_IO)
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

    # 6) Parse ONCE. At this point subcommands exist.
    opts = parser.parse_args(sys.argv[1:])
    opts._root_parser = parser
    
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
        t0  = time.time()
        msg = handler(opts)
        dt  = time.time() - t0
        logger.info(f"{msg} in {dt:.2f}s")
        
        if getattr(opts, "_early_exit", False):
            return
        
        if getattr(opts, "_mutates_yaml", False):
            save_yaml(opts)  # if your handlers mutate opts.in_config[0]
            
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
