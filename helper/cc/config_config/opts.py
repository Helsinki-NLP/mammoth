# cli.py
# A friendly, grouped command-line interface for the config tool.
# - Regular help:   python -m config_tool.cli -h
# - Detailed help:  python -m config_tool.cli --help-all
#
# This file only wires arguments and calls into the step functions.
# All business logic lives in the sibling modules.

# from __future__ import annotations

import argparse
import textwrap

# from .utils import init_logging, save_yaml, UserConfigError
# from .utils import init_logging, save_yaml, logger  # logger only used for consistent formatting in messages
# from .io_helpers import load_yaml, load_distmat_csv

from . import (# Import modules that each expose register(subparsers)
    schedule,       # corpora_schedule
    clustering,     # cluster_languages
    sharing,        # sharing_groups
    transforms,     # set_transforms
    devices,        # allocate_devices
    adapters,       # adapter_config
    translations,   # complete_language_pairs, translation_configs, remove_temporary_keys, config_all
    extras,         # extra_cpu, extra_fully_shared_hack, extra_copy_gpu_assignment
)

def _wrap(s: str) -> str:
    """Dedent and strip for neat multi-line help texts."""
    return textwrap.dedent(s).strip()

def build_parser() -> argparse.ArgumentParser:
    description = _wrap("""
        Configuration pipeline driver.

        Commands are grouped by stage:
          1) Data discovery & expansion: complete_language_pairs
          2) Scheduling & curriculum:    corpora_schedule
          3) Language grouping:          cluster_languages
          4) Architecture sharing:       sharing_groups
          5) Device placement:           allocate_devices
          6) Transforms:                 set_transforms
          7) Adapters:                   adapter_config
          8) Translation utilities:      translation_configs, remove_temporary_keys, config_all
          9) Extras & maintenance:       extra_cpu, extra_fully_shared_hack, extra_copy_gpu_assignment
    """)
    parser = argparse.ArgumentParser(
        prog="config_config",
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--help-all", action="store_true",
                        help="Show grouped overview after normal help.")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")
    for mod in (translations, schedule, clustering, sharing, devices, transforms, adapters, extras):
        mod.register(subparsers) # Let each module attach its own subparser and args
    return parser

def main():
    init_logging()
    
    parser = build_parser()

    if "--help-all" in sys.argv:
        parser.print_help()
        sys.exit(0)
    if len(sys.argv) == 1: # require a subcommand
        parser.print_help()
        sys.exit(2)

    opts = parser.parse_args()
    try:
        # Each subparser set a 'handler'
        handler = getattr(opts, "handler", None)
        if handler is None:
            parser.print_help()
            sys.exit(2)
        handler(opts)   # run the command
        save_yaml(opts) # and serialize the result

    except UserConfigError as err:
        # neat, user-facing error with exit code 2 (bad usage)
        print(f"\n✖ {err}\n")
        print("Hint: run this command with -h or --help for usage details.")
        sys.exit(2)




