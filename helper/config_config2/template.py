from __future__ import annotations

import sys
from typing import List, Dict, Any

from .utils import logger
from .utils import UserConfigError, register_command_io, register_command_template_extras

def _dump_yaml_to(path: str | None, obj: Dict[str, Any]) -> None:
    """Write YAML to a file if path is given, else to stdout (UTF-8)."""
    try:
        import yaml
        try:
            from yaml import CDumper as _YDumper
        except Exception:
            from yaml import Dumper as _YDumper
        text = yaml.dump(obj, Dumper=_YDumper, sort_keys=False, allow_unicode=True)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            sys.stdout.write(text)
    except Exception as e:
        raise RuntimeError(f"yaml_template: failed to serialize YAML: {e}") from e
        

def yaml_template_command(opts):
    schema_known = set(COMMAND_IO.keys())
    unknown = [c for c in cmds if c not in schema_known]
    if unknown:
        raise UserConfigError("yaml_template: unknown command(s): " + ", ".join(unknown))

    tpl = build_yaml_template_for_commands(cmds, mode=mode)
    _dump_yaml_to(outp, tpl)

def yaml_template_command(opts) -> None:
    """
    Build a starter YAML template for one or more commands.
    """
    cmds: List[str] = getattr(opts, "cmd", None)
    if not cmds:
        raise UserConfigError("yaml_template: please provide at least one command via --cmd COMMAND [...].")
    mode: str = getattr(opts, "mode", "inputs")
    outp: str | None = getattr(opts, "out_config", None)

    # fetch the root parser that main() attached
    import argparse
    
    root = getattr(opts, "_root_parser", None)
    registered = set()
    if root is not None:
        spa = next((a for a in root._actions if isinstance(a, argparse._SubParsersAction)), None)
        if spa and spa.choices:
            registered = set(spa.choices.keys())
    
    # Validate commands against registered subcommands OR schema-known commands
    from .schema import COMMAND_IO, build_yaml_template_for_commands

    schema_known = set(COMMAND_IO.keys())
    unknown = [c for c in cmds if c not in schema_known]
    # unknown = [c for c in cmds if c not in registered and c not in schema_known]
    if unknown:
        raise UserConfigError(f"yaml_template: unknown command(s): {', '.join(unknown)}")
    tpl = build_yaml_template_for_commands(cmds, mode=mode)
    _dump_yaml_to(outp, tpl)

    
def register(subparsers):
    p = subparsers.add_parser(
        "yaml_template",
        help="Generate a starter YAML for one or more commands.",
        description=(
            "Generate a minimal input YAML for the specified commands. "
            "Use --mode=inputs (default) to include only required inputs, "
            "--mode=all for union of reads+extras, or --mode=raw for strict reads."
        ),
    )
    p.add_argument("--cmd", nargs="+", metavar="COMMAND", required=True,
                   help="One or more subcommands to target (e.g., complete_language_pairs corpora_schedule).")
    p.add_argument("--mode", choices=["inputs", "all", "raw"], default="inputs",
                   help=("Template mode: 'inputs' (default, only required user inputs), 'all'"
                         "(union of reads+extras), or 'raw' (strict reads only)."))
    p.add_argument("--out_config", metavar="FILE.yaml",
                   help="Write YAML to this file. If omitted, prints to stdout.")
    p.set_defaults(handler=yaml_template_command, _parser=p, _early_exit=True, _mutates_yaml=False)

    register_command_io("yaml_template", {
        "reads": [],  # it doesn't read YAML; it's a generator
        "writes": [], # it writes a new YAML file; not a YAML key
        "summary": "Generate a starter YAML for one or more commands."
    })
    
