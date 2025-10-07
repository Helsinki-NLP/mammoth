# config_config2/clone.py
from __future__ import annotations
from .utils import logger, UserConfigError, load_yaml, save_yaml, register_command_io, register_command_template_extras, resolve_command_inputs, _dump_yaml_with_style

def clone_command(opts):
    """
    Copy the YAML as-is to --out_config, optionally changing only the list
    formatting style. No in-memory mutation.
    """
    # 1) Normalize & collect command inputs (e.g., --style) under config_config.clone.*
    inputs = resolve_command_inputs("clone", opts)

    # 2) Validate we have an input document
    if not getattr(opts, "in_config", None) or not opts.in_config:
        raise UserConfigError("No input YAML loaded; pass --in_config FILE.yaml")

    # 3) Output path is still a top-level CLI thing (not part of config_config.*)
    out_path = getattr(opts, "out_config", None)
    if not out_path:
        raise UserConfigError("Please provide an output path with --out_config FILE.yaml")

    # 4) Style comes from resolver (CLI overrides YAML), default "auto"
    style = inputs.get("style", "auto")
    if style not in ("auto", "block", "all-inline"):
        raise UserConfigError("--style must be one of: auto, block, all-inline")

    # 5) Dump with the requested style
    _dump_yaml_with_style(opts, style)

    return "wrote {out_path} using style={style}"


def register(subparsers):
    p = subparsers.add_parser(
        "clone",
        help="Copy a YAML file, optionally changing only its list formatting style.",
        description=(
            "Reads --in_config and writes --out_config. "
            "Use --style to control list formatting:\n"
            "  auto       = project defaults (inline only selected lists)\n"
            "  block      = force all lists/maps to block style (- bullets)\n"
            "  all-inline = force everything to inline ([a, b, c])"
        ),
    )
    p.add_argument("--in_config", required=True, type=load_yaml, metavar="FILE.yaml",
                   help="Path to the input YAML.")
    p.add_argument("--out_config", required=True, metavar="FILE.yaml",
                   help="Path to write the cloned YAML.")
    p.add_argument("--style", choices=["auto", "block", "all-inline"], default="auto",
                   help="List formatting style for the output file (default: auto).")

    p.set_defaults(handler=clone_command, _parser=p, _mutates_yaml=False, _early_exit=True)

    register_command_io("clone", {
        "reads" : ["(entire document)"],
        "writes" : ["(entire document, re-serialized)"],
        "summary" : "Copy a YAML file; optionally change how lists are formatted."
    })
    register_command_template_extras("clone", [])
