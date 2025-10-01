import sys

# ---------- Pretty printers (no external deps) ----------

def note(msg: str) -> None:
    print(f"[info] {msg}", file=sys.stderr)

def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)

def err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)

def rule(char: str = "─", width: int = 70) -> str:
    return char * width

# ---------- user-facing config errors ----------

class UserConfigError(Exception):
    """Raised when a required option is missing or invalid (from CLI or YAML)."""
# ---------- YAML helpers (fast loader when available) ----------
try:
    import yaml
    from yaml import CLoader as _YLoader, CDumper as _Ydumper
except Exception:  # PyYAML may not have C extensions
    import yaml  # type: ignore
    from yaml import Loader as _YLoader, Dumper as _Ydumper  # type: ignore
        
# ---------- logging ----------

import logging
logger = logging.getLogger("config_config")

def init_logging() -> None:
    """Configure the module-level logger with a stream handler and a clear format."""
    
    # Set the logger severity threshold to INFO, so INFO+ messages are shown.
    logger.setLevel(logging.INFO)
    
    # avoid duplicate handlers if init_logging() is called more than once
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):

        # Create a handler that writes log records to standard error.
        ch = logging.StreamHandler()

        # Define the on-screen text format for each log line.
        fmt = "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
        
        # Attach the formatter to the handler so it uses that format.
        ch.setFormatter(logging.Formatter(fmt))
 
        # Add the handler to the logger so messages will be emitted.
        logger.addHandler(ch)

# ---------- user-facing config errors ----------

class UserConfigError(Exception):
    """Raised when a required option is missing or invalid (from CLI or YAML)."""

# ---------- YAML helpers (fast loader when available) ----------

try:
    import yaml
    from yaml import CLoader as _YLoader, CDumper as _Ydumper
except Exception:  # PyYAML may not have C extensions
    import yaml  # type: ignore
    from yaml import Loader as _YLoader, Dumper as _Ydumper  # type: ignore


# PUT THIS IN: io_helpers.py
def load_yaml(fname):
    # Open the YAML file path 'fname' for reading text.
    with open(fname, 'r') as istr:
        # Parse YAML into a Python object (usually dict).
        config = yaml.safe_load(istr)
    # Return the parsed object and the original filename as a tuple.
    return config, fname

def save_yaml(opts) -> None:
    """Serialize opts.in_config[0] to YAML; write to --out_config or stdout."""
    serialized = yaml.dump(
        opts.in_config[0],
        Dumper=_Ydumper,
        default_flow_style=False,
        allow_unicode=True,
    )
    # If an output file path was provided, write there...
    if getattr(opts, "out_config", None):
        with open(opts.out_config, "w") as f:
            f.write(serialized)
    # otherwise print to standard output...
    else:
        print(serialized)

# ---------- CLI/YAML value resolver with nice errors ----------

from typing import Any, Optional

def coalesce(
    opts,
    cc_opts: dict,
    name: str,
    *,
    cli_flag: Optional[str] = None,
    yaml_key: Optional[str] = None,
    required: bool = False,
    default: Any = None,
    type_desc: str = "value",
) -> Any:
    """
    Return CLI value if given, else YAML config_config value, else default or error.

    opts       : argparse Namespace
    cc_opts    : dict like opts.in_config[0]["config_config"]
    name       : attribute name on opts (e.g., 'temperature')
    cli_flag   : how the flag looks on CLI (for error text), default --<name>
    yaml_key   : key under config_config (for error text), default <name>
    required   : if True, raise UserConfigError when both CLI and YAML are missing
    default    : fallback value if not required
    type_desc  : human description for error text (e.g. 'integer', 'path template')
    """
    cli_flag = cli_flag or f"--{name.replace('_','-')}"
    yaml_key = yaml_key or name

    # prefer CLI (argparse sets attribute; may be None if not provided)
    val = getattr(opts, name, None)
    if val is None:
        val = cc_opts.get(yaml_key, None)

    if val is None:
        if required:
            raise UserConfigError(
                f"Missing {type_desc} for '{name}'. "
                f"Provide {cli_flag} on the command line OR set 'config_config.{yaml_key}' in your YAML."
            )
        return default
    return val

#----------------- yaml diff -------------------------
from copy import deepcopy
from typing import Any, Tuple

def _walk(d: Any, path=()):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from _walk(v, path + (str(k),))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            yield from _walk(v, path + (f"[{i}]",))
    else:
        yield (".".join(path), d)

def diff_yaml(before: dict, after: dict) -> Tuple[list[Tuple[str, Any]], list[Tuple[str, Any]], list[Tuple[str, Any]]]:
    """Return (added, removed, changed) lists of (path, value/new_value)."""
    b = {k: v for k, v in _walk(before)}
    a = {k: v for k, v in _walk(after)}
    added = [(k, a[k]) for k in sorted(a.keys() - b.keys())]
    removed = [(k, b[k]) for k in sorted(b.keys() - a.keys())]
    changed = [(k, (b[k], a[k])) for k in sorted(k for k in a.keys() & b.keys() if a[k] != b[k])]
    return added, removed, changed

def print_yaml_diff(before: dict, after: dict) -> None:
    added, removed, changed = diff_yaml(before, after)
    if not (added or removed or changed):
        print("No YAML changes.")
        return
    print("YAML diff:")
    if added:
        print("\n  Added:")
        for k, v in added:
            print(f"    + {k}: {v}")
    if removed:
        print("\n  Removed:")
        for k, v in removed:
            print(f"    - {k}: {v}")
    if changed:
        print("\n  Changed:")
        for k, (old, new) in changed:
            print(f"    ~ {k}: {old} -> {new}")
