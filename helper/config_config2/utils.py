from __future__ import annotations
import sys
import argparse
from typing import Any, Dict, Tuple, Iterable, Optional, Literal

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq


# ---------- Pretty printers (no external deps) ----------

def note(msg: str) -> None:
    print(f"[info] {msg}", file=sys.stderr)

def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)

def err(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)

def rule(char: str = "─", width: int = 70) -> str:
    return char * width

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

# ---------- user-facing config errors ----------

class UserConfigError(Exception):
    """Raised when a required option is missing or invalid (from CLI or YAML)."""

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
        # fmt = "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
        fmt = "%(levelname)s - %(message)s"
        
        # Attach the formatter to the handler so it uses that format.
        ch.setFormatter(logging.Formatter(fmt))
 
        # Add the handler to the logger so messages will be emitted.
        logger.addHandler(ch)

import yaml

class FlowList(list):
    """A list that should be emitted in flow style: [a, b, c]."""
    pass

def _as_flow(seq):
    """Wrap a Python list so it dumps as [a, b, c] with PyYAML."""
    return FlowList(seq)

def _represent_flow_list(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

# Register once (SafeDumper)
yaml.add_representer(FlowList, _represent_flow_list)
yaml.add_representer(type(FlowList()), _represent_flow_list)  # PyYAML quirk

def _wrap_inline_lists(doc: dict) -> dict:
    """
    Make a deep copy and force only specific lists to be dumped inline:
      - top-level: gpu_ranks
      - tasks.*:   enc_sharing_group, dec_sharing_group
      - config_config.sharing_groups.enc_sharing_groups
    Everything else stays in block style (bullets).
    """
    
    from copy import deepcopy
    d = deepcopy(doc)

    # 1) top-level gpu_ranks
    if isinstance(d.get("gpu_ranks"), list):
        d["gpu_ranks"] = _as_flow(d["gpu_ranks"])

    # 2) per-task sharing groups
    tasks = d.get("tasks")
    if isinstance(tasks, dict):
        for tname, t in tasks.items():
            if not isinstance(t, dict):
                continue
            if isinstance(t.get("enc_sharing_group"), list):
                t["enc_sharing_group"] = _as_flow(t["enc_sharing_group"])
            if isinstance(t.get("dec_sharing_group"), list):
                t["dec_sharing_group"] = _as_flow(t["dec_sharing_group"])

    # 3) macro list under config_config.sharing_groups
    cc = d.get("config_config")
    if isinstance(cc, dict):
        if isinstance(cc.get("enc_sharing_groups"), list):
            cc["enc_sharing_groups"] = _as_flow(cc["enc_sharing_groups"])
        if isinstance(cc.get("dec_sharing_groups"), list):
            cc["dec_sharing_groups"] = _as_flow(cc["dec_sharing_groups"])
        sg = cc.get("sharing_groups")
        if isinstance(sg, dict):
            if isinstance(sg.get("enc_sharing_groups"), list):
                sg["enc_sharing_groups"] = _as_flow(sg["enc_sharing_groups"])
            if isinstance(sg.get("dec_sharing_groups"), list):
                sg["dec_sharing_groups"] = _as_flow(sg["dec_sharing_groups"])
            if isinstance(sg.get("enc_layers"), list):
                sg["enc_layers"] = _as_flow(sg["enc_layers"])
            if isinstance(sg.get("dec_layers"), list):
                sg["dec_layers"] = _as_flow(sg["dec_layers"])
    return d

def _set_and_log(task: dict, key: str, value, *, level: str = "info"):
    """Set task[key] = value and emit a log line."""
    task[key] = value
    msg = f"task[{key}] = {value!r}"
    if level == "debug":
        logger.debug(msg)
    else:
        logger.info(msg)

# ---------- YAML helpers (fast loader when available) ----------

# PyYAML has two implementations of its loaders/dumpers:
# - C-accelerated versions (CLoader, CDumper) that are much faster but
#   only available if PyYAML was built with libyaml.
# - Pure-Python fallbacks (Loader, Dumper) that always exist but are slower.
try:
    import yaml
    from yaml import CSafeLoader as YSafeLoader, CSafeDumper as YSafeDumper
except Exception:  # PyYAML may not have C extensions
    import yaml  # type: ignore
    from yaml import SafeLoader as YSafeLoader, SafeDumper as YSafeDumper

def load_yaml(fname):
    # Open the YAML file path 'fname' for reading text.
    with open(fname, 'r') as istr:
        # print(f"Reading config file {fname}...")
        # Parse YAML into a Python object (usually dict).
        config = yaml.safe_load(istr) #, Loader=YSafeLoader)
    # Return the parsed object and the original filename as a tuple.
    return [config, fname]

def save_yaml(doc: dict) -> None:
    _dump_yaml_with_style(doc, style="auto")

def _dump_yaml_with_style(doc: dict, style: Literal["auto", "block", "all-inline"]) -> None:
    """
    style:
      - "auto": use project save_yaml (inlines only gpu_ranks / sharing lists you configured)
      - "block": force all lists/maps to block style (- bullets)
      - "all-inline": force everything to inline flow style ([a, b, c])
    """
    if style == "auto":
        prepared = _wrap_inline_lists(doc.in_config[0])
        serialized = yaml.dump(
            prepared, 
            sort_keys=False,
            default_flow_style=False,  # keep everything block-style by default
            allow_unicode=True)
    else:
        serialized = yaml.dump(
            doc.in_config[0],
            sort_keys=False,
            default_flow_style=(style == "all-inline"),
            allow_unicode=True)

    if getattr(doc, "out_config", None):
        with open(doc.out_config, "w", encoding="utf-8") as f:
            f.write(serialized)
    else:
        print(serialized)

# ---------- CLI/YAML value resolver with nice errors ----------

class ResolvedInputs:
    """Uniform accessor to command inputs + their normalized YAML paths."""
    def __init__(self, values: Dict[str, Any], paths: Dict[str, str]):
        self._values = values
        self._paths = paths

    def get(self, name: str, default: Any = None) -> Any:
        return self._values.get(name, default)

    def path(self, name: str) -> str:
        p = self._paths.get(name)
        if p is None:
            raise KeyError(f"No path known for input '{name}'.")
        return p

    def items(self):
        return self._values.items()

    def __contains__(self, k: str) -> bool:
        return k in self._values

# ----------------------- core resolver -----------------------

def resolve_command_inputs(cmd: str, opts) -> ResolvedInputs:
    """
    Resolve and normalize inputs for a command:
      - Verifies command is known (registered or in COMMAND_IO).
      - Loads first YAML doc (mapping) from opts.in_config.
      - For each read key in COMMAND_IO[cmd]:
          CLI (if not None) overrides YAML.
          YAML is searched (backward compatible):
            new: config_config.<cmd>.<basename>
            old: config_config.<basename> (moved to new)
            top-level reads remain at top level (path == '<basename>')
          Required keys must be present (else raise).
      - Writes normalized YAML back into opts.in_config[0].
      - Returns ResolvedInputs with:
            .get(name)  -> value  (optional -> None)
            .path(name) -> full dotted YAML path
    """
    # ---- argparse registry (loaded commands) ----
    def _registered_commands_from_root() -> set[str]:
        root = getattr(opts, "_root_parser", None)
        if root is None:
            return set()
        spa = next((a for a in root._actions if isinstance(a, argparse._SubParsersAction)), None)
        return set(spa.choices.keys()) if spa and spa.choices else set()

    # ---- basic YAML access helpers ----
    def _first_doc(obj: Any) -> Dict[str, Any]:
        if obj is None:
            raise UserConfigError("No input config loaded; did you pass --in_config FILE.yaml?")
        if isinstance(obj, list):
            if not obj:
                raise UserConfigError("Input config list is empty; expected at least one YAML document.")
            if not isinstance(obj[0], dict):
                raise UserConfigError(f"First YAML document must be a mapping; got {type(obj[0]).__name__}.")
            return obj[0]
        if isinstance(obj, dict):
            return obj
        raise UserConfigError(f"Input config must be a mapping or a list of mappings; got {type(obj).__name__}.")

    def _get_path(doc: Dict[str, Any], path: Iterable[str]) -> Tuple[bool, Any]:
        cur = doc
        parts = list(path)
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                return False, None
            cur = cur[p]
        if not isinstance(cur, dict):
            return False, None
        return (parts[-1] in cur), cur.get(parts[-1])

    def _set_path(doc: Dict[str, Any], path: Iterable[str], value: Any) -> None:
        cur = doc
        parts = list(path)
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value

    def _del_path(doc: Dict[str, Any], path: Iterable[str]) -> None:
        cur = doc
        parts = list(path)
        for p in parts[:-1]:
            cur = cur.get(p, {})
            if not isinstance(cur, dict):
                return
        if isinstance(cur, dict):
            cur.pop(parts[-1], None)


    if cmd not in (_registered_commands_from_root() | set(COMMAND_IO.keys())):
        raise UserConfigError(
            f"Unknown command '{cmd}'. Use '--list-commands' or 'config_config2 yaml_help --all'."
        )
    if cmd not in COMMAND_IO:
        raise UserConfigError(
            f"Command '{cmd}' has no COMMAND_IO entry; make sure it calls register_command_io()."
        )
    reads = list(COMMAND_IO.get(cmd, {}).get("reads", []) or [])
    if not reads:
        logger.warning(f"Command '{cmd}' declares no inputs (reads) in COMMAND_IO.")

    schema = COMMAND_IO.get(cmd, {})
    doc = _first_doc(getattr(opts, "in_config", None))

    values: Dict[str, Any] = {}
    paths:  Dict[str, str]  = {}
    missing_required: list[str] = []

    # For each read key, normalize and pick value
    for dotted in reads:
        # Split into "is_under_cc" and basename
        if dotted.startswith("config_config."):
            basename = dotted.split(".", 1)[1]
            legacy_path = ("config_config", basename)
            norm_path   = ("config_config", cmd, basename)
            full_norm   = f"config_config.{cmd}.{basename}"
            under_cc    = True
        else:
            basename = dotted
            legacy_path = (basename,)
            norm_path   = legacy_path
            full_norm   = basename
            under_cc    = False

        # CLI override (attribute name == basename)
        cli_val = getattr(opts, basename, None)

        have_norm, yaml_norm   = _get_path(doc, norm_path)
        have_legacy, yaml_lege = _get_path(doc, legacy_path) if under_cc else (False, None)

        if cli_val is not None:
            value = cli_val
        elif have_norm:
            value = yaml_norm
        elif have_legacy:
            value = yaml_lege
            # migrate legacy → normalized storage
            _set_path(doc, norm_path, yaml_lege)
            _del_path(doc, legacy_path)
            logger.debug(f"Normalized '{'.'.join(legacy_path)}' → '{'.'.join(norm_path)}'")
        else:
            value = None

        # Required? (by schema)
        required = bool(schema.get(dotted, {}).get("required", False))
        if required and value is None:
            missing_required.append(dotted)

        # Soft shape checks (optional; keeps error messages nicer)
        typ_str = str(schema.get(dotted, {}).get("type", "")).lower()
        if value is not None and typ_str:
            if (typ_str in ("str", "string") or typ_str.startswith("path")) and not isinstance(value, str):
                raise UserConfigError(f"{cmd}: '{dotted}' expects string; got {type(value).__name__}.")
            if (typ_str in ("int", "integer", "number")) and not isinstance(value, int):
                raise UserConfigError(f"{cmd}: '{dotted}' expects integer; got {type(value).__name__}.")
            if (typ_str.startswith("bool")) and not isinstance(value, bool):
                raise UserConfigError(f"{cmd}: '{dotted}' expects boolean; got {type(value).__name__}.")
            if (typ_str.startswith("list")) and not isinstance(value, list):
                raise UserConfigError(f"{cmd}: '{dotted}' expects list; got {type(value).__name__}.")
            if (typ_str.startswith("mapping") or typ_str.startswith("dict") or typ_str.startswith("map")) and not isinstance(value, dict):
                raise UserConfigError(f"{cmd}: '{dotted}' expects mapping; got {type(value).__name__}.")

        # Store results
        values[basename] = value
        paths[basename]  = full_norm

        # Ensure normalized YAML stores the chosen value for config_config.* reads
        if under_cc and value is not None:
            _set_path(doc, norm_path, value)

    if missing_required:
        lines = [
            f"{cmd}: missing required key(s):",
            *[f"  - {k}" for k in missing_required],
            "",
            f"Fill them under 'config_config.{cmd}.*' in YAML, or pass as CLI flags.",
            f"See:  config_config2 yaml_help --cmd {cmd}",
            f"Or generate a starter:  config_config2 yaml_template --cmd {cmd} --mode inputs",
        ]
        if COMMAND_TEMPLATE_EXTRAS.get(cmd):
            lines.append("(Template extras are available for this command.)")
        raise UserConfigError("\n".join(lines))

    return ResolvedInputs(values, paths)


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

#--------------- registration utilities --------------------

COMMAND_IO: dict[str, dict] = {}                           # empty
COMMAND_TEMPLATE_EXTRAS: dict[str, dict] = {}              # cmd -> extras dict to inject in templates

def register_command_io(cmd: str, io: dict) -> None:
    """
    io: { 'reads': [...], 'writes': [...], 'summary': '...' }
    """
    old = COMMAND_IO.get(cmd, {})
    merged = {}
    merged["reads"]  = list(dict.fromkeys((old.get("reads")  or []) + (io.get("reads")  or [])))
    merged["writes"] = list(dict.fromkeys((old.get("writes") or []) + (io.get("writes") or [])))
    if "summary" in old or "summary" in io:
        merged["summary"] = io.get("summary", old.get("summary"))
    COMMAND_IO[cmd] = merged

def register_command_template_extras(cmd: str, extras: dict) -> None:
    """
    Extras merged into yaml_template output for this command (e.g., default stubs under config_config).
    """
    if not extras:
        return
    existing = COMMAND_TEMPLATE_EXTRAS.get(cmd, {})
    # shallow merge is fine for typical extras; deep-merge only if you want
    existing.update(extras)
    COMMAND_TEMPLATE_EXTRAS[cmd] = existing

