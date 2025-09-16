#!/usr/bin/env bash
# check-config-paths.sh — find & validate all paths referenced by a Mammoth/HF/x-transformers YAML
# Usage:  ./check-config-paths.sh /path/to/config.yaml
# Exit codes: 0 = all OK, 1 = usage error, 2 = missing paths

set -euo pipefail

usage() {
  cat <<EOF
Usage: ${0##*/} CONFIG.yaml

Scans CONFIG.yaml for path-like fields (e.g., tasks.*.path_*, *_vocab, *_subword_*,
tensorboard_log_dir, save_model, etc.), prints a deduplicated list, and validates
that they exist. Relative paths are resolved against the directory of CONFIG.yaml.

Exits non-zero if any files/dirs are missing.
EOF
}

[[ $# -eq 1 ]] || { usage >&2; exit 1; }
CONFIG="$1"
[[ -f "$CONFIG" ]] || { echo "❌ No such file: $CONFIG" >&2; exit 1; }
CONFIG_DIR="$(cd "$(dirname "$CONFIG")" && pwd -P)"

# --- Run an embedded Python scanner to extract candidate paths ----------------
# Output format (TSV): KIND<TAB>KEYPATH<TAB>RAWVALUE
#   KIND ∈ {file,dir,path}  (our best guess)
#   KEYPATH is a dotted path to the YAML key for context
#   RAWVALUE is the literal string from YAML
MAP_OUTPUT="$(
python3 - <<'PY' "$CONFIG"
import sys, os, re, json, pathlib
cfg_path = sys.argv[1]

def load_yaml(p):
    try:
        import yaml  # type: ignore
        with open(p, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception:
        # Minimal fallback: parse simple "key: value" and lists; this suits many configs.
        data = {}
        key_stack = [data]
        indent_stack = [0]
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                # strip comments while preserving URLs with # (rare here)
                if '#' in line:
                    line = line.split('#',1)[0]
                if not line.strip():
                    continue
                indent = len(line) - len(line.lstrip(' '))
                while indent_stack and indent < indent_stack[-1]:
                    indent_stack.pop()
                    key_stack.pop()
                cur = key_stack[-1]
                if ':' in line:
                    k, v = line.split(':', 1)
                    k = k.strip().strip('"\'')
                    v = v.strip()
                    if v == '':
                        # start of nested mapping
                        cur[k] = {}
                        key_stack.append(cur[k])
                        indent_stack.append(indent+2)
                    else:
                        # scalar
                        v = v.strip('"\'')
                        cur[k] = v
                elif line.strip().startswith('- '):
                    item = line.strip()[2:].strip().strip('"\'')
                    if not isinstance(cur, list):
                        # turn last inserted key into a list if needed
                        pass
        return data

def is_pathlike(keypath, s):
    if not isinstance(s, str) or not s:
        return False
    # obvious path patterns
    if s.startswith(('/', './', '../')): return True
    if '/' in s: return True
    # common file extensions in these configs
    exts = ('.txt','.tsv','.spm','.model','.onmt','.json','.yaml','.yml','.pt','.ckpt')
    if s.lower().endswith(exts): return True
    # keys that imply paths
    k = keypath.lower()
    hints = ('path','vocab','subword','model','dir','file','save','tensorboard')
    return any(h in k for h in hints)

def guess_kind(keypath, s):
    kl = keypath.lower()
    if s.endswith('/') or any(x in kl for x in ('dir','folder','save','log_dir','output_dir')):
        return 'dir'
    # if it has an extension, assume file
    if re.search(r'\.[A-Za-z0-9]{2,5}$', s): return 'file'
    # if ends with known directory names
    if re.search(r'/(models?|logs?|vocab|data|checkpoints?)/?$', s):
        return 'dir'
    return 'path'

def walk(obj, prefix=[]):
    if isinstance(obj, dict):
        for k,v in obj.items():
            walk(v, prefix+[str(k)])
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            walk(v, prefix+[str(i)])
    else:
        keypath = '.'.join(prefix)
        if is_pathlike(keypath, obj):
            s = str(obj)
            kind = guess_kind(keypath, s)
            print(f"{kind}\t{keypath}\t{s}")

try:
    data = load_yaml(cfg_path)
    if data is not None:
        walk(data, [])
except Exception as e:
    print(f"ERROR: YAML parse failed: {e}", file=sys.stderr)
    sys.exit(1)
PY
)"

# If Python scanned nothing, warn (but still proceed)
if [[ -z "$MAP_OUTPUT" ]]; then
  echo "⚠️  No path-like entries found in: $CONFIG" >&2
fi

# --- Normalize, resolve, and validate ----------------------------------------
# We resolve relative RAWVALUE against CONFIG_DIR.
# We print a deduplicated list and a validation report.

# Store to temp arrays
declare -A SEEN
declare -i missing=0 present=0
printf '### Paths referenced in %s (resolved against %s)\n' "$CONFIG" "$CONFIG_DIR"

# Build an array of "kind<TAB>key<TAB>raw<TAB>resolved"
ALL=()
while IFS=$'\t' read -r KIND KEYP RAW; do
  [[ -z "${RAW:-}" ]] && continue
  # If raw contains whitespace, keep as-is; YAML usually doesn't for filenames.
  if [[ "$RAW" = /* ]]; then
    RES="$RAW"
  else
    RES="$CONFIG_DIR/$RAW"
  fi
  # Collapse .. and .
  RES="$(python3 - <<'PY' "$RES"
import sys,os
print(os.path.normpath(sys.argv[1]))
PY
)"
  key="${KIND}"$'\t'"${KEYP}"$'\t'"${RAW}"$'\t'"${RES}"
  if [[ -z "${SEEN[$key]+x}" ]]; then
    SEEN["$key"]=1
    ALL+=("$key")
  fi
done <<< "$MAP_OUTPUT"

# Print list
for entry in "${ALL[@]}"; do
  IFS=$'\t' read -r KIND KEYP RAW RES <<< "$entry"
  printf '%-5s  %-40s  raw=%s\n' "$KIND" "$RES" "$RAW"
done

echo
echo "### Validation"
for entry in "${ALL[@]}"; do
  IFS=$'\t' read -r KIND KEYP RAW RES <<< "$entry"
  # Decide test: dir vs file vs generic
  if [[ "$KIND" == "dir" ]]; then
    if [[ -d "$RES" ]]; then
      printf '✔ dir   %s\n' "$RES"; ((present++))
    else
      printf '❌ dir   %s   (from %s)\n' "$RES" "$KEYP"; ((missing++))
    fi
  elif [[ "$KIND" == "file" ]]; then
    if [[ -f "$RES" ]]; then
      printf '✔ file  %s\n' "$RES"; ((present++))
    else
      printf '❌ file  %s   (from %s)\n' "$RES" "$KEYP"; ((missing++))
    fi
  else
    # generic path: accept either file or dir
    if [[ -e "$RES" ]]; then
      printf '✔ path  %s\n' "$RES"; ((present++))
    else
      printf '❌ path  %s   (from %s)\n' "$RES" "$KEYP"; ((missing++))
    fi
  fi
done

echo
printf 'Summary: %d present, %d missing\n' "$present" "$missing"

# Non-zero exit if anything missing
if (( missing > 0 )); then
  exit 2
fi
