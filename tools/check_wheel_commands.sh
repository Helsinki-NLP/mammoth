#!/usr/bin/env bash
# Install a built wheel into a fresh venv and run every mammoth_* console script with --help.
#
#   python -m build && tools/check_wheel_commands.sh dist/mammoth_nlp-*.whl
#
# The venv lives in a temp dir and the commands run from there, so the source tree cannot stand in for
# what the wheel really installs.
# Set TORCH_INDEX_URL (e.g. https://download.pytorch.org/whl/cpu) to pick the torch build.
set -euo pipefail

wheel=$(realpath "${1:?usage: $0 path/to/mammoth_nlp-*.whl}")
python=${PYTHON:-python3}

workdir=$(mktemp -d)
trap 'rm -rf "$workdir"' EXIT
cd "$workdir"

"$python" -m venv venv
venv/bin/python -m pip install --quiet --upgrade pip
if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    venv/bin/python -m pip install --quiet torch --index-url "$TORCH_INDEX_URL"
fi
venv/bin/python -m pip install --quiet "${wheel}[server]"

venv/bin/python - <<'EOF'
import os
import subprocess
import sys
from importlib.metadata import distribution

bindir = os.path.dirname(sys.executable)
names = sorted(ep.name for ep in distribution('mammoth-nlp').entry_points if ep.group == 'console_scripts')
failed = []
for name in names:
    result = subprocess.run([os.path.join(bindir, name), '--help'], capture_output=True, text=True)
    ok = result.returncode == 0 and 'usage' in result.stdout.lower()
    print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        failed.append(name)
        print(result.stderr[-2000:], file=sys.stderr)
if not names:
    sys.exit('no console_scripts found in the wheel')
if failed:
    sys.exit(f'{len(failed)} of {len(names)} commands failed: {", ".join(failed)}')
EOF
