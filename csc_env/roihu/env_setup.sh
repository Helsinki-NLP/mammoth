#!/bin/bash
# One-time setup on Roihu: create a venv on top of the PyTorch module and pip install MAMMOTH into it.
# The venv inherits the module's PyTorch (--system-site-packages); pip only adds MAMMOTH and the
# packages the module lacks. Submit with: sbatch csc_env/roihu/env_setup.sh

#SBATCH -A project_xxxxxxx
#SBATCH -J mammoth_setup
#SBATCH -o ./log/setup_%j.out
#SBATCH -e ./log/setup_%j.err
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:gh200:1

set -euo pipefail

# ==== EDIT THESE ====
VENV=/scratch/project_xxxxxxx/$USER/mammoth-venv   # the venv to create (the job scripts use the same path)
MAMMOTH_PKG=mammoth-nlp                            # or e.g. mammoth-nlp==0.3.0; for development: -e /path/to/mammoth
# Optional: install MAMMOTH with --no-deps and take the other packages from the module plus this
# file only (e.g. /path/to/mammoth/csc_env/requirements.txt). Empty: pip adds what the module lacks.
REQ_FILE=
# ====================

module purge
module load python-pytorch

echo "Starting setup at $(date)"
echo "Venv: $VENV"

if [ ! -d "$VENV" ]; then
    python -m venv --system-site-packages "$VENV"
fi
"$VENV/bin/pip" install --upgrade pip
torch_before=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')

if [ -n "$REQ_FILE" ]; then
    "$VENV/bin/pip" install --no-deps $MAMMOTH_PKG
    "$VENV/bin/pip" install -r "$REQ_FILE"
    # fail if a package MAMMOTH needs is missing or too old (unlike pip check, ignores the module's own conflicts)
    "$VENV/bin/python" -P - <<'EOF'
import sys
from importlib.metadata import PackageNotFoundError, distribution, version
try:
    from packaging.requirements import Requirement
except ImportError:  # a bare venv: use the copy vendored in pip
    from pip._vendor.packaging.requirements import Requirement
problems = []
for spec in distribution('mammoth-nlp').requires or []:
    req = Requirement(spec)
    if req.marker is not None and not req.marker.evaluate({'extra': ''}):
        continue  # optional extra, e.g. [server]
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        problems.append(f'{req}: not installed')
        continue
    if not req.specifier.contains(installed, prereleases=True):
        problems.append(f'{req}: found {installed}')
if problems:
    sys.exit('mammoth-nlp requirements not met (add them to REQ_FILE):\n  ' + '\n  '.join(problems))
print('mammoth-nlp requirements OK')
EOF
else
    "$VENV/bin/pip" install $MAMMOTH_PKG
fi

"$VENV/bin/python" -P - <<'EOF'
import torch, transformers, mammoth
print('torch       ', torch.__version__, torch.__file__)
print('GPU         ', torch.cuda.is_available(), torch.version.cuda)
print('transformers', transformers.__version__, transformers.__file__)
print('mammoth     ', mammoth.__file__)
EOF
torch_after=$("$VENV/bin/python" -c 'import torch; print(torch.__version__)')
if [ "$torch_before" != "$torch_after" ]; then
    echo "ERROR: pip replaced the module's PyTorch ($torch_before -> $torch_after)" >&2
    exit 1
fi
"$VENV/bin/mammoth_train" --help > /dev/null
echo 'mammoth_train OK'

echo "Finished at $(date)"
