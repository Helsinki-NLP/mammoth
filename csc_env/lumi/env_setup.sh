#!/bin/bash
# One-time setup on LUMI: create a venv inside the container and pip install MAMMOTH into it.
# The venv inherits the container's ROCm PyTorch (--system-site-packages); pip only adds MAMMOTH
# and the packages the container lacks. Submit with: sbatch csc_env/lumi/env_setup.sh

#SBATCH -A project_xxxxxxxxx
#SBATCH -J mammoth_setup
#SBATCH -o ./log/setup_%j.out
#SBATCH -e ./log/setup_%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -euo pipefail

# ==== EDIT THESE ====
PROJECT_DIR=/scratch/project_xxxxxxxxx           # bind-mounted into the container
VENV=$PROJECT_DIR/$USER/mammoth-venv             # the venv to create; change as you like, but use the same path in the job scripts
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
MAMMOTH_PKG=mammoth-nlp                          # or e.g. mammoth-nlp==0.3.0; for development: -e /path/to/mammoth
# Optional: install MAMMOTH with --no-deps and take the other packages from the container plus this
# file only (e.g. /path/to/mammoth/csc_env/requirements.txt). Empty: pip adds what the container lacks.
REQ_FILE=
# ====================

echo "Starting setup at $(date)"
echo "Venv: $VENV"

singularity exec \
    -B "$PROJECT_DIR:$PROJECT_DIR:rw" \
    "$CONTAINER" \
    bash -c "
    set -euo pipefail
    if [ ! -d '$VENV' ]; then
        python -m venv --system-site-packages '$VENV'
    fi

    torch_before=\$('$VENV/bin/python' -c 'import torch; print(torch.__version__)')

    if [ -n '$REQ_FILE' ]; then
        '$VENV/bin/pip' install --no-deps $MAMMOTH_PKG
        '$VENV/bin/pip' install -r '$REQ_FILE'
        '$VENV/bin/pip' check   # fails if a package MAMMOTH needs is missing or too old
    else
        '$VENV/bin/pip' install $MAMMOTH_PKG
    fi

    '$VENV/bin/python' - <<'EOF'
import torch, transformers, mammoth
print('torch       ', torch.__version__, torch.__file__)
print('GPU         ', torch.cuda.is_available(), torch.version.hip)
print('transformers', transformers.__version__, transformers.__file__)
print('mammoth     ', mammoth.__file__)
EOF
    torch_after=\$('$VENV/bin/python' -c 'import torch; print(torch.__version__)')
    if [ \"\$torch_before\" != \"\$torch_after\" ]; then
        echo \"ERROR: pip replaced the container's PyTorch (\$torch_before -> \$torch_after)\" >&2
        exit 1
    fi
    '$VENV/bin/mammoth_train' --help > /dev/null
    echo 'mammoth_train OK'
"

echo "Finished at $(date)"
