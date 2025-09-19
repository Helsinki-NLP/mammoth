#! /usr/bin/bash
set -euo pipefail

# (c) 2025 Anssi Yli-Jyrä, CC-BY

# This script is for building installing mammoth/feat/integration in a correct way
#
#    cd $GIT_ROOT
#    clone-branches.sh
#    build-venv.sh
#
# This script is idenpotent, but it has not been checked against
# accidental overwriting of uncommitted files.

# --- Helpers ---------------------------------------------------------------

. $COMMON

# --- config ---------------------------------------------------------------

THIS="${BASH_SOURCE[0]}"  # /install/build-venv.sh"
BASENAME=$(basename "$0")
DIRNAME="$(cd -- "$(dirname -- "$THIS")" >/dev/null 2>&1 && pwd -P)"

COMMON="$DIRNAME/../install/common.sh"
CLONE="$DIRNAME/../install/clone-branches.sh"
LOADS="$DIRNAME/../bin/slurm/4-module-loads.sh"
VENV="$DIRNAME/../venv"

check_under "$DIRNAME" \
	    "../install/common.sh" \
	    "../install/clone-branches..." \
	    "../bin/slurm/4-module-loads.sh" \
	    "../venv/README.md"

# --- Clean State------------------------------------------------------------

rm -rf $VENV/{bin,include,lib,lib64,pyvenv.cfg,share}
$CLONE

# --- Inherit Site Packages--------------------------------------------------

# always load site packages and python wrappers first
JOB_NODE_KIND=gpu . $LOADS

# then link them to the venv
python -m venv --system-site-packages $VENV
. $VENV/bin/activate

# --- Add Packages and Close-------------------------------------------------

# minimalistic approach of feat/integration branch
export PIP_USER=no

# We use `pip -e .` to install mammoth to the virtual environment via symlinks
pip install -r $DIRNAME/requirements.txt

deactivate

# --- Usage-------------------------------------------------------------------

# Now on, we do not need venv activation.  The venv will be activated
# automatically by python wrapper that comes with
# `pytorch-rocm-mammoth` module.

# Any helper command will immediately know where is the other mammoth
# directories.

