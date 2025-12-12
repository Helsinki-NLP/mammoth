#! /usr/bin/bash
# (c) 2025 Anssi Yli-Jyrä, CC-BY
# This script builds the singularity-venv combo.
# The venv will be at ../venv

# Usage:
#   srun -A <account> --partition=dev-g --ntasks=1 --gres=gpu:mi250:1 --time=0:10:00 --mem=25G --pty bash
#   sh install/build-venv.sh

set -euo pipefail
export DEBUG=1

# configuration
THIS="${BASH_SOURCE[0]}"   # THIS is the wrapper file path (argv0 label to show in ps/top).
BASENAME=$(basename "$0")  # BASENAME becomes the name you invoked the wrapper as (e.g. python, pip).
THISDIR=$(cd -P -- "$(dirname -- "$THIS")" && pwd) || { echo "cannot resolve THISDIR" >&2; return 1 2>/dev/null || exit 1; }
BASEDIR=$(cd -P -- "$THISDIR/.." && pwd)           || { echo "cannot resolve BASEDIR" >&2; return 1 2>/dev/null || exit 1; }
source $BASEDIR/install/common.sh
check_under "$BASEDIR" "venv/README.md" \
	    "install/common.sh" "install/module-loads.sh" "install/requirements_lumi.txt"

# cleaning
rm -rf $BASEDIR/venv/{bin,include,lib,lib64,pyvenv.cfg,share}

# initialisation
. $BASEDIR/install/module-loads.sh

# building
VENV="$BASEDIR/venv"
singularity exec "$SING_IMAGE" bash -lc '
	    python -m venv --system-site-packages "'"$VENV"'"
	    . "'"$VENV"'/bin/activate"
	    export PIP_REQUIRE_VIRTUALENV=1
	    python -m pip install --no-user -U pip
	    python -m pip install --no-user -r "'"$THISDIR"'/requirements_lumi.txt"'



echo "Usage: initialisation"
echo "    . $BASEDIR/install/module-loads.sh"
echo "Usage: singularity + venv activation @ python"
echo "    python"
echo "    import loguru, frozendict, configargparse, einx"
echo "    import torch"
echo "    pip install   --no-user streamlit"
echo "Usage: singularity + venv activation @ bash"
echo "    sing-bash"
echo "    python -m pip install --no-user -U pip"
echo "    pip uninstall loguru streamlit"
echo "    exit"



