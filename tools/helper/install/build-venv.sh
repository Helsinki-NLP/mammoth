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
echo "now: singularity exec "$SING_IMAGE"  env -u BASH_ENV  bash -lc "
singularity exec "$SING_IMAGE"  env -u BASH_ENV bash -lc '
            echo "now: python -m venv --system-site-packages '"$VENV"'"
	    python -m venv --system-site-packages "'"$VENV"'"
	    echo "now: . '"$VENV"'/bin/activate"
	    . "'"$VENV"'/bin/activate"
	    echo "now: PIP_REQUIRE_VIRTUALENV=1 python -m pip install --no-user -U pip"
	    export PIP_REQUIRE_VIRTUALENV=1
	    python -m pip install --no-user -U pip
	    echo "now: PIP_REQUIRE_VIRTUALENV=1 python -m pip install --no-user -r '"$THISDIR"'/requirements_lumi.txt"'
	    python -m pip install --no-user -r "'"$THISDIR"'/requirements_lumi.txt"'



echo "To activate the modules, the singularity and the environment:"
echo "    . $BASEDIR/install/module-loads.sh"
echo "The activation does not execute singularity; it is delayed until pip/python/sing-bash commands"
echo "After the activation you can use "
echo "   python"
echo "   pip" 
echo "commands normally and they will be running insider the container.  You can also run"
echo "   sing-bash"
echo "to launch bash inside the activated singularity"



