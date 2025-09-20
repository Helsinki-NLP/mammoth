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

# --- config ---------------------------------------------------------------

THIS="${BASH_SOURCE[0]}"   # THIS is the wrapper file path (argv0 label to show in ps/top).
BASENAME=$(basename "$0")  # BASENAME becomes the name you invoked the wrapper as (e.g. python, pip).
THISDIR=$(cd -P -- "$(dirname -- "$THIS")" && pwd) || { echo "cannot resolve THISDIR" >&2; return 1 2>/dev/null || exit 1; }
BASEDIR=$(cd -P -- "$THISDIR/.." && pwd)           || { echo "cannot resolve BASEDIR" >&2; return 1 2>/dev/null || exit 1; }
VENV="$BASEDIR/venv"

source $BASEDIR/install/common.sh
check_under "$BASEDIR" "venv/README.md" \
	    "install/common.sh" "install/module-loads.sh" "install/requirements_lumi.txt"
export DEBUG=1
. $BASEDIR/install/module-loads.sh

rm -rf $VENV/{bin,include,lib,lib64,pyvenv.cfg,share}
singularity exec "$SING_IMAGE" bash -lc '
	    python -m venv --system-site-packages "'"$VENV"'"
	    . "'"$VENV"'/bin/activate"
	    export PIP_REQUIRE_VIRTUALENV=1
	    python -m pip install --no-user -U pip
	    python -m pip install --no-user -r "'"$THISDIR"'/requirements_lumi.txt"'

echo "Built venv $VENV. Now you can start using it, but you need to load modules first."
echo "Usage 1:"
echo "    . $BASEDIR/install/module-loads.sh"
echo "    python"
echo "    import loguru, frozendict, configargparse, einx"
echo "    import torch"
echo "Usage 2:"
echo "    . $BASEDIR/install/module-loads.sh"
echo "    sing-bash"
echo "    python -m pip install --no-user -U pip"
echo "    pip install   --no-user streamlit"
echo "    pip uninstall loguru streamlit"
echo "    exit"



