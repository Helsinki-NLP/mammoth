#! /usr/bin/bash
THIS="${BASH_SOURCE[0]}"
DIRNAME="$(cd -- "$(dirname -- "$THIS")" >/dev/null 2>&1 && pwd -P)"
BASENAME=$(basename "$0")
set -euo pipefail

# (c) 2025 Anssi Yli-Jyrä, CC-BY

# This script is for building installing mammoth/feat/integration in a correct way
#
# This script is idenpotent, but it has not been checked against
# accidental overwriting of uncommitted files.
#
# helper/install/build-venv.sh

"$SCRIPT_DIR/clone-branches.sh

# --- Helpers ---------------------------------------------------------------------

. "$SCRIPT_DIR/common.sh"


