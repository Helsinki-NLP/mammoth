echo integrity-check.sh ...
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
is_sourced()  { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
require_set() {
  local v
  for v; do
    # ${!v-} expands to empty if unset (safe with set -u)
    if [[ -z "${!v-}" ]]; then
      printf '❌ %s must be set\n' "$v" >&2
      return 1
    fi
  done
}
require_set SYSTEM SLURM_JOB_NAME SLURM_SUBMIT_DIR || is_sourced && return 1 || exit 1;
require_set JOB_NAME JOB_PATTERN JOB_SCRIPT JOB_ARGS || is_sourced && return 1 || exit 1;
require_set GUARD_TIME GUARD_MAX_NODES || is_sourced && return 1 || exit 1;

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
SUBMIT_BASENAME="$(basename -- "$SUBMIT_DIR")"
JOB_NAME="${SLURM_JOB_NAME:-unknown}"
if [[ "$JOB_NAME" != "$SUBMIT_BASENAME" ]]; then
  echo "⚠️  Job name ('$JOB_NAME') != directory name ('$SUBMIT_BASENAME') from $SUBMIT_DIR" >&2
  exit 1
else
  echo "✅ Job name matches directory: $JOB_NAME"
fi

echo ==============================================
echo " SYSTEM (.profile) : $SYSTEM"
echo " SLURM_JOB_NAME    : $SLURM_JOB_NAME"
echo " JOB_NAME          : $JOB_NAME"
echo " SLURM_SUBMIT_DIR  : $SLURM_SUBMIT_DIR"
echo " JOB_PATTERN       : $JOB_PATTERN"
echo " JOB_SCRIPT        : $JOB_SCRIPT"
echo " JOB_ARGS          : $JOB_ARGS"
echo " GUARD_TIME        : $GUARD_TIME"
echo " GUARD_MAX_NODES   : $GUARD_MAX_NODES"
echo ==============================================

export JOB_DIR="${SLURM_SUBMIT_DIR:-$(pwd -P)}"
export JOB_SLURM=base/mammoth-helper/helper/bin/slurm
export JOB_VENV=base/venv/mammoth-hf
export JOB_LIB=base/helper/lib
export JOB_LOGS=logs
echo ==============================================
echo " JOB_DIR (`pwd`)   : $JOB_DIR"
echo " JOB_SLURM         : ./$JOB_SLURM"
echo " JOB_VENV          : ./$JOB_VENV"
echo " JOB_LIB           : ./$JOB_LIB"
echo " JOB_LOGS          : ./$JOB_LOGS"
echo ==============================================

check_under() {
  local base="$1"; shift
  local name f missing=0
  for name in "$@"; do
    f="$base/$name"
    [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; missing=1; }
  done
  (( missing == 0 )) || exit 1
}
check_under "$JOB_SLURM" \
  1-integrity-checks.sh \
  2-distributed-setup.sh \
  3-sanity-checks.sh \
  4-module-loads.sh \
  5-comms-setup.sh \
  6-task-wrapper.sh \
  7-local-setup.sh
check_under "$JOB_VENV" \
  activate  

INTEGRITY_CHECKS_OK=1

