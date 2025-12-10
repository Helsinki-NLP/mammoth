# job-setup.sh
echo "experiment-specific setup (job-setup.sh) ..."
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
set -euo pipefail  # make failures fatal and undefined vars errors (optional)
: "${SLURM_JOB_NAME:?❌ SLURM_JOB_NAME is not set}" 
: "${MASTER_ADDR:?❌ MASTER_ADDR is not set}"
: "${MASTER_PORT:?❌ MASTER_PORT is not set}"

export EXP_ID=2025-08-21_enfi_1n1g-30m
##########################
# SLURM_JOB_NAME is EXP_ID
##########################
# Require both to be set
: "${EXP_ID:?❌ EXP_ID is empty or unset}"
if [[ -z "${SLURM_JOB_NAME:-}" ]]; then
  # fallback to scheduler record if env missing
  SLURM_JOB_NAME="$(scontrol show -d job "${SLURM_JOB_ID:?}" | awk -F'[= ]' '/JobName=/{print $2; exit}')" || true
fi
: "${SLURM_JOB_NAME:?❌ SLURM_JOB_NAME is empty or unset}"
if [[ "$SLURM_JOB_NAME" != "$EXP_ID" ]]; then 
  echo "❌ Name mismatch: SLURM_JOB_NAME='$SLURM_JOB_NAME' ≠ EXP_ID='$EXP_ID'."
  echo "   Fix with: sbatch --job-name \"$EXP_ID\" …  or export EXP_ID='$SLURM_JOB_NAME'."
  exit 1
fi
echo "... OK: experiment name in slurm batch and job-setup.sh are identical"

# avoid spaces in the file names
JOB_DIR=$PROJDATA/exp/$SLURM_JOB_NAME
LOG_DIR=$JOB_DIR/logs
CONF_DIR=$JOB_DIR
SAVE_DIR=$JOB_DIR/models
TENSOR_DIR=$JOB_DIR/tensorboard
CONF_FILE=$CONF_DIR/conf.yml

export MAMMOTH=$PROJHOME/venv/mammoth
export MAMMOTH=$PROJHOME/venv/mammoth-hf
export CODE_DIR=$PROJHOME/mammoth/mammoth/bin
export CODE_DIR=$PROJHOME/mammoth-hf/mammoth/bin
export TRAIN_SCRIPT="$CODE_DIR/train.py"

mkdir -p "$TENSOR_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$SAVE_DIR"
f="$TRAIN_SCRIPT"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
d="$CONF_DIR"; [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
f="$CONF_FILE"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
d="$JOB_DIR";  [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
d="$MAMMOTH";  [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
d="$CODE_DIR"; [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
d="$SAVE_DIR"; [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
d="$TENSOR_DIR"; [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
d="$LOG_DIR"; [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
echo "... OK: set up all the directories"

export TRAIN_ARGS="-config \"${CONF_FILE}\" -tensorboard_log_dir \"${TENSOR_DIR}\" -tensorboard -tasks tasks"
    # -save_model \"${SAVE_DIR}\" \
    #--reset_optim states 
    #--train_from  $TRAIN_FROM
echo "... OK: set up TRAIN_ARGS"

export PATTERN="slurm"  # slurm or torchrun
export MASTER_ARGS="-master_addr ${MASTER_ADDR} -master_port ${MASTER_PORT}"
export MASTER_ARGS=" "  # 1-node
echo "... OK: set up SLURM and MASTER_ARGS"

export GUARD_MAX_NODES=4  # do not change
# Guard threshold (default 24h). Accepts same Slurm formats.
# Set a different guard with export GUARD_TIME="12:00:00" (12h) or GUARD_TIME="2-00:00:00" (2 days).
export GUARD_TIME="${GUARD_TIME:-0-01:00:00}"
echo "... OK: set up GUARDS"

echo =======================================================================================
echo " EXP_ID           : $EXP_ID"
echo " SLURM_JOB_NAME   : $SLURM_JOB_NAME"
echo " local JOB_DIR    : $JOB_DIR"
echo " local CONF_DIR   : $CONF_DIR"
echo " local CONF_FILE  : $CONF_FILE"
echo " LOG_DIR          : $LOG_DIR"
echo " local SAVE_DIR   : $SAVE_DIR"
echo " local TENSOR_DIR : $TENSOR_DIR"
echo " MAMMOTH          : $MAMMOTH"
echo " local CODE_DIR   : $CODE_DIR"
echo " TRAIN_SCRIPT     : $TRAIN_SCRIPT"
echo " TRAIN_ARGS       : $TRAIN_ARGS"
echo " MASTER_ARGS      : $MASTER_ARGS"
echo " PATTERN          : $PATTERN"
echo " GUARD_MAX_NODES  : $GUARD_MAX_NODES"
echo " GUARD_TIME       : $GUARD_TIME"
echo =======================================================================================
