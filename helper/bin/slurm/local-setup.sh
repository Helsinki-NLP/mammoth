# local-setup.sh — SOURCE this *inside* srun (per task).
# Only per-task bits (need SLURM_*ID). Pick PATTERN explicitly: slurm | torchrun.
echo local-setup...
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }

# Don’t set any *VISIBLE_DEVICES before srun.  Set them inside the srun task, and only
# when you’re using the Slurm pattern (1 task per GPU). For the torchrun pattern, don’t
# set them at all (or explicitly unset) so all local GPUs are visible to the launcher.
#
# Do not do this (!):
#  export HIP_VISIBLE_DEVICES=$SLURM_LOCALID # only correct inside srun
#  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5   # 0,1,2,3,4,5 lines are fragile/incorrect
#  export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5   # 0,1,2,3,4,5 lines are fragile/incorrect
# LUMI has 8 GCDs; plus you shouldn’t hardcode).
#
# Never export *VISIBLE_DEVICES in the batch script before srun. Outside srun there’s
# no SLURM_LOCALID, and you’ll end up exposing all GPUs to every task (the “bad” case
# you saw in the LUMI AI course) .
#

set -euo pipefail

# --- must be inside an srun-launched task ---
: "${SLURM_PROCID:?❌ SLURM_PROCID is not set - must be run insider srun}"
: "${SLURM_LOCALID:?❌ SLURM_LOCALID is not set}"
: "${SLURM_NODEID:?❌ SLURM_NODEID is not set}"
: "${SLURM_NTASKS:?❌ SLURM_NTASKS is not set}"
: "${SYSTEM:?❌ SYSTEM is not set}"
: "${PATTERN:?❌ PATTERN is not set}"
: "${LOG_DIR:?❌ LOG_DIR is not set}"
: "${EXP_ID:?❌ EXP_ID is not set}"

###################################
# step-level safety check for nodes
###################################
# It makes sure that the current srun you’re inside actually
# has ≥1 node, not just that the overall job requested nodes.  SLURM_NNODES describes
# the current job step created by srun (it can be different—e.g., you requested 4 nodes
# for the job but launch an srun step on just 1).
# If the step-level count is zero/empty, you’ve likely launched the script outside srun,
# or mis-specified srun options so the step has no resources. 
# Compare: SLURM_JOB_NUM_NODES / NumNodes= (from scontrol) describe the job allocation.
# Job asked for 4 nodes (NODES=4). You run srun --nodes=1 ... → inside the step SLURM_NNODES=1, check passes.
if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
  NODES="$SLURM_JOB_NUM_NODES"
else
  JOBINFO="$(scontrol show -d job ${SLURM_JOB_ID:?})"
  NODES="$(awk -F'[= ]' '/NumNodes=/{print $2; exit}' <<<"$JOBINFO")"
fi
: "${NODES:?could not determine node count}"
STEP_NODES="${SLURM_NNODES:-$NODES}"
if (( STEP_NODES == 0 )); then
  echo "❌ ERROR: step has 0 nodes. Check your srun --nodes/--ntasks-per-node." >&2
  exit 1  #  The script fails fast with a clear message.
fi

# ---------- GPU visibility (safe defaults) ----------
# Pick the right env var names once
case "$SYSTEM" in
  lumi)           GPU_ENV1=HIP_VISIBLE_DEVICES; GPU_ENV2=ROCR_VISIBLE_DEVICES ;;
  puhti|mahti|*)  GPU_ENV1=CUDA_VISIBLE_DEVICES; GPU_ENV2= ;;
esac

case "$PATTERN" in
  slurm)
    # One Slurm task per GPU → bind task to GPU == LOCAL_RANK
    export RANK="$SLURM_PROCID"
    export LOCAL_RANK="$SLURM_LOCALID"
    export WORLD_SIZE="$SLURM_NTASKS"
    export "$GPU_ENV1"="$SLURM_LOCALID"
    [[ -n "${GPU_ENV2:-}" ]] && export "$GPU_ENV2"="$SLURM_LOCALID"
    ;;
  torchrun)
    # torchrun spawns per-GPU workers and sets ranks itself → expose all local GPUs
    [[ -n "${GPU_ENV1:-}" ]] && unset "$GPU_ENV1"
    [[ -n "${GPU_ENV2:-}" ]] && unset "$GPU_ENV2"
    # Optional: many users set this for torchrun
    export NODE_RANK="${SLURM_NODEID:-0}"
    ;;
  *)
    echo "ERROR: PATTERN must be 'slurm' or 'torchrun' (got '$PATTERN')." >&2
    exit 1
    ;;
esac

# --- per-node cache that needs SLURM_NODEID (useful on ROCm; harmless elsewhere) ---
BASE=${TMPDIR:-/tmp}   # uses the job’s local scratch if your system provides it.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$BASE/$USER-miopen-${SLURM_JOB_ID:-0}-${SLURM_NODEID:-0}}"
# Including both SLURM_JOB_ID and SLURM_NODEID avoids collisions across jobs and nodes.
# Keep this inside srun so SLURM_NODEID exists.
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}"
mkdir -p -- "$MIOPEN_USER_DB_PATH" >/dev/null 2>&1 || true
#
# Match OpenMP threads to cpus-per-task (data loaders often separate; this is safe)
if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 0 ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
  export OMP_PLACES="${OMP_PLACES:-cores}"
fi


# Optional: quick affinity print + guardrails (teach-by-error)
# The following prints what each task sees: node, local rank, the visible GPU index,
# CPU affinity list, and NUMA binding. It also raises errors for common footguns.
print_affinity() {
  local pid=$$
  local gpu="${CUDA_VISIBLE_DEVICES-}"; [[ -z "$gpu" ]] && gpu="${ROCR_VISIBLE_DEVICES-}"; [[ -z "$gpu" ]] && gpu="${HIP_VISIBLE_DEVICES-}"
  local cpu_list
  cpu_list="$(taskset -pc "$pid" 2>/dev/null | awk -F': ' '{print $2}' | xargs || true)"
  echo "AFFINITY node=${SLURM_NODEID} rank=${SLURM_PROCID} lrank=${SLURM_LOCALID} gpu_env=${gpu:-unset} cpus={${cpu_list:-unknown}}"
  if [[ -z "$cpu_list" ]]; then
    echo "ERROR: No CPU affinity visible. Use --cpu-bind=cores and set --cpus-per-task in SBATCH/srun." >&2
    exit 1
  fi
  if [[ "$PATTERN" == "slurm" ]]; then
    if [[ "$gpu" =~ , ]]; then
      echo "ERROR: Multiple GPUs visible to one task ('$gpu'). In 'slurm' pattern each task must see exactly one GPU." >&2
      exit 1
    fi
    if [[ "$gpu" =~ ^[0-9]+$ ]] && [[ "$gpu" != "$SLURM_LOCALID" ]]; then
      echo "ERROR: GPU id ($gpu) != LOCAL_RANK ($SLURM_LOCALID). Export *VISIBLE_DEVICES* inside srun." >&2
      exit 1
    fi
  fi
}

# Start lightweight GPU monitor
MON_PID=""
start_monitor() {
  if [[ "$SYSTEM" == "puhti" || "$SYSTEM" == "mahti" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi dmon -s mu -d 5 -o TD > "${LOG_DIR}/gpu_load-${EXP_ID}-${PPID}.log" &
      MON_PID=$!
    fi
  else # LUMI / ROCm
    if command -v rocm-smi >/dev/null 2>&1; then
      # sample every 5s: util, power, temp, vram
      while true; do
        rocm-smi --showuse --showtemp --showpower --showmemuse || true
        sleep 5
      done > "${LOG_DIR}/gpu_load-${EXP_ID}-${PPID}.log" &
      MON_PID=$!
    fi
  fi
}
stop_monitor() { [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null || true; }


echo ==============================================
echo " SYSTEM                    : $SYSTEM"
echo " PATTERN                   : $PATTERN"
echo " SLURM_JOB_NUM_NODES       : $SLURM_JOB_NUM_NODES"
echo " NODES                     : NODES"
echo " SLURM_NNODES              : $SLURM_NNODES"
echo " SLURM_NODEID              : $SLURM_NODEID"
echo " SLURM_PROCID              : $SLURM_PROCID"
echo " RANK                      : ${RANK-}"
echo " NODE_RANK                 : ${NODE_RANK-}"
echo " SLURM_LOCALID             : $SLURM_LOCALID"
echo " LOCAL_RANK                : ${LOCAL_RANK-}" 
echo " SLURM_NTASKS              : $SLURM_NTASKS"
echo " WORLD_SIZE                : ${WORLD_SIZE-}"
echo " CUDA_VISIBLE_DEVICES      : ${CUDA_VISIBLE_DEVICES-}"
echo " HIP_VISIBLE_DEVICES       : ${HIP_VISIBLE_DEVICES-}"
echo " ROCR_VISIBLE_DEVICES      : ${ROCR_VISIBLE_DEVICES-}"
echo " MIOPEN_USER_DB_PATH       : $MIOPEN_USER_DB_PATH"
echo " MIOPEN_CUSTOM_CACHE_DIR   : $MIOPEN_CUSTOM_CACHE_DIR"
echo " LOG_DIR                   : $LOG_DIR"
echo " EXP_ID                    : $EXP_ID"
echo " PPID_ID                   : $PPID"
echo ==============================================
