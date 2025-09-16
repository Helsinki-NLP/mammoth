#!/usr/bin/env -S -u BASH_ENV bash --noprofile --norc
echo running task-wrapper.sh...

# Use -u for Python, not for the bash wrapper.
# In bash, -u means treat unset vars as an error (nounset). It has
# nothing to do with buffering. It’s still a good safety flag for the
echo "task-wrapper.sh..."
set -euo pipefail
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }

# Usage:
#   source $SLURM/local-setup.sh

# --- must be inside an srun-launched task ---
require_vars SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURM_NTASKS
require_vars SYSTEM JOB_PATTERN JOB_LOGS JOB_NAME
require_vars MASTER_ADDR MASTER_PORT MASTER_ARGS
require_vars GPUS_PER_NODE SLURM_NNODES

# ------- step-level safety check for nodes -----------
#
# Make sure that the current srun you’re inside actually has ≥1 node,
# not just that the overall job requested nodes.  SLURM_NNODES
# describes the current job step created by srun (it can be
# different—e.g., you requested 4 nodes for the job but launch an srun
# step on just 1).
if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
  NODES="$SLURM_JOB_NUM_NODES"
else
  # Compare: SLURM_JOB_NUM_NODES / NumNodes= (from scontrol) describe
  # the job allocation.  Job asked for 4 nodes (NODES=4). You run srun
  # --nodes=1 ... → inside the step SLURM_NNODES=1, check passes.
  JOBINFO="$(scontrol show -d job ${SLURM_JOB_ID:?})"
  NODES="$(awk -F'[= ]' '/NumNodes=/{print $2; exit}' <<<"$JOBINFO")"
fi
: "${NODES:?could not determine node count}"
STEP_NODES="${SLURM_NNODES:-$NODES}"

# If the step-level count is zero/empty, you’ve likely launched the
# script outside srun, or mis-specified srun options so the step has
# no resources.
if (( STEP_NODES == 0 )); then
  echo "❌ ERROR: step has 0 nodes. Check your srun --nodes/--ntasks-per-node." >&2
  exit 1  #  The script fails fast with a clear message.
fi

# ---------- GPU visibility (safe defaults) ----------
#
# Don’t set any *VISIBLE_DEVICES before srun.  Never export
# *VISIBLE_DEVICES in the batch script before srun. Outside srun
# there’s no SLURM_LOCALID, and you’ll end up exposing all GPUs to
# every task (the “bad” case you saw in the LUMI AI course) .
#
# Do not do this (!):
#  export HIP_VISIBLE_DEVICES=$SLURM_LOCALID # only correct inside srun
#  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5   # 0,1,2,3,4,5 lines are fragile/incorrect
#  export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5   # 0,1,2,3,4,5 lines are fragile/incorrect
# LUMI has 8 GCDs; plus you shouldn’t hardcode).
#
# Pick the right env var names once
case "$SYSTEM" in
  lumi)           GPU_ENV1=HIP_VISIBLE_DEVICES;  GPU_ENV2=ROCR_VISIBLE_DEVICES ;;
  puhti|mahti|*)  GPU_ENV1=CUDA_VISIBLE_DEVICES; GPU_ENV2= ;;
esac

case "$JOB_PATTERN" in
  slurm)
    # One Slurm task per GPU → bind task to GPU == LOCAL_RANK
    export MASTER_ADDR="${MASTER_ADDR}"        # you already set these
    export MASTER_PORT="${MASTER_PORT}"
    export RANK="${SLURM_PROCID}"
    export WORLD_SIZE="${SLURM_NTASKS}"
    export LOCAL_RANK="${SLURM_LOCALID}"       # per-node rank
    export NODE_RANK="${SLURM_NODEID}"         # some tools/scripts use this

    # Set VISIBILITY inside the srun task, and only when you’re using the Slurm
    # pattern (1 task per GPU). 
    export "$GPU_ENV1"="$SLURM_LOCALID"
    [[ -n "${GPU_ENV2:-}" ]] && export "$GPU_ENV2"="$SLURM_LOCALID"
    ;;
  
  torchrun)
    # For the torchrun pattern, don’t set them at all (or explicitly
    # unset) so all local GPUs are visible to the launcher.
    # torchrun spawns per-GPU workers and sets ranks itself → expose all local GPUs
    [[ -n "${GPU_ENV1:-}" ]] && unset "$GPU_ENV1"
    [[ -n "${GPU_ENV2:-}" ]] && unset "$GPU_ENV2"

    # Optional: many users set this for torchrun
    export NODE_RANK="${SLURM_NODEID:-0}"
    ;;
  *)
    echo "ERROR: JOB_PATTERN must be 'slurm' or 'torchrun' (got '$JOB_PATTERN')." >&2
    exit 1
    ;;
esac

echo ==============================================
echo " JOB_LOGS                  : $JOB_LOGS"
echo " JOB_NAME                  : $JOB_NAME"
echo " PPID                      : $PPID"
echo ==============================================
echo " SYSTEM                    : $SYSTEM"
echo " JOB_PATTERN               : $JOB_PATTERN"
echo ==============================================
echo " NODES                     : $NODES"
echo "   SLURM_JOB_NUM_NODES     : $SLURM_JOB_NUM_NODES"
echo "   SLURM_NNODES            : $SLURM_NNODES"
echo " SLURM_NODEID              : $SLURM_NODEID"
echo "   NODE_RANK               : ${NODE_RANK-}"
echo " RANK                      : ${RANK-}"
echo "   SLURM_PROCID            : $SLURM_PROCID"
echo " LOCAL_RANK                : ${LOCAL_RANK-}" 
echo "   SLURM_LOCALID           : $SLURM_LOCALID"
echo " WORLD_SIZE                : ${WORLD_SIZE-}"
echo "   SLURM_NTASKS            : $SLURM_NTASKS"
echo " CUDA_VISIBLE_DEVICES      : ${CUDA_VISIBLE_DEVICES-}"
echo " HIP_VISIBLE_DEVICES       : ${HIP_VISIBLE_DEVICES-}"
echo " ROCR_VISIBLE_DEVICES      : ${ROCR_VISIBLE_DEVICES-}"
echo ==============================================

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
echo ==============================================
echo " MIOPEN_USER_DB_PATH       : $MIOPEN_USER_DB_PATH"
echo " MIOPEN_CUSTOM_CACHE_DIR   : $MIOPEN_CUSTOM_CACHE_DIR"
echo " OMP_NUM_THREADS           : $OMP_NUM_THREADS"
echo " OMP_PROC_BIND             : $OMP_PROC_BIND"
echo " OMP_PLACES                : $OMP_PLACES"
echo ==============================================

# Optional function: quick affinity print + guardrails (teach-by-error
# for slurm programmers). Calling the following prints what each task
# sees: node, local rank, the visible GPU index, CPU affinity list,
# and NUMA binding. It also raises errors for common footguns.
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
  if [[ "$JOB_PATTERN" == "slurm" ]]; then
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

# Start lightweight GPU monitor by calling the following function:
MON_PID=""
start_monitor() {
  if [[ "$SYSTEM" == "puhti" || "$SYSTEM" == "mahti" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi dmon -s mu -d 5 -o TD > "${JOB_LOGS}/gpu_load-${JOB_NAME}-${PPID}.log" &
      MON_PID=$!
    fi
  else # LUMI / ROCm
    if command -v rocm-smi >/dev/null 2>&1; then
      # sample every 5s: util, power, temp, vram
      while true; do
        rocm-smi --showuse --showtemp --showpower --showmemuse || true
        sleep 5
      done > "${JOB_LOGS}/gpu_load-${JOB_NAME}-${PPID}.log" &
      MON_PID=$!
    fi
  fi
}
stop_monitor() { [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null || true; }


: "${NPROC_PER_NODE:=${GPUS_PER_NODE:-1}}"	# Set NPROC_PER_NODE

print_affinity # Optional: print CPU/GPU binding sanity

trap stop_monitor EXIT  # clean with trap
start_monitor  # Start monitor (comment out if you don’t want it)

case "${PATTERN:-slurm}" in
    slurm)
	RUNTIME="python"
      	RUNTIME_ARGS=("-u")
	
	# POST_ARGS=(--node_rank "${SLURM_NODEID:-0}" ${MASTER_ARGS} )
	POST_ARGS=()
	# PyTorch honors env vars: 
	#    MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, LOCAL_RANK, NODE_RANK
	# These were already exported previously (above)
	;;
    torchrun)
	# python/pytorch uses MASTER_ARGS (already set) but torchrun uses RDZV_ARGS:
	RDZV_ARGS="--master_addr=${MASTER_ADDR}" --master_port=${MASTER_PORT}" # old notation
	RDZV_ARGS="--rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
	RUNTIME="torchrun"
    	RUNTIME_ARGS=(--node_rank "${SLURM_NODEID:-0}" ${RDZV_ARGS} \
		      --nnodes="${SLURM_JOB_NUM_NODES:?}" \
		      --nproc_per_node="${SLURM_GPUS_ON_NODE}")  # alternative 
    	RUNTIME_ARGS=(--node_rank "${SLURM_NODEID:-0}" ${RDZV_ARGS} \
		      --nnodes="${SLURM_NNODES:?}" \
		      --nproc_per_node="${NPROC_PER_NODE}")
        POST_ARGS=()
	;;
esac

echo ==============================================
echo " RUNTIME           : $RUNTIME"
echo " RUNTIME_ARGS      : ${RUNTIME_ARGS[@]}"
echo "   SLURM_NODEID    : $SLURM_NODEID"
echo "   RDZV_ARGS       : ${RDZV_ARGS[@]}"
echo "   SLURM_NNODES    : $SLURM_NNODES"
echo "   NPROC_PER_NODE  : $NPROC_PER_NODE"
echo "     GPUS_PER_NODE : $GPUS_PER_NODE"
echo " COMMAND LINE ARGS : $@"
echo " POST_ARGS         : ${POST_ARGS[@]}"
echo "   SLURM_NODEID    : $SLURM_NODEID"
echo "   MASTER_ARGS     : $MASTER_ARGS"
echo ==============================================
echo =========================================================================================
echo " EXECUTING         : ${RUNTIME} ${RUNTIME_ARGS[@]} "$@" ${POST_ARGS[@]}"
echo =========================================================================================
echo
echo "Starting ${RUNTIME} and the Python script at `date`"
exec ${RUNTIME} ${RUNTIME_ARGS[@]} "$@" ${POST_ARGS[@]}
