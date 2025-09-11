echo Setting up distributed computing environment ...
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
: "${SYSTEM:?❌ SYSTEM is not set}"

# Do not hardcode CUDA_VISIBLE_DEVICES=0,1,2…. Let binding happen per task.

echo Retrospective summary of Slurm env variables
echo ==============================================
echo " SLURM_JOB_NAME            : $SLURM_JOB_NAME"
echo " SLURM_JOB_ID              : $SLURM_JOB_ID"               
echo " SLURM_NTASKS              : $SLURM_NTASKS"
echo " SLURM_NTASKS_PER_NODE     : $SLURM_NTASKS_PER_NODE"
echo " SLURM_CPUS_ON_NODE        : $SLURM_CPUS_ON_NODE"
echo " SLURM_CPUS_PER_TASK       : $SLURM_CPUS_PER_TASK"
echo " SLURM_GPUS_ON_NODE        : $SLURM_GPUS_ON_NODE"
echo " SLURM_GPUS_PER_TASK       : ${SLURM_GPUS_PER_TASK:-}"
echo " SLURM_GRES                : ${SLURM_GRES:-}"
echo " SLURM_HINT                : $SLURM_HINT"
echo " SLURM_JOB_PARTITION       : $SLURM_JOB_PARTITION"
echo " SLURM_JOB_ACCOUNT         : $SLURM_JOB_ACCOUNT"
echo " SLURM_JOB_CPUS_PER_NODE   : $SLURM_JOB_CPUS_PER_NODE"
echo " SLURM_JOB_GPUS            : $SLURM_JOB_GPUS"
echo " SLURM_JOB_NAME            : $SLURM_JOB_NAME"
echo " SLURM_JOB_NUM_NODES       : $SLURM_JOB_NUM_NODES"
echo " SLURM_MEM_PER_GPU         : $SLURM_MEM_PER_GPU"
echo " SLURM_NNODES              : $SLURM_NNODES"
echo " SLURM_NODELIST            : $SLURM_NODELIST"
echo " SLURM_NPROCS              : $SLURM_NPROCS"
echo " SLURM_NTASKS              : $SLURM_NTASKS"
echo " SLURM_NTASKS_PER_NODE     : $SLURM_NTASKS_PER_NODE"
echo " SLURM_TASKS_PER_NODE      : $SLURM_TASKS_PER_NODE"
echo ==============================================
# env | egrep 'SLURM' | sort

if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
  NODES="$SLURM_JOB_NUM_NODES"
else
  JOBINFO="$(scontrol show -d job ${SLURM_JOB_ID:?})"
  NODES="$(awk -F'[= ]' '/NumNodes=/{print $2; exit}' <<<"$JOBINFO")"
fi
: "${NODES:?could not determine node count}"

# --- derive CPUS_PER_TASK and GPUS_PER_NODE from Slurm (if possible) ---------

# 1) Respect explicit Slurm env first
[[ -n "${SLURM_CPUS_PER_TASK-}"   ]] && CPUS_PER_TASK="$SLURM_CPUS_PER_TASK"
[[ -n "${SLURM_GPUS_PER_NODE-}"   ]] && GPUS_PER_NODE="$(grep -oE '^[0-9]+' <<<"$SLURM_GPUS_PER_NODE")"
[[ -n "${SLURM_GPUS_ON_NODE-}"    && -z "${GPUS_PER_NODE-}" ]] && GPUS_PER_NODE="$(grep -oE '^[0-9]+' <<<"$SLURM_GPUS_ON_NODE")"

# 2) If still missing, query the job record
JOBINFO="$(scontrol show -d job "${SLURM_JOB_ID:?}" 2>/dev/null || true)"

if [[ -z "${CPUS_PER_TASK-}" ]]; then
  CPT="$(awk -F'[= ]' '/Cpus\/Task=/{print $2; exit}' <<<"$JOBINFO")"
  [[ -n "$CPT" && "$CPT" != 0 ]] && CPUS_PER_TASK="$CPT"
fi

if [[ -z "${GPUS_PER_NODE-}" ]]; then
  # From GresPerNode, sum tokens gpu[:type]:N → N
  GPN="$(awk -F'[= ]' '/GresPerNode=/{print $2; exit}' <<<"$JOBINFO")"
  if [[ -n "$GPN" ]]; then
    GPUS_PER_NODE=0
    while read -r tok; do
      n=$(awk -F: '{print $NF}' <<<"$tok" | grep -oE '[0-9]+' || true)
      [[ -n "$n" ]] && (( GPUS_PER_NODE += n ))
    done < <(grep -oE 'gpu[^, ]*' <<<"$GPN")
    [[ "$GPUS_PER_NODE" -eq 0 ]] && unset GPUS_PER_NODE
  fi
fi

# 3) If CPUS_PER_TASK still unknown, compute a sensible value from node capacity
if [[ -z "${CPUS_PER_TASK-}" ]]; then
  # tasks-per-node (prefer env; else parse TasksPerNode= from job)
  if [[ -n "${SLURM_NTASKS_PER_NODE-}" ]]; then
    TPN="$SLURM_NTASKS_PER_NODE"
  else
    TPN_RAW="$(awk -F'[= ]' '/TasksPerNode=/{print $2; exit}' <<<"$JOBINFO")"
    # normalize patterns like "4(x2),8" → 4
    TPN="$(awk -F',' '{print $1}' <<<"$TPN_RAW" | sed -E 's/\(x[0-9]+\)//; s/[^0-9].*$//')"
  fi
  # fall back to 1 task/GPU policy if not found
  [[ -z "$TPN" && -n "${GPUS_PER_NODE-}" ]] && TPN="$GPUS_PER_NODE"

  # get node CPU capacity
  FIRST_NODE="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?}" | head -n1)"
  CPUTOT="$(scontrol show node "$FIRST_NODE" 2>/dev/null | awk -F'[= ]' '/CPUTot=/{print $2; exit}')"

  if [[ -n "$CPUTOT" && -n "$TPN" && "$TPN" -gt 0 ]]; then
    CPUS_PER_TASK="$(( CPUTOT / TPN ))"
    (( CPUS_PER_TASK < 1 )) && CPUS_PER_TASK=1
  fi
fi

# 4) You will still set policy defaults per system next (only if unset)
case "$SYSTEM" in
  lumi)
      # On LUMI (AMD/ROCm), Slurm’s gpu-bind may be a no-op;
      # we bind via HIP/ROCR_VISIBLE_DEVICES using SLURM_LOCALID
      # (as in your world-setup).
      
      : "${GPUS_PER_NODE:=8}"     # policy default 8 GCDs per LUMI-G node
      : "${CPUS_PER_TASK:=7}"     # policy default (≈56 cores / 8 GCDs)
      
      # choose either way:
      
      CPU_BIND_OPS="--cpu-bind=cores"  # rely on HIP/ROCR_VISIBLE_DEVICES=$SLURM_LOCALID *inside* srun.
      # Just use --cpu-bind=cores and map tasks→GCDs via HIP/ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
      # inside srun.

      # Optional: if you want the exact topology mapping from the LUMI slides, use
      # the provided CPU mask string instead of --cpu-bind=cores. It’s a fine-tuning, not a requirement.
      #  LUMI_CPU_MASKS=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
      #  CPU_BIND_OPS="--cpu-bind=$LUMI_CPU_MASKS" # the slide’s exact LUMI masks
      # Slide: https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf

      GPU_BIND_FLAG=""            # do NOT use --gpu-bind on ROCm/LUMI CPU masks?
      # Put them under a *CPU* bind var (see note below)
      # --gpu-bind is an NVIDIA feature; on LUMI you want CPU binding, not GPU binding.
      
      # Never export *VISIBLE_DEVICES in the batch script before srun.
      # Outside srun there’s no SLURM_LOCALID, you’ll end up exposing all GPUs to every task
      ;;
  
  puhti)
      # --gpu-bind=closest is great on NVIDIA (Puhti/Mahti).
      : "${GPUS_PER_NODE:=4}"
      : "${CPUS_PER_TASK:=10}"            # 40 cores / 4 GPUs
      CPU_BIND_OPS=""
      GPU_BIND_FLAG="--gpu-bind=closest"  # NVIDIA-only helper (Puhti/Mahti): bind task to the PCIe-closest GPU
      ;;
  
  mahti)
      : "${GPUS_PER_NODE:=4}"
      : "${CPUS_PER_TASK:=8}"             # start with 8; can go higher after profiling
      CPU_BIND_OPS=""
      GPU_BIND_FLAG="--gpu-bind=closest"  # NVIDIA-only helper (Puhti/Mahti): bind task to the PCIe-closest GPU 
      ;;
esac
export GPUS_PER_NODE
export CPUS_PER_TASK

# On NVIDIA (Puhti/Mahti), pass --gpu-bind=closest to srun so the task is placed near “its” GPU
# On LUMI (ROCm), rely on HIP/ROCR_VISIBLE_DEVICES=$SLURM_LOCALID + --cpu-bind=cores

export DISTR_OPS="--unbuffered --nodes=$NODES --ntasks-per-node=$GPUS_PER_NODE\
 --gpus-per-task=1 --cpus-per-task=$CPUS_PER_TASK $CPU_BIND_OPS $GPU_BIND_FLAG"

######################################################
# ---------- torchrun rendezvous (MASTER_*) ----------
######################################################
# These are safe to set once per job.
# Use first hostname from nodelist, avoid plain `hostname` (not always in containers)
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
# Pick a port that collides less: 29500 + (JOBID mod 1000)
export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"

echo ==============================================
echo " SYSTEM              : $SYSTEM"
echo " SLURM_JOB_NUM_NODES : $SLURM_JOB_NUM_NODES"
echo " local NODES         : $NODES"
echo " GPUS_PER_NODE       : $GPUS_PER_NODE"
echo " CPUS_PER_TASK       : $CPUS_PER_TASK"
echo " local CPU_BIND_OPS  : $CPU_BIND_OPS"
echo " local GPU_BIND_FLAG : $GPU_BIND_FLAG"
echo " DISTR_OPS           : $DISTR_OPS"
echo " MASTER_PORT         : $MASTER_PORT"
echo " MASTER_ADDR         : $MASTER_ADDR"
echo ==============================================

