# shellcheck shell=sh
echo sanity-checks...

# ---- sourced vs executed -----------------------------------------------------
# Detect whether this file is sourced (portable)
_SOURCED=0
( return 0 2>/dev/null ) && _SOURCED=1 || _SOURCED=0
die() { if [ "${_SOURCED:-0}" -eq 1 ]; then return 1; else exit 1; fi; }

# ---- tiny utils (POSIX) ------------------------------------------------------
num_last() {
  # last integer in a string (e.g. "gpu:mi250:1" -> 1)
  printf '%s\n' "${1-}" | awk '{ s=$0;n="";
    while (match(s,/[0-9]+/)) { n=substr(s,RSTART,RLENGTH); s=substr(s,RSTART+RLENGTH) }
    if (n!="") print n }'
}
strip_parens_all() { printf '%s\n' "${1-}" | sed 's/([^)]*)//g'; }
field_from() {  # first NAME=VALUE in a single-line blob
  blob=$1; name=$2
  printf '%s\n' "$blob" | tr ' ' '\n' | awk -F= -v k="$name" '$1==k {print $2; exit}'
}
sum_gpus_from_list() { # sum N from tokens "gpu:mi250:N,gpu:mi250:N"
  list=${1-}; tot=0; oldIFS=$IFS; IFS=,; set -- $list; IFS=$oldIFS
  for t do case $t in gres:gpu:*|gpu:*) n=$(num_last "$t"); [ -n "$n" ] && tot=$((tot+n));; esac; done
  printf '%s' "$tot"
}
slurm_time_to_seconds() {
  t=$1; [ -n "$t" ] || return 1
  awk -v T="$t" 'BEGIN{
    d=0;h=0;m=0;s=0;n=split(T,A,"-"); if(n==2){d=A[1]; T=A[2]} else {T=A[1]}
    n=split(T,B,":");
    if(n==3){h=B[1];m=B[2];s=B[3]}
    else if(n==2){h=0;m=B[1];s=B[2]}
    else if(n==1){h=B[1];m=0;s=0}
    if(h=="")h=0;if(m=="")m=0;if(s=="")s=0;if(d=="")d=0
    print d*86400 + h*3600 + m*60 + s }'
}
mb_to_gb() { mb=${1:-0}; echo $(( (mb + 1023) / 1024 )); }

# -----------------------------------------------------------------------------
#                                PHASE 1: COLLECT
# -----------------------------------------------------------------------------

# Normalize inputs (don’t fail yet)
SYSTEM=$(printf '%s' "${SYSTEM-}" | tr '[:upper:]' '[:lower:]')
JOB_PATTERN=${JOB_PATTERN:-slurm}   # slurm|torchrun
GUARD_MAX_NODES=${GUARD_MAX_NODES:-4}
GUARD_TIME=${GUARD_TIME:-0-01:00:00}

# Pull job/step records (single-line blobs)
JOBREC=$(scontrol show -d job  "${SLURM_JOB_ID:-}" 2>/dev/null | tr '\n' ' ')
STEPREC=$(scontrol show -d step "${SLURM_JOB_ID:-}.batch" 2>/dev/null | tr '\n' ' ')
REC_ALL="$JOBREC $STEPREC"

# Primary environment values (may be empty)
SLURM_JOB_ID=${SLURM_JOB_ID-}
SLURM_JOB_ACCOUNT=${SLURM_JOB_ACCOUNT-}
SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION-}
SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES-}
SLURM_NTASKS=${SLURM_NTASKS-}
SLURM_NTASKS_PER_NODE=${SLURM_NTASKS_PER_NODE-}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK-}
SLURM_MEM_PER_GPU=${SLURM_MEM_PER_GPU-}
SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU-}
SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE-}
SLURM_GPUS=${SLURM_GPUS-}
SLURM_GPUS_PER_TASK=${SLURM_GPUS_PER_TASK-}
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE-}

# Derived: time limit
TL_RAW=$(squeue -h -j "${SLURM_JOB_ID:-}" -o %l 2>/dev/null)
TL_SEC=
[ -n "$TL_RAW" ] && TL_SEC=$(slurm_time_to_seconds "$TL_RAW" 2>/dev/null || echo "" )
GT_SEC=$(slurm_time_to_seconds "$GUARD_TIME" 2>/dev/null || echo "")

# Derived: stdout/stderr paths
STDOUT_PATH=$(scontrol show job "${SLURM_JOB_ID:-}" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -n1)
STDERR_PATH=$(scontrol show job "${SLURM_JOB_ID:-}" | sed -n 's/.*StdErr=\([^ ]*\).*/\1/p' | head -n1)

# Derived: nodes & tasks
NODES=$SLURM_JOB_NUM_NODES
case "$NODES" in ''|*[!0-9]*) NODES=$(field_from "$REC_ALL" NumNodes);; esac
NTASKS=$SLURM_NTASKS
TPN_ENV=$SLURM_NTASKS_PER_NODE
if [ -n "$TPN_ENV" ]; then
  TPN=$(printf '%s\n' "$TPN_ENV" | tr ',' '\n' | sed -n 's/^\([0-9][0-9]*\).*/\1/p' | head -n1)
else
  if [ -n "$NODES" ] && [ "$NODES" -gt 0 ] 2>/dev/null; then TPN=$(( ${NTASKS:-0} / NODES )); else TPN=0; fi
fi

# Derived: GPU totals and per-node and GRES string
GPUS_TOTAL=
GPUS_PER_NODE=
GRES_STR=

# Totals from env
if [ -n "$SLURM_GPUS" ]; then
  v=$(num_last "$SLURM_GPUS"); [ -n "$v" ] && GPUS_TOTAL=$v
elif [ -n "$SLURM_GPUS_PER_TASK" ] && [ -n "$SLURM_NTASKS" ]; then
  a=$(num_last "$SLURM_GPUS_PER_TASK"); b=$(num_last "$SLURM_NTASKS")
  [ -n "$a" ] && [ -n "$b" ] && GPUS_TOTAL=$(( a * b ))
fi
# Totals from TRES
if [ -z "$GPUS_TOTAL" ]; then
  tot=$(printf '%s\n' "$REC_ALL" \
    | sed -n 's/.*\(ReqTRES\|AllocTRES\)=\([^ ]*\).*/\2/p' \
    | tr ',' '\n' | sed -n 's/.*gres\/gpu[^=]*=\([0-9][0-9]*\).*/\1/p' | head -n1)
  [ -n "$tot" ] && GPUS_TOTAL=$tot
fi

# Per-node from TresPerNode
GPN_RAW=$(field_from "$REC_ALL" TresPerNode)
if [ -n "$GPN_RAW" ]; then
  GRES_STR=$(strip_parens_all "$GPN_RAW")
  totpn=$(sum_gpus_from_list "$GRES_STR")
  [ -n "$totpn" ] && [ "$totpn" -gt 0 ] 2>/dev/null && GPUS_PER_NODE=$totpn
fi
# Fallback per-node from Gres/JOB_GRES
if [ -z "$GPUS_PER_NODE" ]; then
  GRES_RAW=$(field_from "$REC_ALL" Gres)
  [ -z "$GRES_RAW" ] && GRES_RAW=$(field_from "$REC_ALL" JOB_GRES)
  if [ -n "$GRES_RAW" ]; then
    GRES_STR=$(strip_parens_all "$GRES_RAW")
    totpn=$(sum_gpus_from_list "$GRES_STR")
    [ -n "$totpn" ] && [ "$totpn" -gt 0 ] 2>/dev/null && GPUS_PER_NODE=$totpn
  fi
fi
# Derive total from per-node * nodes
if [ -z "$GPUS_TOTAL" ] && [ -n "$GPUS_PER_NODE" ] && [ -n "$NODES" ] && [ "$NODES" -gt 0 ] 2>/dev/null; then
  GPUS_TOTAL=$(( GPUS_PER_NODE * NODES ))
fi

# Per-system static knobs (for later checks)
case "$SYSTEM" in
  puhti)
    GPUS_PER_NODE_MAX=4
    GPU_OK_RX='v100|tesla_v100'
    REC_CPUS_PER_GPU_MIN=10; REC_CPUS_PER_GPU_MAX=10
    REC_MEM_PER_GPU_GB=95
    NODE_RAM_GB=384
    ;;
  mahti)
    GPUS_PER_NODE_MAX=4
    GPU_OK_RX='a100|a100_1g\.5gb|a100_2g\.10gb|a100_3g\.20gb|a100_7g\.80gb'
    REC_CPUS_PER_GPU_MIN=8;  REC_CPUS_PER_GPU_MAX=32
    REC_MEM_PER_GPU_GB=128
    NODE_RAM_GB=512
    ;;
  lumi)
    GPUS_PER_NODE_MAX=8
    GPU_OK_RX='mi250x?|mi200|(^|,)gpu(:|,|$)'
    REC_CPUS_PER_GPU_MIN=6;  REC_CPUS_PER_GPU_MAX=8
    REC_MEM_PER_GPU_GB=60
    NODE_RAM_GB=512
    ;;
  *) : ;;
esac

# -----------------------------------------------------------------------------
#                                PHASE 2: PRINT
# -----------------------------------------------------------------------------
echo "====================================================================="
echo " SYSTEM                    : ${SYSTEM:-<unset>}"
echo " SLURM_JOB_ID              : ${SLURM_JOB_ID:-<unset>}"
echo " SLURM_JOB_ACCOUNT         : ${SLURM_JOB_ACCOUNT:-<unset>}"
echo " SLURM_JOB_PARTITION       : ${SLURM_JOB_PARTITION:-<unset>}"
echo " TIMELIMIT                 : ${TL_RAW:-<unset>}"
echo " GUARD_TIME                : ${GUARD_TIME:-<unset>}"
echo " GUARD_MAX_NODES           : ${GUARD_MAX_NODES:-<unset>}"
echo " JOB_PATTERN               : ${JOB_PATTERN:-<unset>}"
echo " STDOUT                    : ${STDOUT_PATH:-<unset>}"
echo " STDERR                    : ${STDERR_PATH:-<unset>}"
echo " SLURM_NTASKS              : ${NTASKS:-<unset>}"
echo " SLURM_NTASKS_PER_NODE     : ${SLURM_NTASKS_PER_NODE:-<unset>}"
echo " SLURM_CPUS_PER_TASK       : ${SLURM_CPUS_PER_TASK:-<unset>}"
echo " SLURM_MEM_PER_GPU         : ${SLURM_MEM_PER_GPU:-<unset>}"
echo " SLURM_MEM_PER_CPU         : ${SLURM_MEM_PER_CPU:-<unset>}"
echo " SLURM_MEM_PER_NODE        : ${SLURM_MEM_PER_NODE:-<unset>}"
echo " SLURM_GPUS                : ${SLURM_GPUS:-<unset>}"
echo " SLURM_GPUS_PER_TASK       : ${SLURM_GPUS_PER_TASK:-<unset>}"
echo " SLURM_GPUS_PER_NODE       : ${SLURM_GPUS_PER_NODE:-<unset>}"
echo " local NODES               : ${NODES:-<unset>}"
echo " local TASKS_PER_NODE (TPN): ${TPN:-<unset>}"
echo " local GPUS_TOTAL          : ${GPUS_TOTAL:-<unset>}"
echo " local GPUS_PER_NODE       : ${GPUS_PER_NODE:-<unset>}"
echo " local GPUS_PER_NODE_MAX   : ${GPUS_PER_NODE_MAX:-<unset>}"
echo " local GRES_STR            : ${GRES_STR:-<unset>}"
echo " local GPU_OK_REGEX        : ${GPU_OK_RX:-<unset>}"
echo " local REC_CPUS_PER_GPU_MIN: ${REC_CPUS_PER_GPU_MIN:-<unset>}"
echo " local REC_MEM_PER_GPU_GB  : ${REC_MEM_PER_GPU_GB:-<unset>}"
echo " local NODE_RAM_GB         : ${NODE_RAM_GB:-<unset>}"
echo "====================================================================="

# -----------------------------------------------------------------------------
#                                PHASE 3: CHECK
# -----------------------------------------------------------------------------

# Required baseline (after printing → friendlier debugging)
missing=0
for v in SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION; do
  eval "vv=\${$v-}"; if [ -z "$vv" ]; then printf '❌ %s must be set\n' "$v" >&2; missing=1; fi
done
[ "$missing" -eq 0 ] || die

# GPU presence (env or job record)
GPU_REQ_SEEN=0
[ -n "${SLURM_GPUS-}" ]            && GPU_REQ_SEEN=1
[ -n "${SLURM_GPUS_PER_TASK-}" ]   && GPU_REQ_SEEN=1
[ -n "${SLURM_GPUS_PER_NODE-}" ]   && GPU_REQ_SEEN=1
echo "$REC_ALL" | grep -E -q '(^|[[:space:]])(Gres=.*gpu|JOB_GRES=.*gpu|ReqTRES=.*gres/gpu|AllocTRES=.*gres/gpu|TresPerNode=.*gpu)' && GPU_REQ_SEEN=1
if [ "$GPU_REQ_SEEN" -eq 0 ]; then
  echo "❌ No GPUs requested. Use either --gpus-* OR --gres=gpu[:type]:N." >&2; die
fi

# Nodes / time guards
case "$NODES" in ''|*[!0-9]*|0) echo "❌ could not determine node count"; die;; esac
if [ "$NODES" -gt "$GUARD_MAX_NODES" ]; then
  echo "❌ ERROR: requested $NODES nodes (> $GUARD_MAX_NODES). Reduce nodes or increase GUARD_MAX_NODES." >&2; die
fi
case "$TL_RAW" in ''|NOT_SET|UNLIMITED|N/A) echo "❌ No valid time limit set for the job. Use --time=<D-HH:MM:SS>." >&2; die;; esac
if [ -z "$TL_SEC" ] || [ -z "$GT_SEC" ]; then
  echo "❌ Could not parse time values (TL='$TL_RAW', GUARD_TIME='$GUARD_TIME')." >&2; die
fi
if [ "$TL_SEC" -gt "$GT_SEC" ]; then
  echo "❌ Time limit $TL_RAW exceeds guard ${GUARD_TIME}. Reduce --time." >&2; die
fi
echo "====================================================================="
printf 'Time Allocation: requested=%s (=%ss) ≤ guard=%s (=%ss)\n' "$TL_RAW" "$TL_SEC" "$GUARD_TIME" "$GT_SEC"
echo "====================================================================="

##############################################################################
#                               PHASE 3: CHECK
##############################################################################

# --- helpers used here only --------------------------------------------------
to_int_or_zero() { case "$1" in ''|*[!0-9]*) echo 0;; *) echo "$1";; esac; }
mb_to_gb() { mb=${1:-0}; echo $(( (mb + 1023) / 1024 )); }

# normalize a few numbers we’ll compare
CPUS_PER_TASK_NUM=$(to_int_or_zero "${SLURM_CPUS_PER_TASK-}")
GPUS_PER_TASK_NUM=$(to_int_or_zero "${SLURM_GPUS_PER_TASK-}")

# 0) minimal baseline
missing=0
for v in SLURM_JOB_ID SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION; do
  eval "vv=\${$v-}"; if [ -z "$vv" ]; then printf '❌ %s must be set\n' "$v" >&2; missing=1; fi
done
[ "$missing" -eq 0 ] || { echo "✖ Baseline failed."; die; }
echo "OK[baseline]: core Slurm env present (job/account/partition). Good."

# 1) GPUs requested at all?
GPU_REQ_SEEN=0
[ -n "${SLURM_GPUS-}" ]          && GPU_REQ_SEEN=1
[ -n "${SLURM_GPUS_PER_TASK-}" ] && GPU_REQ_SEEN=1
[ -n "${SLURM_GPUS_PER_NODE-}" ] && GPU_REQ_SEEN=1
echo "$REC_ALL" | grep -E -q '(^|[[:space:]])(Gres=.*gpu|JOB_GRES=.*gpu|ReqTRES=.*gres/gpu|AllocTRES=.*gres/gpu|TresPerNode=.*gpu)' && GPU_REQ_SEEN=1

if [ "$GPU_REQ_SEEN" -eq 0 ] || [ -z "${GPUS_TOTAL-}" ]; then
  echo "❌ No GPUs requested. Use either --gpus-* OR --gres=gpu[:type]:N." >&2; die
fi
echo "OK[gpu-request]: GPUs requested (total=${GPUS_TOTAL}${GPUS_PER_NODE:+, per-node=$GPUS_PER_NODE})."

# 2) nodes/time guards
case "$NODES" in ''|*[!0-9]*|0) echo "❌ could not determine node count"; die;; esac
if [ "$NODES" -le "$GUARD_MAX_NODES" ]; then
  echo "OK[nodes]: nodes=$NODES within guard ≤ $GUARD_MAX_NODES. Nice."
else
  echo "❌ nodes=$NODES exceeds guard ($GUARD_MAX_NODES)."; die
fi

case "$TL_RAW" in ''|NOT_SET|UNLIMITED|N/A) echo "❌ No valid time limit set (use --time=…)."; die;; esac
if [ -n "$TL_SEC" ] && [ -n "$GT_SEC" ] && [ "$TL_SEC" -le "$GT_SEC" ]; then
  echo "OK[time]: timelimit $TL_RAW is within guard $GUARD_TIME. Good."
else
  echo "❌ Time limit $TL_RAW exceeds guard $GUARD_TIME."; die
fi

# 3) stdout/stderr dirs writable
ok_logs=1
for f in "$STDOUT_PATH" "$STDERR_PATH"; do
  [ -n "$f" ] || { ok_logs=0; break; }
  [ "$f" = "/dev/null" ] && continue
  d=${f%/*}; [ "x$d" = "x$f" ] && d=${SLURM_SUBMIT_DIR:-$PWD}
  [ -d "$d" ] && [ -w "$d" ] || { ok_logs=0; break; }
done
if [ "$ok_logs" -eq 1 ]; then
  echo "OK[logs]: Stdout/Stderr directories exist and are writable."
else
  echo "❌ Missing/unwritable StdOut/StdErr directories."; die
fi

# 4) task layout consistency (NODES × TPN == NTASKS)
case "${NTASKS:-}" in ''|*[!0-9]*|0) echo "❌ SLURM_NTASKS is 0. Set --ntasks or --ntasks-per-node."; die;; esac
if [ $(( TPN * NODES )) -eq "$NTASKS" ]; then
  echo "OK[layout]: $NODES×$TPN tasks-per-node = $NTASKS total tasks. Perfect."
else
  echo "❌ Inconsistent tasks: ntasks=$NTASKS but nodes=$NODES and ntasks-per-node=$TPN." >&2
  echo "   Fix: --nodes=X --ntasks-per-node=Y  OR  --ntasks=Z where Z=X*Y." >&2
  die
fi

# 5) per-system GPU policy & type
case "$SYSTEM" in
  puhti|mahti|lumi) : ;;
  *) echo "❌ SYSTEM must be {puhti,mahti,lumi}, got '${SYSTEM:-<unset>}'"; die;;
esac

# typed kind (if visible)
if [ -n "${GRES_STR-}" ]; then
  if printf '%s\n' "$GRES_STR" | grep -Eq -- "${GPU_OK_RX:-.}"; then
    echo "OK[gpu-kind]: typed GPU matches $SYSTEM policy ($GRES_STR). Good."
  else
    echo "❌ Wrong/missing GPU type for $SYSTEM. Got: '$GRES_STR'." >&2
    case "$SYSTEM" in
      puhti) echo "   Use: --gres=gpu:v100:<N>." >&2 ;;
      mahti) echo "   Use: --gres=gpu:a100:<N> (or a100_1g.5gb on gpusmall)." >&2 ;;
      lumi)  echo "   Use: --gres=gpu:mi250:<K> or plain --gpus-per-node=<K>." >&2 ;;
    esac
    die
  fi
else
  # No typed GRES visible
  if [ "$SYSTEM" = "lumi" ] && [ -n "${GPUS_PER_NODE-}" ] && [ "${GPUS_PER_NODE:-0}" -gt 0 ] 2>/dev/null; then
    echo "OK[gpu-kind]: LUMI is homogeneous; untyped --gpus-per-node=${GPUS_PER_NODE} is fine. Keep it."
  else
    echo "❌ GRES not visible. On $SYSTEM, request typed GPUs via --gres=gpu:<type>:<N>."; die
  fi
fi

# cap per node
if [ -n "${GPUS_PER_NODE-}" ] && [ "${GPUS_PER_NODE:-0}" -le "${GPUS_PER_NODE_MAX:-999}" ] 2>/dev/null; then
  echo "OK[gpu-per-node]: ${GPUS_PER_NODE:-?} ≤ max ${GPUS_PER_NODE_MAX} on $SYSTEM."
else
  echo "❌ Requested ${GPUS_PER_NODE:-?} GPUs per node (> ${GPUS_PER_NODE_MAX}) on $SYSTEM."; die
fi

# 6) partition guidance (positive where applicable)
PART=${SLURM_JOB_PARTITION:-unknown}
case "$SYSTEM" in
  lumi)
    if [ "$NODES" -le 4 ] 2>/dev/null && { [ "$PART" = "small-g" ] || [ "$PART" = "dev-g" ]; }; then
      echo "OK[partition]: $PART fits ≤4-node use nicely on LUMI."
    elif [ "$PART" = "standard-g" ] && [ "${GPUS_PER_NODE:-0}" -eq 8 ] 2>/dev/null; then
      echo "OK[partition]: standard-g with 8 GPUs per node requests whole nodes. Good."
    fi
    ;;
  puhti)
    if [ "$GPUS_TOTAL" -gt 0 ] 2>/dev/null && { [ "$PART" = "gpu" ] || [ "$PART" = "gputest" ]; }; then
      echo "OK[partition]: $PART is appropriate for GPU jobs on Puhti."
    fi
    ;;
  mahti)
    if [ "$GPUS_TOTAL" -le 2 ] 2>/dev/null && [ "$PART" = "gpusmall" ]; then
      echo "OK[partition]: gpusmall matches ≤2 GPU jobs on Mahti."
    elif [ "$GPUS_TOTAL" -gt 2 ] 2>/dev/null && [ "$PART" = "gpumedium" ]; then
      echo "OK[partition]: gpumedium matches >2 GPU jobs on Mahti."
    fi
    ;;
esac

# 7) launch pattern specifics
if [ "$CPUS_PER_TASK_NUM" -le 0 ] 2>/dev/null; then
  echo "❌ SLURM_CPUS_PER_TASK is 0. Set --cpus-per-task."; die
fi

if [ "${JOB_PATTERN:-slurm}" = "torchrun" ]; then
  if [ "$TPN" -eq 1 ] 2>/dev/null; then
    echo "OK[torchrun-layout]: ntasks-per-node=1; torchrun will fan out per GPU. Good."
  else
    echo "❌ torchrun requires --ntasks-per-node=1 (got $TPN)."; die
  fi
  if [ -n "${NPROC_PER_NODE-}" ] && [ "${NPROC_PER_NODE:-0}" -eq "${GPUS_PER_NODE:-0}" ] 2>/dev/null; then
    echo "OK[torchrun-nproc]: NPROC_PER_NODE=${NPROC_PER_NODE} matches GPUs per node (${GPUS_PER_NODE})."
  else
    echo "❌ torchrun NPROC_PER_NODE must equal GPUs per node (${GPUS_PER_NODE:-0})."; die
  fi
else
  # classic: one task per GPU
  if [ -n "${GPUS_PER_NODE-}" ] && [ "$GPUS_PER_NODE" -gt 0 ] 2>/dev/null && [ "$TPN" -eq "$GPUS_PER_NODE" ] 2>/dev/null; then
    echo "OK[slurm-layout]: one task per GPU (tpn=$TPN, gpn=$GPUS_PER_NODE). Exactly right."
  else
    echo "❌ Use one task per GPU: set --ntasks-per-node=${GPUS_PER_NODE:-?}."; die
  fi
  if [ "$GPUS_PER_TASK_NUM" -eq 0 ] || [ "$GPUS_PER_TASK_NUM" -eq 1 ] 2>/dev/null; then
    echo "OK[slurm-gpt]: --gpus-per-task=${GPUS_PER_TASK_NUM:-0} is acceptable for one-task-per-GPU."
  else
    echo "❌ For one-task-per-GPU, use --gpus-per-task=1 (got $GPUS_PER_TASK_NUM)."; die
  fi
fi

# 8) memory policy
mix=0
[ -n "${SLURM_MEM_PER_GPU-}"  ] && mix=$((mix+1))
[ -n "${SLURM_MEM_PER_CPU-}"  ] && mix=$((mix+1))
[ -n "${SLURM_MEM_PER_NODE-}" ] && mix=$((mix+1))
if [ "$mix" -gt 1 ]; then
  echo "❌ Do not mix --mem/--mem-per-cpu/--mem-per-gpu. Pick ONE." >&2; die
fi

if [ "${GPUS_TOTAL:-0}" -gt 0 ] 2>/dev/null; then
  if [ -z "${SLURM_MEM_PER_GPU-}" ]; then
    echo "❌ GPU job without --mem-per-gpu. Suggest ~${REC_MEM_PER_GPU_GB}G per GPU."; die
  fi
  mem_per_gpu_gb=$(mb_to_gb "$SLURM_MEM_PER_GPU")
  if [ "$mem_per_gpu_gb" -le "$REC_MEM_PER_GPU_GB" ] 2>/dev/null; then
    echo "OK[mem-gpu]: --mem-per-gpu=${mem_per_gpu_gb}G ≤ ${REC_MEM_PER_GPU_GB}G recommended on $SYSTEM."
  else
    echo "❌ --mem-per-gpu=${mem_per_gpu_gb}G too high (> ${REC_MEM_PER_GPU_GB}G) on $SYSTEM."; die
  fi
  if [ -n "${GPUS_PER_NODE-}" ] && [ "$GPUS_PER_NODE" -gt 0 ] 2>/dev/null; then
    req_node_gb=$(( mem_per_gpu_gb * GPUS_PER_NODE ))
    if [ "$req_node_gb" -le "$NODE_RAM_GB" ] 2>/dev/null; then
      echo "OK[mem-node]: per-node memory ${req_node_gb}G within node RAM ${NODE_RAM_GB}G."
    else
      echo "❌ Per-node memory ${req_node_gb}G exceeds node RAM ${NODE_RAM_GB}G."; die
    fi
  fi
else
  # CPU-only jobs
  if [ -n "${SLURM_MEM_PER_GPU-}" ]; then
    echo "❌ CPU job should not set --mem-per-gpu. Use --mem-per-cpu or --mem."; die
  fi
  if [ -n "${SLURM_MEM_PER_CPU-}" ]; then
    mem_cpu_gb=$(mb_to_gb "$SLURM_MEM_PER_CPU")
    cpus_node=$(( CPUS_PER_TASK_NUM * TPN ))
    req_node_gb=$(( mem_cpu_gb * cpus_node ))
    if [ "$req_node_gb" -le "$NODE_RAM_GB" ] 2>/dev/null; then
      echo "OK[mem-cpu]: per-node ${req_node_gb}G within node RAM ${NODE_RAM_GB}G."
    else
      echo "❌ CPU memory ${req_node_gb}G exceeds node RAM ${NODE_RAM_GB}G."; die
    fi
  elif [ -n "${SLURM_MEM_PER_NODE-}" ]; then
    mem_node_gb=$(mb_to_gb "$SLURM_MEM_PER_NODE")
    if [ "$mem_node_gb" -le "$NODE_RAM_GB" ] 2>/dev/null; then
      echo "OK[mem-node]: --mem=${mem_node_gb}G within node RAM ${NODE_RAM_GB}G."
    else
      echo "❌ --mem=${mem_node_gb}G exceeds node RAM ${NODE_RAM_GB}G."; die
    fi
  else
    echo "❌ CPU job without memory request. Use --mem-per-cpu=<GB> or --mem=<GB>."; die
  fi
fi

# 9) CPU per GPU guidance
if [ -n "${GPUS_PER_NODE-}" ] && [ "$GPUS_PER_NODE" -gt 0 ] 2>/dev/null; then
  if [ "$CPUS_PER_TASK_NUM" -ge "${REC_CPUS_PER_GPU_MIN:-0}" ] 2>/dev/null \
     && [ "$CPUS_PER_TASK_NUM" -le "${REC_CPUS_PER_GPU_MAX:-999}" ] 2>/dev/null
  then
    echo "OK[cpu-per-gpu]: cpus-per-task=$CPUS_PER_TASK_NUM is within recommended range on $SYSTEM."
  else
    echo "❌ cpus-per-task=$CPUS_PER_TASK_NUM is not recommended on $SYSTEM."; die
  fi
fi

export SANITY_CHECKS_OK=1
echo "✅ All sanity checks passed. Keep this setup — it follows best practices."

# -----------------------------------------------------------------------------
#                            ADVICE: tuning & etiquette
# -----------------------------------------------------------------------------
echo "====================================================================="

to_int_or_zero() { case "$1" in ''|*[!0-9]*) echo 0;; *) echo "$1";; esac; }
mb_to_gb() { mb=${1:-0}; echo $(( (mb + 1023) / 1024 )); }

# CPU capacity per node (usable cores) for friendly math
case "$SYSTEM" in
  lumi)   CPUS_NODE_TOTAL=64; CPUS_NODE_USABLE=56 ;;   # leave ~8 for OS/stack
  mahti)  CPUS_NODE_TOTAL=64; CPUS_NODE_USABLE=64 ;;
  puhti)  CPUS_NODE_TOTAL=40; CPUS_NODE_USABLE=40 ;;
  *)      CPUS_NODE_TOTAL=40; CPUS_NODE_USABLE=40 ;;
esac

cpus_per_task_num=$(to_int_or_zero "${SLURM_CPUS_PER_TASK-}")
tpn_num=$(to_int_or_zero "${TPN-0}")
gpn_num=$(to_int_or_zero "${GPUS_PER_NODE-0}")
gpus_total_num=$(to_int_or_zero "${GPUS_TOTAL-0}")
mem_per_gpu_mb="${SLURM_MEM_PER_GPU-}"
mem_per_gpu_gb=""
[ -n "$mem_per_gpu_mb" ] && mem_per_gpu_gb=$(mb_to_gb "$mem_per_gpu_mb")

# Per-node totals actually requested
cpus_node_req=$(( cpus_per_task_num * tpn_num ))
mem_node_req=""
[ -n "$mem_per_gpu_gb" ] && [ "$gpn_num" -gt 0 ] 2>/dev/null && mem_node_req=$(( mem_per_gpu_gb * gpn_num ))
mem_headroom=""
[ -n "$mem_node_req" ] && mem_headroom=$(( NODE_RAM_GB - mem_node_req ))

# 1) CPU per GPU advice
if [ "$gpn_num" -gt 0 ] 2>/dev/null; then
  # Positive reinforcement inside recommended band
  if [ "$cpus_per_task_num" -ge "${REC_CPUS_PER_GPU_MIN:-0}" ] 2>/dev/null \
     && [ "$cpus_per_task_num" -le "${REC_CPUS_PER_GPU_MAX:-999}" ] 2>/dev/null; then
    echo "ADVICE[cpu]: 👍 cpus-per-task=$cpus_per_task_num per GPU is within the recommended range on $SYSTEM."
  fi

  # If below min → suggest raising
  if [ "$cpus_per_task_num" -lt "${REC_CPUS_PER_GPU_MIN:-0}" ] 2>/dev/null; then
    echo "ADVICE[cpu]: Consider increasing cpus-per-task to ~${REC_CPUS_PER_GPU_MIN}-${REC_CPUS_PER_GPU_MAX}/GPU for better throughput on $SYSTEM."
  fi

  # If above max → suggest trimming for friendliness
  if [ "$cpus_per_task_num" -gt "${REC_CPUS_PER_GPU_MAX:-999}" ] 2>/dev/null; then
    echo "ADVICE[cpu]: You’re over-reserving CPU per GPU; trimming toward ${REC_CPUS_PER_GPU_MAX} helps others share the node without hurting you."
  fi

  # Node-level CPU etiquette
  if [ "$cpus_node_req" -gt 0 ] 2>/dev/null; then
    # Fraction of usable cores taken
    # (integer percent; keep it coarse)
    cpu_pct=$(( 100 * cpus_node_req / CPUS_NODE_USABLE ))
    if [ "$cpu_pct" -le 50 ] 2>/dev/null; then
      echo "ADVICE[cpu]: You’re using ~${cpu_pct}% of usable CPU on the node; if training is CPU-bound, consider +1 core/GPU."
    elif [ "$cpu_pct" -ge 90 ] 2>/dev/null && [ "$gpn_num" -lt "${GPUS_PER_NODE_MAX:-999}" ] 2>/dev/null; then
      echo "ADVICE[cpu]: You occupy ~${cpu_pct}% of CPU with only $gpn_num GPUs; be mindful others may struggle to co-locate. If you need stability, consider reserving more GPUs or a full node."
    fi
  fi
fi

# 2) Memory per GPU advice
if [ -n "$mem_per_gpu_gb" ]; then
  # Positive: within recommended ceilings
  if [ "$mem_per_gpu_gb" -le "${REC_MEM_PER_GPU_GB:-999}" ] 2>/dev/null; then
    echo "ADVICE[mem]: 👍 --mem-per-gpu=${mem_per_gpu_gb}G respects the recommended ceiling (${REC_MEM_PER_GPU_GB}G) on $SYSTEM."
  fi

  # Too high vs recommendation
  if [ "$mem_per_gpu_gb" -gt "${REC_MEM_PER_GPU_GB:-999}" ] 2>/dev/null; then
    echo "ADVICE[mem]: --mem-per-gpu=${mem_per_gpu_gb}G is quite high on $SYSTEM; lowering closer to ${REC_MEM_PER_GPU_GB}G/GPU improves node sharing and queue fairness."
  fi

  # Node-level headroom etiquette
  if [ -n "$mem_node_req" ]; then
    # Headroom percent (integer)
    if [ "$NODE_RAM_GB" -gt 0 ] 2>/dev/null; then
      mem_used_pct=$(( 100 * mem_node_req / NODE_RAM_GB ))
      mem_free_pct=$(( 100 - mem_used_pct ))
      if [ "$mem_free_pct" -le 10 ] 2>/dev/null && [ "$gpn_num" -lt "${GPUS_PER_NODE_MAX:-999}" ] 2>/dev/null; then
        echo "ADVICE[mem]: You leave ~${mem_free_pct}% node RAM free while using only $gpn_num/${GPUS_PER_NODE_MAX:-?} GPUs; this may crowd co-runners."
        echo "             If stability is critical, request more GPUs or a whole node; if not, consider trimming --mem-per-gpu."
      elif [ "$mem_free_pct" -ge 50 ] 2>/dev/null; then
        echo "ADVICE[mem]: You leave ample RAM (~${mem_free_pct}%) on the node. If you see OOM or heavy paging, a small bump (e.g., +4–8G/GPU) may help."
      fi
    fi
  fi
fi

# 3) Contention / whole-node suggestions per system
PART=${SLURM_JOB_PARTITION:-unknown}
if [ "$SYSTEM" = "lumi" ]; then
  if [ "$PART" = "small-g" ] || [ "$PART" = "dev-g" ]; then
    if [ "$gpn_num" -lt 8 ] 2>/dev/null; then
      echo "ADVICE[contention]: LUMI $PART is shared. With $gpn_num/8 GPUs per node, co-runners may slow you down."
      echo "                    For stable performance: use 8 GPUs on 'standard-g' (whole node) or ask for --exclusive if allowed."
    fi
  elif [ "$PART" = "standard-g" ]; then
    if [ "$gpn_num" -eq 8 ] 2>/dev/null; then
      echo "ADVICE[contention]: 👍 standard-g with 8 GPUs per node requests whole nodes—least contention."
    else
      echo "ADVICE[contention]: standard-g is intended for whole nodes; prefer 8 GPUs per node here."
    fi
  fi
elif [ "$SYSTEM" = "mahti" ]; then
  if [ "$gpus_total_num" -le 2 ] 2>/dev/null && [ "$PART" = "gpusmall" ]; then
    echo "ADVICE[contention]: 👍 Mahti gpusmall is appropriate for ≤2 GPUs; good neighbor choice."
  fi
  if [ "$gpus_total_num" -gt 2 ] 2>/dev/null && [ "$PART" != "gpumedium" ]; then
    echo "ADVICE[contention]: For >2 GPUs on Mahti, 'gpumedium' tends to schedule better and reduces contention."
  fi
elif [ "$SYSTEM" = "puhti" ]; then
  if [ "$gpus_total_num" -gt 0 ] 2>/dev/null; then
    if [ "$PART" = "gpu" ] || [ "$PART" = "gputest" ]; then
      echo "ADVICE[contention]: 👍 Puhti $PART is correct for GPU jobs. Use 'gputest' only for ≤15 min trials."
    else
      echo "ADVICE[contention]: Consider the 'gpu' partition for production GPU runs on Puhti."
    fi
  fi
fi

# 4) When to DROP --mem-per-gpu (or switch style)
if [ "$gpus_total_num" -eq 0 ] 2>/dev/null && [ -n "${SLURM_MEM_PER_GPU-}" ]; then
  echo "ADVICE[mem-flag]: CPU-only job: drop --mem-per-gpu; use --mem-per-cpu or --mem instead."
fi
if [ "$SYSTEM" = "lumi" ] && [ "$PART" = "standard-g" ] && [ "$gpn_num" -eq 8 ] 2>/dev/null; then
  echo "ADVICE[mem-flag]: On LUMI whole-node jobs (8 GPUs on standard-g), --mem-per-gpu is optional."
  echo "                  You may keep it for portability, or omit memory flags and rely on node exclusivity."
fi

# 5) If you’re far below node resources with partial GPUs
if [ "$gpn_num" -gt 0 ] 2>/dev/null && [ -n "$mem_per_gpu_gb" ]; then
  # If both CPU and MEM are modest while only some GPUs used → consider +1 GPU or +CPU for better scaling
  low_cpu=$([ "$cpus_per_task_num" -lt "${REC_CPUS_PER_GPU_MIN:-0}" ] && echo 1 || echo 0)
  low_mem=$([ "$mem_per_gpu_gb" -lt "${REC_MEM_PER_GPU_GB:-0}" ] && echo 1 || echo 0)
  if [ "$low_cpu" -eq 1 ] || [ "$low_mem" -eq 1 ]; then
    echo "ADVICE[scaling]: You run modest CPU/MEM per GPU with only $gpn_num/${GPUS_PER_NODE_MAX:-?} GPUs."
    echo "                 If throughput is low, try +1 core/GPU or +4–8G/GPU; or, if the model scales, add 1–2 GPUs."
  fi
fi

##############################################################################
#                         LUMI: deeper advice (CPUs & RAM)
##############################################################################
# Run advice if we're on LUMI and we *likely* have a GPU job (any hint)
to_int_or_zero() { case "$1" in ''|*[!0-9]*) echo 0;; *) echo "$1";; esac; }
mb_to_gb()      { mb=${1:-0}; echo $(( (mb + 1023) / 1024 )); }

if [ "${SYSTEM:-}" = "lumi" ]; then
  GPUS_TOTAL_NUM=$(to_int_or_zero "${GPUS_TOTAL-}")
  # Build an *effective* GPUs-per-node even if GPUS_PER_NODE is missing
  GPN_EFFECTIVE=$(to_int_or_zero "${GPUS_PER_NODE-}")
  if [ "$GPN_EFFECTIVE" -eq 0 ]; then
    # Try SLURM_GPUS_PER_NODE (may look like "mi250:K" or "K")
    raw_gpn="${SLURM_GPUS_PER_NODE-}"
    # take last colon field or raw number
    GPN_EFFECTIVE=$(to_int_or_zero "$(printf '%s\n' "$raw_gpn" | awk -F: '{print $NF}')")
  fi
  if [ "$GPN_EFFECTIVE" -eq 0 ] && [ "$GPUS_TOTAL_NUM" -gt 0 ] 2>/dev/null; then
    NODES_INT=$(to_int_or_zero "${NODES-0}")
    [ "$NODES_INT" -gt 0 ] 2>/dev/null && GPN_EFFECTIVE=$(( GPUS_TOTAL_NUM / NODES_INT ))
  fi

  MEM_PER_GPU_MB="${SLURM_MEM_PER_GPU-}"
  MEM_PER_GPU_GB=""
  [ -n "$MEM_PER_GPU_MB" ] && MEM_PER_GPU_GB=$(mb_to_gb "$MEM_PER_GPU_MB")

  # Only proceed if we have any GPU hint at all
  if [ "$GPUS_TOTAL_NUM" -gt 0 ] 2>/dev/null || [ "$GPN_EFFECTIVE" -gt 0 ] 2>/dev/null || [ -n "$MEM_PER_GPU_GB" ]; then
    # --- Site tunables (adjust if policy changes) ---
    LUMI_CPUS_SCHEDULABLE=${LUMI_CPUS_SCHEDULABLE:-60}   # “usable” cores for user tasks
    LUMI_RAM_FULLNODE_GB=${LUMI_RAM_FULLNODE_GB:-480}    # full node
    LUMI_RAM_DEVG_GB=${LUMI_RAM_DEVG_GB:-256}            # dev-g typical cap
    LUMI_RAM_SMALLG_GB=${LUMI_RAM_SMALLG_GB:-240}        # small-g typical cap

    part="${SLURM_JOB_PARTITION:-unknown}"
    case "$part" in
      standard-g) LUMI_PART_CAP_GB=$LUMI_RAM_FULLNODE_GB ;;
      dev-g)      LUMI_PART_CAP_GB=$LUMI_RAM_DEVG_GB ;;
      small-g)    LUMI_PART_CAP_GB=$LUMI_RAM_SMALLG_GB ;;
      *)          LUMI_PART_CAP_GB=$LUMI_RAM_FULLNODE_GB ;;
    esac

    CPT=$(to_int_or_zero "${SLURM_CPUS_PER_TASK-}")
    TPN_INT=$(to_int_or_zero "${TPN-0}")
    CPUS_NODE_REQ=$(( CPT * TPN_INT ))

    MEM_NODE_REQ=""
    [ -n "$MEM_PER_GPU_GB" ] && [ "$GPN_EFFECTIVE" -gt 0 ] 2>/dev/null && MEM_NODE_REQ=$(( MEM_PER_GPU_GB * GPN_EFFECTIVE ))

    echo "---- LUMI CPU & RAM guidance ----"

    # CPUs per GPU: 7 vs 8 explanation
    if [ "$GPN_EFFECTIVE" -gt 0 ] 2>/dev/null && [ "$CPT" -gt 0 ] 2>/dev/null; then
      total_if7=$(( 7 * GPN_EFFECTIVE ))
      total_if8=$(( 8 * GPN_EFFECTIVE ))
      if [ "$CPT" -eq 7 ]; then
        head=$(( LUMI_CPUS_SCHEDULABLE - total_if7 )); [ "$head" -lt 0 ] && head=0
        echo "ADVICE[lumi-cpu]: 👍 7 CPUs/GPU → uses ${total_if7}/${LUMI_CPUS_SCHEDULABLE} "
	echo "                  schedulable cores; ~${head} left for OS/runtime."
        echo "                  Good headroom on shared nodes; keep 7 unless profiling proves you need more."
      elif [ "$CPT" -eq 8 ]; then
        head=$(( LUMI_CPUS_SCHEDULABLE - total_if8 ))
        echo "ADVICE[lumi-cpu]: 8 CPUs/GPU → uses ${total_if8}/${LUMI_CPUS_SCHEDULABLE} cores."
        if [ "$head" -le 0 ] 2>/dev/null; then
          echo "                  Fair share per GPU; leaves little headroom—best when you reserve a whole node."
        else
          echo "                  Leaves ~${head} cores; fine, but 7/GPU is often enough and more neighbor-friendly."
        fi
      else
        echo "ADVICE[lumi-cpu]: On LUMI, 7 CPUs/GPU = sweet spot (headroom); 8/GPU = fair for whole-node use."
        [ "$CPT" -lt 7 ] 2>/dev/null && echo "                  You’re quite modest on CPU; try 7/GPU unless profiling says otherwise."
        [ "$CPT" -gt 8 ] 2>/dev/null && echo "                  >8/GPU rarely helps; consider trimming to 7–8/GPU for fairness."
      fi

      # Node-level CPU etiquette hint
      if [ "$CPUS_NODE_REQ" -gt 0 ] 2>/dev/null; then
          cpu_pct=$(( 100 * CPUS_NODE_REQ / LUMI_CPUS_SCHEDULABLE ))
        if [ "$cpu_pct" -le 50 ] 2>/dev/null; then
          echo "ADVICE[cpu]: You’re using ~${cpu_pct}% of schedulable CPU on LUMI; if training is CPU-bound, +1 core/GPU may help."
        elif [ "$cpu_pct" -ge 90 ] 2>/dev/null && [ "$GPN_EFFECTIVE" -lt 8 ] 2>/dev/null; then
          echo "ADVICE[cpu]: You occupy ~${cpu_pct}% CPU with only ${GPN_EFFECTIVE}/8 GPUs; co-location may suffer."
          echo "             For stability, consider reserving more GPUs or a whole node (standard-g)."
        fi
      fi
    fi

    # Memory per GPU: 28 GiB rationale and caps
    if [ -n "$MEM_PER_GPU_GB" ]; then
      echo "ADVICE[lumi-mem]: --mem-per-gpu=${MEM_PER_GPU_GB}G × ${GPN_EFFECTIVE} GPUs ⇒ per-node ≈ ${MEM_NODE_REQ:-0}G."
      if [ "$MEM_PER_GPU_GB" -eq 28 ] 2>/dev/null; then
        head_full=$(( LUMI_RAM_FULLNODE_GB - (28 * GPN_EFFECTIVE) ))
        head_part=$(( LUMI_PART_CAP_GB     - (28 * GPN_EFFECTIVE) ))
        echo "                  28G/GPU is a solid default: generous for dataloaders, yet leaves headroom."
        echo "                  Headroom examples: full-node ${LUMI_RAM_FULLNODE_GB}G → ~${head_full}G; ${part} cap ${LUMI_PART_CAP_GB}G → ~${head_part}G."
      elif [ "$MEM_PER_GPU_GB" -lt 24 ] 2>/dev/null; then
        echo "                  Quite modest on RAM/GPU. If you see stalls/OOM, try 28–32G/GPU."
      elif [ "$MEM_PER_GPU_GB" -gt 40 ] 2>/dev/null; then
        echo "                  That’s hefty. Unless you *need* it, trimming toward 28–32G/GPU improves sharing/queuing."
      fi

      if [ -n "$MEM_NODE_REQ" ]; then
        if [ "$MEM_NODE_REQ" -le "$LUMI_PART_CAP_GB" ] 2>/dev/null; then
          free=$(( LUMI_PART_CAP_GB - MEM_NODE_REQ ))
          echo "ADVICE[lumi-mem]: Within ${part} memory cap (${LUMI_PART_CAP_GB}G); ~${free}G free per node."
        else
          over=$(( MEM_NODE_REQ - LUMI_PART_CAP_GB ))
          echo "ADVICE[lumi-mem]: ⚠️ Over the ${part} cap by ~${over}G. Trim --mem-per-gpu or change partition/shape."
        fi
      fi
    else
      # When it’s OK to drop --mem-per-gpu
      if [ "$part" = "standard-g" ] && [ "$GPN_EFFECTIVE" -eq 8 ] 2>/dev/null; then
        echo "ADVICE[lumi-mem]: Whole-node (8 GPUs on standard-g): you may omit --mem-per-gpu and rely on node exclusivity."
        echo "                  Keeping 28–32G/GPU remains fine for portability."
      fi
    fi

    # Shared partitions: remind about contention
    if [ "$part" = "small-g" ] || [ "$part" = "dev-g" ]; then
      if [ "$GPN_EFFECTIVE" -lt 8 ] 2>/dev/null; then
        echo "ADVICE[lumi-share]: ${part} is shared. With ${GPN_EFFECTIVE}/8 GPUs, co-runners may compete for CPU/IO."
        echo "                    For steadier throughput, consider standard-g with 8 GPUs (whole node)."
      fi
    fi

    # Too modest without strong reason?
    if { [ "$CPT" -gt 0 ] && [ "$CPT" -lt 7 ] 2>/dev/null; } || { [ -n "$MEM_PER_GPU_GB" ] && [ "$MEM_PER_GPU_GB" -lt 24 ] 2>/dev/null; }; then
      echo "ADVICE[lumi-modesty]: You may be under-allocating CPU and/or RAM per GPU without a clear benefit."
      echo "                      Unless profiling proves it’s fine, consider ~7 CPUs/GPU and ~28–32G/GPU."
    fi
  fi
fi
echo "====================================================================="
