(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
echo sanity-checks...

set -euo pipefail  # make failures fatal and undefined vars errors (optional)
: "${SYSTEM:?❌ SYSTEM is not set}"         
: "${PATTERN:?❌ PATTERN is not set}"         
: "${GUARD_MAX_NODES:?❌ GUARD_MAX_NODES is not set}"   
: "${GUARD_TIME:?❌ GUARD_TIME is not set}"   
: "${SLURM_CPUS_ON_NODE:?❌ SLURM_CPUS_ON_NODE is not set}"
: "${SLURM_CPUS_PER_TASK:?❌ SLURM_CPUS_PER_TASK is not set}"
: "${SLURM_GPUS_ON_NODE:?❌ SLURM_GPUS_ON_NODE is not set}"
: "${SLURM_HINT:?❌ SLURM_HINT is not set}"
: "${SLURM_JOB_ACCOUNT:?❌ missing \#SBATCH --account=...}"         # not SLURM_ACCOUNT
: "${SLURM_JOB_CPUS_PER_NODE:?❌ SLURM_JOB_CPUS_PER_NODE is not set}"
: "${SLURM_JOB_GPUS:?❌ SLURM_JOB_GPUS is not set}"
: "${SLURM_JOB_ID:?❌ missing SLURM_JOB_ID - did you run `sbatch`}" # 
: "${SLURM_JOB_NUM_NODES:?❌ missing \#SBATCH --nodes=...}"         #
: "${SLURM_JOB_PARTITION:?❌ missing \#SBATCH --partition=...}"     # not SLURM_PARTITION
: "${SLURM_NNODES:?❌ SLURM_NNODES is not set}"
: "${SLURM_NODELIST:?❌ SLURM_NODELIST is not set}"
: "${SLURM_NPROCS:?❌ SLURM_NPROCS is not set}"
: "${SLURM_NTASKS:?❌ missing \#SBATCH --ntasks=...}"               # 
: "${SLURM_NTASKS_PER_NODE:?❌ SLURM_NTASKS_PER_NODE is not set}"
: "${SLURM_TASKS_PER_NODE:?❌ SLURM_TASKS_PER_NODE is not set}"

# ----- detect GPUs (env OR job/step record) and compute GPUS_TOTAL ---------
JOBREC="$(scontrol show -d job  "${SLURM_JOB_ID:?}" 2>/dev/null | tr '\n' ' ')"
STEPREC="$(scontrol show -d step "${SLURM_JOB_ID}.batch" 2>/dev/null | tr '\n' ' ')"
REC_ALL="$JOBREC $STEPREC"

# 1) accept modern --gpus* flags via env
if [[ -n "${SLURM_GPUS_PER_TASK-}" || -n "${SLURM_GPUS_PER_NODE-}" || -n "${SLURM_GPUS-}" ]]; then
  :  # ok

# 2) accept any GRES env hints mentioning gpu (rarely exported in batch, but cheap to check)
elif [[ "${SLURM_GRES-}${SLURM_JOB_GRES-}${SLURM_STEP_GRES-}" == *gpu* ]]; then
  :  # ok

# 3) accept if job/step record mentions GPUs in any of the common fields
elif grep -Eq '(^|[[:space:]])(Gres=.*gpu|JOB_GRES=.*gpu|ReqTRES=.*gres/gpu|AllocTRES=.*gres/gpu|TresPerNode=.*gres:gpu)' <<<"$REC_ALL"; then
  :  # ok

else
  echo "❌ No GPUs requested. Use either --gpus-* OR --gres=gpu[:type]:N." >&2
  # DEBUG (optional)
  echo "DBG env: GPUS=${SLURM_GPUS-} GpT=${SLURM_GPUS_PER_TASK-} GpN=${SLURM_GPUS_PER_NODE-} GRES=${SLURM_GRES-}${SLURM_JOB_GRES-}${SLURM_STEP_GRES-}" >&2
  echo "DBG job:  $(sed 's/  */ /g' <<<"$JOBREC")" >&2
  echo "DBG step: $(sed 's/  */ /g' <<<"$STEPREC")" >&2
  exit 1
fi

# Compute total GPUs (GPUS_TOTAL) and per-node GPUs (GPUS_PER_NODE)
GPUS_TOTAL=""; GPUS_PER_NODE=""
# env totals first
if [[ -n "${SLURM_GPUS-}" && "$SLURM_GPUS" =~ ^[0-9]+$ ]]; then
  GPUS_TOTAL="$SLURM_GPUS"
elif [[ -n "${SLURM_GPUS_PER_TASK-}" && -n "${SLURM_NTASKS-}" ]]; then
  GPUS_TOTAL=$(( SLURM_GPUS_PER_TASK * SLURM_NTASKS ))
elif [[ -n "${SLURM_GPUS_PER_NODE-}" && -n "${SLURM_JOB_NUM_NODES-}" ]]; then
  GPUS_PER_NODE="${SLURM_GPUS_PER_NODE%%:*}"
  GPUS_TOTAL=$(( GPUS_PER_NODE * SLURM_JOB_NUM_NODES ))
fi

# ReqTRES/AllocTRES contain total GPUs as gres/gpu( :type )?=N
if [[ -z "$GPUS_TOTAL" && "$REC_ALL" =~ gres\/gpu([^=[:space:]]*)?=([0-9]+) ]]; then
  GPUS_TOTAL="${BASH_REMATCH[2]}"
fi

# Try per-node via TresPerNode=gres:gpu(:type)?:N
if [[ -z "${GPUS_PER_NODE:-}" ]]; then
  GPN="$(sed -n 's/.*TresPerNode=\([^[:space:]]*\).*/\1/p' <<<"$REC_ALL")"
  if [[ -n "$GPN" ]]; then
    GPN="${GPN%%(*}"
    totpn=0
    IFS=, read -ra toks <<<"$GPN"
    for t in "${toks[@]}"; do
      [[ "$t" == gres:gpu* || "$t" == gpu* ]] || continue
      n="${t##*:}"; [[ "$n" =~ ^[0-9]+$ ]] && (( totpn += n ))
    done
    (( totpn > 0 )) && GPUS_PER_NODE="$totpn"
  fi
fi

# Fallback per-node from GRES/JOB_GRES tokens
if [[ -z "$GPUS_PER_NODE" ]]; then
  # prefer JOB_GRES= if present, else GRES=
  if [[ "$REC_ALL" =~ JOB_GRES=([^[:space:]]+) ]]; then
    GRES_STR="${BASH_REMATCH[1]}"
  elif [[ "$REC_ALL" =~ [[:space:]]Gres=([^[:space:]]+) ]]; then
    GRES_STR="${BASH_REMATCH[1]}"
  else
    GRES_STR=""
  fi
  if [[ -n "$GRES_STR" ]]; then
    GRES_STR="${GRES_STR%%(*}"       # drop (IDX:...)
    totpn=0
    IFS=, read -ra toks <<<"$GRES_STR"
    for t in "${toks[@]}"; do
      [[ "$t" == gpu* ]] || continue
      n="${t##*:}"; [[ "$n" =~ ^[0-9]+$ ]] && (( totpn += n ))
    done
    (( totpn > 0 )) && GPUS_PER_NODE="$totpn"
  fi
fi

# If we have per-node and nodes, fill total
if [[ -z "$GPUS_TOTAL" && -n "$GPUS_PER_NODE" ]]; then
  if [[ -n "${SLURM_JOB_NUM_NODES-}" ]]; then
    NODES="$SLURM_JOB_NUM_NODES"
  elif [[ "$REC_ALL" =~ NumNodes=([0-9]+) ]]; then
    NODES="${BASH_REMATCH[1]}"
  else
    NODES=""
  fi
  [[ "$NODES" =~ ^[0-9]+$ ]] && GPUS_TOTAL=$(( GPUS_PER_NODE * NODES ))
fi

: "${GPUS_TOTAL:?❌ No GPUs requested. Use either --gpus-* OR --gres=gpu[:type]:N.}"
echo "OK[gpu-spec]: total=$GPUS_TOTAL per-node=${GPUS_PER_NODE:-unknown}"
# ---------------------------------------------------------------------------

JOBINFO="$(scontrol show --details job "$SLURM_JOB_ID")" || { echo "❌ ERROR: scontrol failed"; exit 1; }
field() { sed -n "s/.*$1=\([^ ]*\).*/\1/p" <<<"$JOBINFO"; } # to access scontrol output
firstnum() { sed -E 's/^([0-9]+).*/\1/'; }

###################################################
# actual stdout/stderr file paths resolved by Slurm
###################################################
out=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p')
err=$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/.*StdErr=\([^ ]*\).*/\1/p')
: "${out:?StdOut path unknown}"; 
: "${err:?StdErr path unknown}"; 
for f in "$out" "$err"; do
  [[ "$f" == "/dev/null" ]] && continue
  d=${f%/*}; [[ "$d" == "$f" ]] && d="${SLURM_SUBMIT_DIR:-$PWD}"
  [[ -d "$d" && -w "$d" ]] || { echo "❌ Missing/unwritable dir for $f: $d" >&2; exit 1; }
done

#############################
# ---- time-limit sanity ----
#############################
# Read job time limit (e.g. "1-00:00:00", "02:30:00", "45:00")
tl="$(squeue -h -j "${SLURM_JOB_ID:?}" -o %l)" || { echo "❌ squeue failed"; exit 1; }
# Treat these as unset/bad for novices
if [[ -z "$tl" || "$tl" =~ ^(NOT_SET|UNLIMITED|N/A)$ ]]; then
  echo "❌ No valid time limit set for the job. Use --time=<D-HH:MM:SS>." >&2
  exit 1
fi
slurm_time_to_seconds() { # Convert Slurm time strings to seconds
  local t="$1" d=0 h=0 m=0 s=0
  [[ -z "$t" ]] && return 1
  if [[ "$t" == *-* ]]; then d="${t%%-*}"; t="${t#*-}"; fi
  IFS=: read -r a b c <<<"$t"
  if [[ -n "$c" ]]; then h="$a"; m="$b"; s="$c"          # HH:MM:SS
  elif [[ -n "$b" ]]; then h=0; m="$a"; s="$b"          # MM:SS
  else h="$a"; m=0; s=0                                  # HH   (rare)
  fi
  echo $(( d*86400 + h*3600 + m*60 + s ))
}
tl_sec="$(slurm_time_to_seconds "$tl")" || { echo "❌ Could not parse time limit '$tl'"; exit 1; }
gt_sec="$(slurm_time_to_seconds "$GUARD_TIME")" || { echo "❌ Bad GUARD_TIME='$GUARD_TIME'"; exit 1; }
if (( tl_sec > gt_sec )); then
  echo "❌ Time limit $tl exceeds guard ${GUARD_TIME} (>${GUARD_TIME}). Reduce --time." >&2
  exit 1
fi
echo =====================================================================
printf 'Time Allocation: requested=%s (=%ss) ≤ guard=%s (=%ss)\n' "$tl" "$tl_sec" "$GUARD_TIME" "$gt_sec"
echo =====================================================================
# If you also want to catch step limits for a specific srun step, repeat with %L from
# squeue -h -j $SLURM_JOB_ID -o %L while in the step (or use SLURM_TIMELIMIT if your site exports it).


###################
# Determine "NODES"
###################
# Prefer the env var
if [[ -n "${SLURM_JOB_NUM_NODES:-}" ]]; then
  NODES="$SLURM_JOB_NUM_NODES"
else
  NODES="$(field NumNodes)"
fi
: "${NODES:?could not determine node count}"
if (( NODES > GUARD_MAX_NODES )); then
  echo "❌ ERROR: requested $NODES nodes (> $GUARD_MAX_NODES). Reduce nodes or increase GUARD_MAX_NODES." >&2
  exit 1
fi
(( NODES > 0 )) || { echo "❌ ERROR: NODES is 0. Set --nodes."; exit 1; }

################
# Tasks per Node
################
NTASKS=${SLURM_NTASKS:-0}
TPN_ENV=${SLURM_NTASKS_PER_NODE:-}      # may look like "4" or "4(x2),8(x1)"
if [[ -n "$TPN_ENV" ]]; then
  TPN=$(tr ',' '\n' <<<"$TPN_ENV" | head -1 | firstnum)
elif (( NODES > 0 )); then
  TPN=$(( NTASKS / NODES ))
else
  TPN=0
fi
(( NTASKS > 0 )) || { echo "❌ ERROR: SLURM_NTASKS is 0. Set --ntasks or --ntasks-per-node."; exit 1; }
# ntasks consistency
if (( TPN * NODES != NTASKS )); then
  echo "❌ ERROR: Inconsistent tasks: ntasks=$NTASKS but nodes=$NODES and ntasks-per-node=$TPN ⇒ ${NODES}×${TPN}=$((NODES*TPN))."
  echo "       Fix with either: (A) --nodes=X --ntasks-per-node=Y  OR  (B) --ntasks=Z matching X*Y."
  exit 1
fi

#####################
# GPUs total (robust)
#####################
# ---- Detect GPUs and compute totals: works for --gpus* and --gres on LUMI ---
set +u  # avoid nounset while parsing
GPUS_TOTAL=""; GPUS_PER_NODE=""
JOBREC="$(scontrol show -d job  "${SLURM_JOB_ID:?}" 2>/dev/null | tr '\n' ' ')"
STEPREC="$(scontrol show -d step "${SLURM_JOB_ID}.batch" 2>/dev/null | tr '\n' ' ')"
REC_ALL="$JOBREC $STEPREC"
: "${JOBREC:=}"
: "${STEPREC:=}"
: "${REC_ALL:=}"
: "${GPU_GRES_STR:=}"   # <-- critical
#echo "DBG env: GPUS=${SLURM_GPUS-} GPUS_PER_TASK=${SLURM_GPUS_PER_TASK-} GPUS_PER_NODE=${SLURM_GPUS_PER_NODE-}" >&2
#echo "DBG job:  $(sed 's/  */ /g' <<<"$JOBREC")" >&2
#echo "DBG step: $(sed 's/  */ /g' <<<"$STEPREC")" >&2

# 1) Modern --gpus* envs (may be absent on LUMI)
if [[ -n "${SLURM_GPUS-}" && "$SLURM_GPUS" =~ ^[0-9]+$ ]]; then
  GPUS_TOTAL="$SLURM_GPUS"
elif [[ -n "${SLURM_GPUS_PER_TASK-}" && -n "${SLURM_NTASKS-}" ]]; then
  GPUS_TOTAL=$(( SLURM_GPUS_PER_TASK * SLURM_NTASKS ))
elif [[ -n "${SLURM_GPUS_PER_NODE-}" && -n "${SLURM_JOB_NUM_NODES-}" ]]; then
  GPUS_PER_NODE="${SLURM_GPUS_PER_NODE%%:*}"
  GPUS_TOTAL=$(( GPUS_PER_NODE * SLURM_JOB_NUM_NODES ))
fi

# 2) TRES has total GPUs (ReqTRES/AllocTRES: "gres/gpu[:type]=N")
if [[ -z "$GPUS_TOTAL" && "$REC_ALL" =~ gres\/gpu([^=[:space:]]*)?=([0-9]+) ]]; then
  GPUS_TOTAL="${BASH_REMATCH[2]}"
fi

# 3) Per-node from TresPerNode=gres:gpu(:type)?:N (preferred on LUMI)
if [[ -z "$GPUS_PER_NODE" && "$REC_ALL" =~ TresPerNode=([^[:space:]]+) ]]; then
    GPN="${BASH_REMATCH[1]}"
    GRES_STR="$GPN"
    GPN="${GPN%%(*}"   # drop decorations like (IDX:...)
    totpn=0
    IFS=, read -ra toks <<<"$GPN"
    for t in "${toks[@]}"; do
	[[ "$t" == gres:gpu* || "$t" == gpu* ]] || continue
	n="${t##*:}"
	[[ "$n" =~ ^[0-9]+$ ]] && (( totpn += n ))
    done
    (( totpn > 0 )) && GPUS_PER_NODE="$totpn"
fi

# 4) Fallback per-node from JOB_GRES= or Gres=
if [[ -z "$GPUS_PER_NODE" ]]; then
  if   [[ "$REC_ALL" =~ JOB_GRES=([^[:space:]]+) ]]; then GRES_STR="${BASH_REMATCH[1]}"
  elif [[ "$REC_ALL" =~ [[:space:]]Gres=([^[:space:]]+) ]]; then GRES_STR="${BASH_REMATCH[1]}"
  else GRES_STR=""; fi
  if [[ -n "$GRES_STR" ]]; then
    GRES_STR="${GRES_STR%%(*}"
    totpn=0
    IFS=, read -ra toks <<<"$GRES_STR"
    for t in "${toks[@]}"; do
      [[ "$t" == gpu* ]] || continue
      n="${t##*:}"
      [[ "$n" =~ ^[0-9]+$ ]] && (( totpn += n ))
    done
    (( totpn > 0 )) && GPUS_PER_NODE="$totpn"
  fi
fi

# 5) If per-node known, compute total via nodes
if [[ -z "$GPUS_TOTAL" && -n "$GPUS_PER_NODE" ]]; then
  if   [[ -n "${SLURM_JOB_NUM_NODES-}" ]]; then NODES="$SLURM_JOB_NUM_NODES"
  elif [[ "$REC_ALL" =~ NumNodes=([0-9]+) ]]; then NODES="${BASH_REMATCH[1]}"
  else NODES=""; fi
  [[ "$NODES" =~ ^[0-9]+$ ]] && GPUS_TOTAL=$(( GPUS_PER_NODE * NODES ))
fi
set -u  # restore nounset

# Optional: forbid mixing styles (env gpus* + GRES in records)
if [[ ( -n "${SLURM_GPUS_PER_TASK-}" || -n "${SLURM_GPUS_PER_NODE-}" || -n "${SLURM_GPUS-}" ) \
      && "$REC_ALL" == *" Gres="*gpu* ]]; then
  echo "❌ Mixed GPU styles detected (--gpus* and --gres). Pick one." >&2
  exit 1
fi

# Final assert (now safe)
: "${GPUS_TOTAL:?❌ No GPUs requested. Use either --gpus-* OR --gres=gpu[:type]:N.}"
echo "OK[gpu-spec]: total=$GPUS_TOTAL per-node=${GPUS_PER_NODE:-unknown}"
# -----------------------------------------------------------------------------



################################
# per-system GPU-per-node limits
################################
case "$SYSTEM" in
  puhti)
    GPUS_PER_NODE_MAX=4     # Puhti: 4× NVIDIA V100 GPUs per node. 
    GPU_OK_REGEX='gpu:(v100)(:|$)'
    REC_CPUS_PER_GPU_MIN=10; REC_CPUS_PER_GPU_MAX=10     # 40 cores / 4 GPUs
    REC_MEM_PER_GPU_GB=95                                 # ~384 GB / 4
    ;;
  mahti)
    GPUS_PER_NODE_MAX=4     # Mahti: 4× NVIDIA A100 GPUs per GPU node (Mahti is mostly CPU nodes, but has 24 GPU nodes).
    GPU_OK_REGEX='gpu:(a100(_1g\.5gb)?)(:|$)'
    REC_CPUS_PER_GPU_MIN=8; REC_CPUS_PER_GPU_MAX=32       # up to 32/core per 1/4 node
    REC_MEM_PER_GPU_GB=128                                # 512 GB / 4
    ;;
  lumi)
    GPUS_PER_NODE_MAX=8     # LUMI-G: 8 GCDs (logical GPUs) per node (4× MI250X, each with 2 GCDs). 
    GPU_OK_REGEX='gpu:(mi250x|mi250)?(:|$)|^gpu(:|$)'
    REC_CPUS_PER_GPU_MIN=6; REC_CPUS_PER_GPU_MAX=8        # 56 usable cores/node ⇒ ~7/GPU
    REC_MEM_PER_GPU_GB=60                                 # recommended upper bound
    ;;
  *) echo "❌ ERROR: SYSTEM must be {puhti,mahti,lumi}, got '$SYSTEM'"; exit 1;;
esac
(( GPUS_PER_NODE > 0 )) || { echo "❌ ERROR: Could not determine GPUs per node from GRES/TRES."; exit 1; }

# GPU type must match the system
if [[ -n "${GRES_STR-}" ]]; then
  if ! grep -Eq "$GPU_OK_REGEX" <<<"$GRES_STR"; then
    echo "❌ ERROR: Wrong or missing GPU type in GPU request for $SYSTEM."
    echo "       Got: '$GRES_STR'"
    case "$SYSTEM" in
      puhti) echo "       Use: --gres=gpu:v100:<N> (Puhti has NVIDIA V100).";;
      mahti) echo "       Use: --gres=gpu:a100:<N> (or a100_1g.5gb on gpusmall).";;
      lumi)  echo "       Use: --gres=gpu:mi250:<K> or plain --gres=gpu:<K> (MI250X, 8 GCDs/node).";;
    esac
    exit 1
  fi
else
  # No typed GRES visible. Be strict on Puhti/Mahti; allow untyped on LUMI if you want.
  if [[ "$SYSTEM" == "lumi" ]]; then
    echo "ℹ️  No typed GRES found in job record; allowing untyped 'gpu:<K>' on LUMI."
  else
    echo "❌ ERROR: GRES not visible here. Request typed GPUs: --gres=gpu:<type>:<N>." ; exit 1
  fi
fi

###################################
# Robust "GPUS_PER_NODE" derivation
###################################
GPUS_PER_NODE=0 # fallback
GPUPN_STR="$(field GresPerNode || true)"    # e.g. "gpu:mi250x:8" or "gpu:a100_1g.5gb:2(IDX:0-1),nvme:1"
if [[ "$GPUPN_STR" == *gpu* ]]; then
    # Sum all gpu:*:* tokens; strip any trailing decorations after the count
    while IFS= read -r tok; do
	n=$(awk -F: '{print $NF}' <<<"$tok" | sed -E 's/[^0-9].*$//')
	[[ -n "$n" ]] && (( GPUS_PER_NODE += n ))
    done < <(grep -oE 'gpu[^, ]*' <<<"$GPUPN_STR")
fi
# Fallbacks if GresPerNode didn’t give it. Safe fallback: only if homogeneous
if (( GPUS_PER_NODE == 0 )); then
  TRES="$(field TRES || true)"                     # e.g. "cpu=56,mem=512G,gres/gpu=8"
  if [[ "$TRES" =~ gres/gpu=([0-9]+) ]] && (( NODES > 0 )); then
    total=${BASH_REMATCH[1]}
    if (( total % NODES == 0 )); then
      GPUS_PER_NODE=$(( total / NODES ))
    else
      echo "ERROR: GPU count is heterogeneous across nodes; cannot infer GPUS_PER_NODE from totals." >&2
      exit 1
    fi
  fi
  if (( NODES > 0 )) && (( GPUS_TOTAL % NODES == 0 )); then
      GPUS_PER_NODE=$(( GPUS_TOTAL / NODES ))
  else
      echo "ERROR: cannot infer GPUs per node (heterogeneous or unknown GresPerNode)" >&2
      exit 1
  fi
fi
# cap GPUs per node if GPUs requested
if (( GPUS_TOTAL > 0 )) && (( GPUS_PER_NODE > GPUS_PER_NODE_MAX )); then
  echo "❌ ERROR: $SYSTEM allows at most $GPUS_PER_NODE_MAX GPUs per node; requested $GPUS_PER_NODE." >&2
  exit 1
fi
# GPUs per node cap
if (( GPUS_PER_NODE > GPUS_PER_NODE_MAX )); then
  echo "❌ ERROR: Requested $GPUS_PER_NODE GPUs per node on $SYSTEM (max $GPUS_PER_NODE_MAX)."; exit 1
fi
# Don’t mix GPU knobs for novices: prefer GRES only
if [[ -n "${SLURM_GPUS:-}" || -n "${SLURM_GPUS_PER_TASK:-}" || -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
  echo "❌ ERROR: Avoid --gpus/--gpus-per-task/--gpus-per-node; use typed --gres=gpu:<type>:<N> only."
  exit 1
fi

#######################
# --- CPU Cores per GPU
#######################
CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-0}
(( CPUS_PER_TASK > 0 )) || { echo "❌ ERROR: SLURM_CPUS_PER_TASK is 0. Set --cpus-per-task."; exit 1; }

GPUS_PER_TASK=${SLURM_GPUS_PER_TASK:-0}

# ---------- CPU cores per GPU ----------
if (( GPUS_PER_NODE > 0 )); then
  # expect CPUS_PER_TASK within recommended bracket
  if (( CPUS_PER_TASK < REC_CPUS_PER_GPU_MIN || CPUS_PER_TASK > REC_CPUS_PER_GPU_MAX )); then
    echo "❌ ERROR: cpus-per-task=$CPUS_PER_TASK is not a good choice on $SYSTEM."
    case "$SYSTEM" in
      puhti) echo "       Use exactly 10 CPU cores per GPU (40 cores/node ÷ 4 GPUs)." ;;
      mahti) echo "       Use 8–32 CPU cores per GPU (up to 1/4 node/GPU).";;
      lumi)  echo "       Use 6–8 CPU cores per GPU; 7 is typical (56 usable cores/node ÷ 8 GCDs)." ;;
    esac
    exit 1
  fi
fi

########################################
# partition hints (Puhti / Mahti / LUMI)
########################################
: "${SLURM_JOB_PARTITION:?❌ SLURM_JOB_PARTITION is not set}"
: "${GPUS_PER_NODE:?❌ GPUS_PER_NODE is not set}"
echo "Running novice's partition checks..."
PART=${SLURM_JOB_PARTITION:-unknown}
case "$SYSTEM" in
  lumi)
    # LUMI-G partitions: small-g (≤4 nodes, GCD-level), dev-g (short), standard-g (whole nodes)
    if (( NODES <= 4 )) && [[ "$PART" != "small-g" && "$PART" != "dev-g" ]]; then
      echo "⚠️  HINT[LUMI]: ≤4 nodes usually fit better in 'small-g' (or 'dev-g' for quick tests). Current: $PART"
    fi
    # Guard against asking >4 nodes in small-g
    if [[ "$PART" == "small-g" ]] && (( NODES > 4 )); then
      echo "❌ ERROR[LUMI]: 'small-g' allows up to 4 nodes. Requested: $NODES. Use 'standard-g' instead." >&2
      exit 1
    fi
    ;;

  puhti)
    # Puhti GPU work belongs to 'gpu' (or 'gputest' for very short trials)
    if (( GPUS_TOTAL > 0 )) && [[ "$PART" != "gpu" && "$PART" != "gputest" ]]; then
      echo "⚠️  HINT[Puhti]: GPU jobs run in 'gpu' (or 'gputest' for ≤15 min tests). Current: $PART"
    fi
    ;;

  mahti)
    # Mahti GPU: gpusmall (≤2 GPUs), gpumedium (>2 GPUs)
    if (( GPUS_TOTAL > 0 )); then
      if (( GPUS_TOTAL <= 2 )) && [[ "$PART" != "gpusmall" ]]; then
        echo "⚠️  HINT[Mahti]: ≤2 GPUs → use 'gpusmall' for better fit. Current: $PART"
      fi
      if (( GPUS_TOTAL > 2 )) && [[ "$PART" != "gpumedium" ]]; then
        echo ⚠️  "HINT[Mahti]: >2 GPUs → use 'gpumedium'. Current: $PART"
      fi
    fi
    ;;
esac
echo PART=$PART
# Partition-specific caps
if [[ "$SYSTEM" == "lumi" && "$PART" == "standard-g" ]] && (( GPUS_PER_NODE != 8 )); then
  echo "❌ ERROR[LUMI]: 'standard-g' allocates whole nodes. Request all 8 GPUs/GCDs per node (got $GPUS_PER_NODE)." ; exit 1
fi
if [[ "$SYSTEM" == "mahti" && "$PART" == "gpusmall" ]] && (( GPUS_PER_NODE > 2 )); then
  echo "❌ ERROR[Mahti]: 'gpusmall' allows at most 2 A100 GPUs per job (got $GPUS_PER_NODE)." ; exit 1
fi









# ---------- Slurm vs. torchrun patterns ----------
if [[ "$PATTERN" == "torchrun" ]]; then
  # Pattern B: 1 task per node; torchrun spawns NPROC_PER_NODE == GPUs/node
  if (( TPN != 1 )); then
    echo "❌ ERROR: With torchrun, use --ntasks-per-node=1 (got $TPN). torchrun will spawn per-GPU workers."
    exit 1
  fi
  if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    echo "❌ ERROR: Set NPROC_PER_NODE to match GPUs per node ($GPUS_PER_NODE) for torchrun."
    exit 1
  fi
  if (( NPROC_PER_NODE != GPUS_PER_NODE )); then
    echo "❌ ERROR: torchrun NPROC_PER_NODE=$NPROC_PER_NODE but GPUs per node=$GPUS_PER_NODE. They must match."
    exit 1
  fi
else
  # Pattern A: Slurm spawns one task per GPU
  if (( TPN != GPUS_PER_NODE )); then
    echo "❌ ERROR: Use one task per GPU. ntasks-per-node=$TPN but GPUs per node=$GPUS_PER_NODE."
    echo "       Fix: --ntasks-per-node=$GPUS_PER_NODE (and keep --gres=gpu:...:$GPUS_PER_NODE)."
    exit 1
  fi
  if (( GPUS_PER_TASK != 0 && GPUS_PER_TASK != 1 )); then
    echo "❌ ERROR: Use --gpus-per-task=1 when Slurm launches one task per GPU (got $GPUS_PER_TASK)."
    exit 1
  fi
fi

######################################################
# ---------- MEMORY SANITY (GPU & CPU jobs) ----------
######################################################
# Slurm exports these (in MB) only if user requested them (MB units from env; absent if not set)
MEM_PER_GPU_MB=${SLURM_MEM_PER_GPU:-}
MEM_PER_CPU_MB=${SLURM_MEM_PER_CPU:-}
MEM_PER_NODE_MB=${SLURM_MEM_PER_NODE:-}

# ---------- Memory per GPU (system RAM, NOT GPU HBM) ----------
if [[ -n "$MEM_PER_NODE_MB" && -n "$MEM_PER_GPU_MB" ]]; then
  echo "❌ ERROR: Don't mix --mem and --mem-per-gpu. Pick one (prefer --mem-per-gpu for GPU jobs)."; exit 1
fi


if [[ -n "$MEM_PER_GPU_MB" ]]; then
    mem_gb=$(( (MEM_PER_GPU_MB + 1023) / 1024 ))
    # Special case: Mahti A100 slices have fixed mem & CPU limits
    if [[ "$SYSTEM" == "mahti" && "$GRES" == *"a100_1g.5gb"* ]]; then
	if (( CPUS_PER_TASK > 4 )); then
	    echo "❌ ERROR[Mahti slice]: a100_1g.5gb allows at most 4 CPU cores per job."; exit 1
	fi
	if (( mem_gb != 18 )); then
	    echo "❌ ERROR[Mahti slice]: a100_1g.5gb jobs get ~17.5 GiB automatically; don't set --mem-per-gpu (got ${mem_gb}G)."
	    exit 1
	fi
    else
	# General caps per system
	case "$SYSTEM" in
	    puhti) if (( mem_gb > 95 )); then echo "❌ ERROR: --mem-per-gpu=${mem_gb}G too high for Puhti; ~95G/GPU is the practical max."; exit 1; fi ;;
	    mahti) if (( mem_gb > 128 )); then echo "❌ ERROR: --mem-per-gpu=${mem_gb}G too high for Mahti; ≤128G/GPU (512G/node ÷ 4)."; exit 1; fi ;;
	    lumi)  if (( mem_gb > 64 )); then echo "❌ ERROR: --mem-per-gpu=${mem_gb}G too high for LUMI; keep ≤60–64G/GPU."; exit 1; fi ;;
	esac
    fi
fi


# Helper: MB->GB (rounded up)
mb_to_gb() { local x=${1:-0}; echo $(( (x + 1023) / 1024 )); }

# Node RAM caps (conservative, rounded) + recommended per-GPU ceilings (GB of system RAM)
case "$SYSTEM" in
  puhti) NODE_RAM_GB=384; REC_PER_GPU_MAX_GB=95  ;;   # 4× V100 per node
  mahti) NODE_RAM_GB=512; REC_PER_GPU_MAX_GB=128 ;;   # 4× A100 per node
  lumi)  NODE_RAM_GB=512; REC_PER_GPU_MAX_GB=64  ;;   # 8 GCDs (MI250X) per node
  *) echo "ERROR: unknown SYSTEM for memory checks"; exit 1;;
esac


# 0) forbid mutually exclusive / confusing mixes
mix_count=0
[[ -n "$MEM_PER_GPU_MB"  ]] && mix_count=$((mix_count+1))
[[ -n "$MEM_PER_CPU_MB"  ]] && mix_count=$((mix_count+1))
[[ -n "$MEM_PER_NODE_MB" ]] && mix_count=$((mix_count+1))
if (( mix_count > 1 )); then
  echo "ERROR: Do not mix --mem/--mem-per-cpu/--mem-per-gpu. Pick ONE. (For GPU jobs: use --mem-per-gpu.)" >&2
  exit 1
fi

# 1) GPU jobs: require --mem-per-gpu; forbid others
#
# Request host memory explicitly, even if GPUs aren’t shareable. GPUs
# and CPU RAM are scheduled separately. On LUMI/Puhti, if you take
# fewer than all GPUs on a node, other jobs can share the node’s host
# memory; without --mem / --mem-per-gpu Slurm may give you too little
# (OOM kills) or your job may overcommit memory that others need.
if (( GPUS_TOTAL > 0 )); then
  if [[ -z "$MEM_PER_GPU_MB" ]]; then
    echo "ERROR: GPU job without --mem-per-gpu. Request memory per GPU (e.g. --mem-per-gpu=${REC_PER_GPU_MAX_GB}G)." >&2
    exit 1
  fi
  if [[ -n "$MEM_PER_CPU_MB" || -n "$MEM_PER_NODE_MB" ]]; then
    echo "ERROR: For GPU jobs, use --mem-per-gpu only (not --mem or --mem-per-cpu)." >&2
    exit 1
  fi

  mem_per_gpu_gb=$(mb_to_gb "$MEM_PER_GPU_MB")
  # per-GPU ceiling (teach safe upper bound)
  if (( mem_per_gpu_gb > REC_PER_GPU_MAX_GB )); then
    echo "ERROR: --mem-per-gpu=${mem_per_gpu_gb}G is too high on $SYSTEM. Keep ≤ ${REC_PER_GPU_MAX_GB}G/GPU." >&2
    exit 1
  fi

  # per-node feasibility
  if (( GPUS_PER_NODE > 0 )); then
    req_per_node_gb=$(( mem_per_gpu_gb * GPUS_PER_NODE ))
    if (( req_per_node_gb > NODE_RAM_GB )); then
      echo "ERROR: Per-node memory ${req_per_node_gb}G exceeds node capacity ${NODE_RAM_GB}G on $SYSTEM." >&2
      echo "       Reduce --mem-per-gpu or GPUs per node." >&2
      exit 1
    fi
  fi

  # Special case: Mahti A100 slice queue (gpusmall) with a100_1g.5gb
  if [[ "$SYSTEM" == "mahti" && "${GRES:-}" == *"a100_1g.5gb"* ]]; then
    # Slice jobs come with fixed RAM (~17.5 GiB). Users should NOT set --mem-per-gpu.
    echo "ERROR[Mahti gpusmall]: a100_1g.5gb slice has fixed RAM; do not set --mem-per-gpu (got ${mem_per_gpu_gb}G)." >&2
    echo "       Remove memory flag; Slurm assigns slice RAM automatically." >&2
    exit 1
  fi
fi

# 2) CPU-only jobs: disallow --mem-per-gpu; allow either --mem-per-cpu (preferred) or --mem
if (( GPUS_TOTAL == 0 )); then
  if [[ -n "$MEM_PER_GPU_MB" ]]; then
    echo "ERROR: CPU job should not set --mem-per-gpu. Use --mem-per-cpu or --mem." >&2
    exit 1
  fi
  if [[ -z "$MEM_PER_CPU_MB" && -z "$MEM_PER_NODE_MB" ]]; then
    echo "ERROR: CPU job without memory request. Set --mem-per-cpu=<GB> or --mem=<GB>." >&2
    exit 1
  fi

  # Validate node feasibility
  tasks_per_node=${TPN:-0}
  if [[ -n "$MEM_PER_CPU_MB" ]]; then
    mem_cpu_gb=$(mb_to_gb "$MEM_PER_CPU_MB")
    # per-node CPUs = cpus-per-task * tasks-per-node
    if (( CPUS_PER_TASK > 0 && tasks_per_node > 0 )); then
      cpus_node=$(( CPUS_PER_TASK * tasks_per_node ))
      req_per_node_gb=$(( mem_cpu_gb * cpus_node ))
      if (( req_per_node_gb > NODE_RAM_GB )); then
        echo "ERROR: --mem-per-cpu=$mem_cpu_gb G → ${req_per_node_gb}G per node exceeds ${NODE_RAM_GB}G on $SYSTEM." >&2
        echo "       Reduce memory per CPU or cpus-per-task." >&2
        exit 1
      fi
    fi
  fi

  if [[ -n "$MEM_PER_NODE_MB" ]]; then
    mem_node_gb=$(mb_to_gb "$MEM_PER_NODE_MB")
    if (( mem_node_gb > NODE_RAM_GB )); then
      echo "ERROR: --mem=${mem_node_gb}G exceeds node capacity ${NODE_RAM_GB}G on $SYSTEM." >&2
      exit 1
    fi
  fi
fi

# 3) Friendly echo (so users see what was picked)
if (( GPUS_TOTAL > 0 )); then
  echo "OK[mem]: --mem-per-gpu=$(mb_to_gb ${MEM_PER_GPU_MB})G  x  ${GPUS_PER_NODE} GPU(s)/node  (node cap ${NODE_RAM_GB}G)"
else
  [[ -n "$MEM_PER_CPU_MB"  ]] && echo "OK[mem]: --mem-per-cpu=$(mb_to_gb ${MEM_PER_CPU_MB})G"
  [[ -n "$MEM_PER_NODE_MB" ]] && echo "OK[mem]: --mem=$(mb_to_gb ${MEM_PER_NODE_MB})G"
fi



# ---------- final friendly summary ----------
echo ==============================================
echo " STDERR                    : $err"
echo " STDOUT                    : $out"
echo " TIMELIMIT                 : $tl"
echo " GUARD_TIME                : $GUARD_TIME"
echo " SYSTEM                    : $SYSTEM"
echo " GUARD_MAX_NODES           : $GUARD_MAX_NODES"
echo " local PATTERN             : $PATTERN"
echo " SLURM_JOB_ID              : $SLURM_JOB_ID"               
echo " local NODES               : $NODES"
echo " SLURM_NTASKS              : $NTASKS"
echo " SLURM_NTASKS_PER_NODE     : $SLURM_NTASKS_PER_NODE"
echo " local TASKS_PER_NODE (TPN): $TPN"
echo " local GPUS_TOTAL          : $GPUS_TOTAL"
echo " local GPUS_PER_NODE_MAX   : $GPUS_PER_NODE_MAX"
echo " local GPUS_PER_NODE       : $GPUS_PER_NODE"
echo " local GPU_OK_REGEX        : $GPU_OK_REGEX"
echo " local GRES_STR            : $GRES_STR"
echo " local REC_CPUS_PER_GPU_MIN: $REC_CPUS_PER_GPU_MIN"
echo " local REC_MEM_PER_GPU_GB  : $REC_MEM_PER_GPU_GB"
echo " SLURM_CPUS_PER_TASK       : $SLURM_CPUS_PER_TASK"
echo " GPUS_PER_TASK             : $GPUS_PER_TASK"
echo " SLURM_JOB_PARTITION       : $SLURM_JOB_PARTITION"
echo " SLURM_MEM_PER_GPU         : ${SLURM_MEM_PER_GPU-}"
echo " SLURM_MEM_PER_CPU         : ${SLURM_MEM_PER_CPU-}"
echo " SLURM_MEM_PER_NODE        : ${SLURM_MEM_PER_NODE-}"
echo " local NODE_RAM_GB         : $NODE_RAM_GB"
echo ==============================================


# Copyright (c) 2025.  Creative Commons.
# - The idea of writing this: Anssi Yli-Jyrä.
# - The implementation:  This was written with the help of ChatGPT 5.  You are free to improve and publish.

# Why these rules (quick refs)
# GPU type per system:
# - Puhti uses V100,
# - Mahti uses A100 (plus slices on gpusmall),
# - LUMI-G uses MI250X (exposed as 8 GCDs/node).
# Request with typed --gres=gpu:<type>:N.  (Docs CSC, Docs CSC, docs.lumi-supercomputer.eu)
#
# One process per GPU is the usual ML pattern; either Slurm launches N tasks per node (one per GPU) or
# torchrun does with --nproc_per_node = GPUs/node (then --ntasks-per-node=1).  (a3s.fi)
#
# CPU cores per GPU (data loading):
# - Puhti ≈10;
# - Mahti up to 32;
# - LUMI about 7 (56 usable cores/node).
# (Docs CSC, csc-guide-preview.rahtiapp.fi, docs.lumi-supercomputer.eu)
#
# Memory per GPU here means system RAM, not HBM on the GPU.
# - Puhti ≈95 GB/GPU (384 GB/node),
# - Mahti ≈128 GB/GPU (512 GB/node),
# - LUMI keep ≤60–64 GB/GPU. 
# (csc-guide-preview.rahtiapp.fi, Docs CSC, lumi-supercomputer.github.io)
#
# LUMI partitions:
# - standard-g is whole-node; don’t request fewer than 8 GPUs there.
# - For ≤4 nodes, prefer small-g/dev-g. (lumi-supercomputer.github.io)
#
# Mahti gpusmall limits:
# - max 2 A100 or one A100 slice with ≤4 CPU cores and ~17.5 GiB RAM. (Docs CSC)

export SANITY_CHECKS_OK=1


