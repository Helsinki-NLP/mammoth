#!/usr/bin/env bash
set -euo pipefail

# ---------- inputs ----------
# Required: JOB_DIR, JOB_NAME, JOB_SYSTEM (lumi|puhti|mahti|roihu), JOB_NODES, JOB_GPUS, JOB_TIME
# Optional: JOB_PATTERN (slurm|torchrun), JOB_PARTITION, JOB_CPUS_PER_TASK, JOB_MEM
# Optional: PARAMS="$JOB_DIR/cfg/params.sh" (sourced if file exists)
# Output  : "$JOB_DIR/cfg/sbatch-entry.slurm" (use OVERWRITE=1 to replace)

# ---- required baseline ----
: "${PROJHOME:?❌ PROJHOME not set}"
: "${JOB_NAME:?❌ JOB_NAME not set}"
JOB_DIR="$PROJHOME/data/$JOB_NAME"
PARAMS="$JOB_DIR/cfg/params.sh"  # we are taking the risk of reading file instead of studying the JOB_NAME
[[ -r "$PARAMS" ]] && . "$PARAMS"
JOB_GPUS_PER_NODE=$JOB_GPUS
JOB_SYSTEM="${JOB_SYSTEM,,}"          # normalize
JOB_PATTERN="${JOB_PATTERN:-slurm}"   # slurm|torchrun
need(){ for v; do [[ -n "${!v-}" ]] || { echo "❌ $v missing" >&2; exit 1; }; done; }
need JOB_DIR JOB_SYSTEM JOB_NODES JOB_GPUS_PER_NODE JOB_TIME


#!/usr/bin/env bash
# Minimal, predictable defaults (+ small knobs)

set -euo pipefail

# ---- REQUIRED env -----------------------------------------------------------
: "${JOB_SYSTEM:?set JOB_SYSTEM=lumi|puhti|mahti|roihu}"
: "${JOB_NODES:?set JOB_NODES}"
: "${JOB_GPUS_PER_NODE:?set JOB_GPUS_PER_NODE (0 for CPU-only)}"
: "${JOB_TIME:?set JOB_TIME (HH:MM:SS or D-HH:MM:SS)}"
JOB_PATTERN="${JOB_PATTERN:-slurm}"             # slurm|torchrun

# ---- POLICY (tiny knobs you may tune) ---------------------------------------
JOB_CPU_SPLIT_POLICY="${JOB_CPU_SPLIT_POLICY:-max}"  # max=share by node max GPUs; used=share by GPUs-in-use
# Floors/caps per GPU (conservative, tweak if you want)
PER_GPU_CPU_CAP_LUMI=14;  MIN_CPU_PER_GPU_LUMI=7
PER_GPU_CPU_CAP_PUHTI=10; MIN_CPU_PER_GPU_PUHTI=8
PER_GPU_CPU_CAP_MAHTI=14; MIN_CPU_PER_GPU_MAHTI=8

# ---- NODE PROFILES ----------------------------------------------------------
case "${JOB_SYSTEM,,}" in
  lumi)  CPUS_NODE=64; MEM_NODE=240; MAX_GPN=8;  GPU_TYPE="mi250"; CPU_HEAD=8; MEM_HEAD=8;
         PER_GPU_CPU_CAP=$PER_GPU_CPU_CAP_LUMI;  MIN_CPU_PER_GPU=$MIN_CPU_PER_GPU_LUMI;  MEM_MIN_PER_GPU=30 ;;
  puhti) CPUS_NODE=40; MEM_NODE=180; MAX_GPN=4;  GPU_TYPE="";       CPU_HEAD=2; MEM_HEAD=8;
         PER_GPU_CPU_CAP=$PER_GPU_CPU_CAP_PUHTI; MIN_CPU_PER_GPU=$MIN_CPU_PER_GPU_PUHTI; MEM_MIN_PER_GPU=45 ;;
  mahti) CPUS_NODE=64; MEM_NODE=240; MAX_GPN=4;  GPU_TYPE="";       CPU_HEAD=4; MEM_HEAD=8;
         PER_GPU_CPU_CAP=$PER_GPU_CPU_CAP_MAHTI; MIN_CPU_PER_GPU=$MIN_CPU_PER_GPU_MAHTI; MEM_MIN_PER_GPU=60 ;;
  roihu) CPUS_NODE=40; MEM_NODE=180; MAX_GPN=4;  GPU_TYPE="";       CPU_HEAD=2; MEM_HEAD=8;
         PER_GPU_CPU_CAP=$PER_GPU_CPU_CAP_PUHTI; MIN_CPU_PER_GPU=$MIN_CPU_PER_GPU_PUHTI; MEM_MIN_PER_GPU=45 ;;
  *)     echo "❌ Unknown JOB_SYSTEM: $JOB_SYSTEM" >&2; exit 1 ;;
esac

usable_cores=$(( CPUS_NODE - CPU_HEAD ))
usable_mem=$(( MEM_NODE - MEM_HEAD  ))
(( usable_cores > 0 )) || usable_cores=1
(( usable_mem  > 0 ))  || usable_mem=1

GPN="${JOB_GPUS_PER_NODE}"

# ---- MEMORY (GiB per node) --------------------------------------------------
if (( GPN == 0 )); then
  # CPU-only default: half the node unless user overrides
  : "${JOB_MEM:=$(( usable_mem / 2 ))G}"
else
  per_gpu_share=$(( usable_mem / MAX_GPN ))          # share-friendly by MAX_GPN
  (( per_gpu_share < MEM_MIN_PER_GPU )) && per_gpu_share=$MEM_MIN_PER_GPU
  want_mem=$(( per_gpu_share * GPN ))
  (( want_mem > usable_mem )) && want_mem=$usable_mem
  : "${JOB_MEM:=${want_mem}G}"
fi

# ---- CPUs & tasks -----------------------------------------------------------
if [[ "$JOB_CPU_SPLIT_POLICY" == "max" ]]; then
  denom=$(( MAX_GPN > 0 ? MAX_GPN : 1 ))            # share by max GPUs/node
else
  denom=$(( GPN    > 0 ? GPN    : 1 ))              # share by GPUs-in-use
fi

base_per_gpu=$(( usable_cores / denom ))            # floor

if [[ "$JOB_PATTERN" == "torchrun" ]]; then
  NTASKS_PER_NODE=1
  GPUS_PER_NODE="$GPN"; GPUS_PER_TASK=""
  if (( GPN == 0 )); then
    : "${JOB_CPUS_PER_TASK:=$usable_cores}"
  else
    want=$(( base_per_gpu * GPN ))
    cap=$(( PER_GPU_CPU_CAP * GPN ))
    (( want > cap )) && want=$cap
    floor=$(( MIN_CPU_PER_GPU * GPN ))
    (( want < floor )) && want=$floor
    (( want > usable_cores )) && want=$usable_cores
    : "${JOB_CPUS_PER_TASK:=$want}"
  fi
else
  if (( GPN == 0 )); then
    NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
    GPUS_PER_TASK=""; GPUS_PER_NODE=""
    : "${JOB_CPUS_PER_TASK:=$usable_cores}"
  else
    NTASKS_PER_NODE="$GPN"
    GPUS_PER_TASK=1; GPUS_PER_NODE=""
    cpt="$base_per_gpu"
    (( cpt > PER_GPU_CPU_CAP )) && cpt=$PER_GPU_CPU_CAP
    (( cpt < MIN_CPU_PER_GPU )) && cpt=$MIN_CPU_PER_GPU
    : "${JOB_CPUS_PER_TASK:=$cpt}"
  fi
fi


# ---- GPU request normalization --------------------------------------------
# Accept JOB_GPUS_PER_NODE or JOB_GPUS_TOTAL; set JOB_GPUS (per node) and JOB_GPUS_TOTAL
normalize_gpu_request() {
  : "${JOB_NODES:?missing JOB_NODES}"
  if [[ -n "${JOB_GPUS_PER_NODE:-}" ]]; then
    JOB_GPUS="$JOB_GPUS_PER_NODE"
  elif [[ -n "${JOB_GPUS_TOTAL:-}" ]]; then
    # pack across nodes, prefer even distribution; clamp to per-node max
    local per=$(( (JOB_GPUS_TOTAL + JOB_NODES - 1) / JOB_NODES ))
    JOB_GPUS=$per
  else
    # default: GPU count from legacy JOB_GPUS or 0 if CPU job
    JOB_GPUS="${JOB_GPUS:-0}"
  fi
  (( JOB_GPUS < 0 )) && JOB_GPUS=0
  if (( JOB_GPUS > MAX_GPN )); then
    echo "⚠️  Requested ${JOB_GPUS} GPUs/node > MAX_GPN=$MAX_GPN on $JOB_SYSTEM; clamping." >&2
    JOB_GPUS=$MAX_GPN
  fi
  JOB_GPUS_TOTAL=$(( JOB_NODES * JOB_GPUS ))
}
normalize_gpu_request


# ---- time helpers & partition choice ---------------------------------------
time_to_minutes(){
  local t="$1" d=0 h=0 m=0 s=0
  if [[ "$t" =~ ^([0-9]+)-([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    d=${BASH_REMATCH[1]}; h=${BASH_REMATCH[2]}; m=${BASH_REMATCH[3]}; s=${BASH_REMATCH[4]}
  elif [[ "$t" =~ ^([0-9]{1,2}):([0-9]{2}):([0-9]{2})$ ]]; then
    h=${BASH_REMATCH[1]}; m=${BASH_REMATCH[2]}; s=${BASH_REMATCH[3]}
  else
    echo "❌ JOB_TIME '$t' not in HH:MM:SS or D-HH:MM:SS" >&2; exit 1
  fi
  echo $(( d*1440 + h*60 + m + (s>0 ? 1 : 0) ))
}

choose_partition(){
  [[ -n "${JOB_PARTITION:-}" ]] && { echo "$JOB_PARTITION"; return; }
  local mins; mins="$(time_to_minutes "$JOB_TIME")"
  if (( JOB_GPUS > 0 )); then
    case "$JOB_SYSTEM" in
      lumi)  (( mins <= 15 )) && echo dev-g      || { (( mins <= 240 )) && echo small-g || echo standard-g; } ;;
      puhti) (( mins <= 15 )) && echo gputest    || echo gpu ;;
      mahti) echo gpusmall ;;
      roihu) echo gpu ;;
      *)     echo gpu ;;
    esac
  else
    case "$JOB_SYSTEM" in
      lumi)  (( mins <= 15 )) && echo debug      || { (( mins <= 240 )) && echo small   || echo standard; } ;;
      puhti|mahti|roihu) echo small ;;
      *) echo small ;;
    esac
  fi
}
JOB_PARTITION="$(choose_partition)"

# ---- CPU/mem defaults (scaled; keep headroom) ------------------------------
# Scaled CPU/mem with headroom.
# Defaults are sharing-friendly: split by MAX GPUs per node (LUMI=8).
# Set JOB_EXCLUSIVE=1 (or JOB_CPU_SPLIT_POLICY=used) to grab more.
choose_defaults() {
  local pattern="${JOB_PATTERN:-slurm}"
  local gpn="${JOB_GPUS:-0}"
  local policy="${JOB_CPU_SPLIT_POLICY:-max}"   # default 'max' for sharing
  [[ "${JOB_EXCLUSIVE:-0}" = "1" ]] && policy="used"

  # node profile already defined earlier:
  # CPUS_NODE, MEM_NODE, MAX_GPN, CPU_HEAD, MEM_HEAD, JOB_SYSTEM, GPU_TYPE
  # per-system caps/floors (tune as needed)
  case "$JOB_SYSTEM" in
    lumi)  PER_GPU_CPU_CAP=${PER_GPU_CPU_CAP:-14}; MIN_CPU_PER_GPU=${MIN_CPU_PER_GPU:-6};  MEM_MIN_PER_GPU=${MEM_MIN_PER_GPU:-30} ;;
    puhti) PER_GPU_CPU_CAP=${PER_GPU_CPU_CAP:-10}; MIN_CPU_PER_GPU=${MIN_CPU_PER_GPU:-6};  MEM_MIN_PER_GPU=${MEM_MIN_PER_GPU:-45} ;;
    mahti) PER_GPU_CPU_CAP=${PER_GPU_CPU_CAP:-14}; MIN_CPU_PER_GPU=${MIN_CPU_PER_GPU:-8};  MEM_MIN_PER_GPU=${MEM_MIN_PER_GPU:-60} ;;
    *)     PER_GPU_CPU_CAP=${PER_GPU_CPU_CAP:-10}; MIN_CPU_PER_GPU=${MIN_CPU_PER_GPU:-6};  MEM_MIN_PER_GPU=${MEM_MIN_PER_GPU:-40} ;;
  esac

  # usable after headroom
  local usable_cores=$(( CPUS_NODE - CPU_HEAD )); (( usable_cores < 1 )) && usable_cores=1
  local usable_mem=$(( MEM_NODE - MEM_HEAD  ));   (( usable_mem  < 1 )) && usable_mem=1

  # ---- memory (GiB per node) ----
  local mem_alloc
  if (( gpn <= 0 )); then
    mem_alloc=${JOB_MEM_GIB:-$(( usable_mem / 2 ))}           # CPU-only default
  else
    # per-GPU share based on MAX_GPN (sharing-friendly), with floor; cap to usable_mem
    local per_gpu_share=$(( usable_mem / (MAX_GPN>0?MAX_GPN:1) ))
    (( per_gpu_share < MEM_MIN_PER_GPU )) && per_gpu_share=$MEM_MIN_PER_GPU
    mem_alloc=$(( per_gpu_share * gpn ))
    (( mem_alloc > usable_mem )) && mem_alloc=$usable_mem
  fi
  [[ -n "${JOB_MEM:-}" ]] || JOB_MEM="${mem_alloc}G"

  # helper: per-GPU CPU base according to policy
  local denom
  if [[ "$policy" == "max" ]]; then denom=$(( MAX_GPN > 0 ? MAX_GPN : 1 ))   # share-friendly
  else                              denom=$(( gpn     > 0 ? gpn     : 1 ))   # grab proportional
  fi
  local base_per_gpu=$(( usable_cores / denom ))

  if [[ "$pattern" = "torchrun" ]]; then
    # 1 Slurm task per node; torchrun forks gpn workers
    NTASKS_PER_NODE=1
    GPUS_PER_NODE="$gpn"; GPUS_PER_TASK=""

    if [[ -z "${JOB_CPUS_PER_TASK:-}" ]]; then
      # total CPUs for the single task = clamp(base_per_gpu * gpn)
      local want=$(( base_per_gpu * (gpn>0?gpn:1) ))
      local cap=$(( PER_GPU_CPU_CAP * (gpn>0?gpn:1) ))
      (( want > cap )) && want=$cap
      local floor=$(( MIN_CPU_PER_GPU * (gpn>0?gpn:1) ))
      (( want < floor )) && want=$floor
      (( want > usable_cores )) && want=$usable_cores
      JOB_CPUS_PER_TASK=$want
    fi

  else
    # per-GPU tasks
    if (( gpn > 0 )); then
      NTASKS_PER_NODE="$gpn"
      GPUS_PER_TASK=1; GPUS_PER_NODE=""
      if [[ -z "${JOB_CPUS_PER_TASK:-}" ]]; then
        local cpt=$base_per_gpu
        (( cpt > PER_GPU_CPU_CAP )) && cpt=$PER_GPU_CPU_CAP
        (( cpt < MIN_CPU_PER_GPU )) && cpt=$MIN_CPU_PER_GPU
        JOB_CPUS_PER_TASK=$cpt
      fi
    else
      # CPU-only
      NTASKS_PER_NODE="${NTASKS_PER_NODE:-1}"
      GPUS_PER_TASK=""; GPUS_PER_NODE=""
      [[ -n "${JOB_CPUS_PER_TASK:-}" ]] || JOB_CPUS_PER_TASK=$usable_cores
    fi
  fi
}
choose_defaults

# ---- ASCII viz --------------------------------------------------------------
viz_block(){
  local pattern="${JOB_PATTERN:-slurm}" cpus="$JOB_CPUS_PER_TASK" gpn="$JOB_GPUS"
  if (( JOB_GPUS <= 0 )); then
    cat <<EOF
# CPU-only node:
# ┌──────────┐
# │ Task 0   │ CPU 0–$((cpus-1))
# └──────────┘
EOF
    return
  fi

  if [[ "$pattern" = "torchrun" ]]; then
    if (( JOB_NODES == 1 )); then
      cat <<EOF
# torchrun: 1 task/node; spawns $gpn GPU workers
# ┌──────────────────────────┐
# │ Task 0                   │ → torchrun → $gpn procs → GPUs 0–$((gpn-1))
# │ CPU 0–$((cpus-1))$( ((CPUS_NODE>cpus)) && printf ' (subset of node)') │
# └──────────────────────────┘
EOF
    else
      cat <<EOF
# torchrun multi-node: 1 task per node; $gpn GPU workers per node
# ╔══════════════════════╗     ╔══════════════════════╗
# ║       Node 1         ║ ... ║      Node ${JOB_NODES}        ║
# ║ torchrun spawns $gpn    ║     ║ torchrun spawns $gpn    ║
# ║ GPU workers (0–$((gpn-1))) ║     ║ GPU workers (0–$((gpn-1))) ║
# ╚══════════════════════╝     ╚══════════════════════╝
EOF
    fi
  else
    # one task per GPU
    local top="┌"; local mid1="│"; local mid2="│"; local bot="└"
    for ((i=0;i<gpn;i++)); do
      top+="──────────";  mid1+=" Task $(printf '%-3d' "$i") │";  mid2+=" GPU  $(printf '%-3d' "$i") │"
      [[ $i -lt $((gpn-1)) ]] && top+="┬" || top+="┐"
    done
    bot+=""; for ((i=0;i<gpn;i++)); do bot+="──────────"; [[ $i -lt $((gpn-1)) ]] && bot+="┴"; done; bot+="┘"
    echo "# $top"
    echo "# $mid1"
    echo "# $mid2"
    printf '# │'
    local start=0 end
    for ((i=0;i<gpn;i++)); do end=$((start+cpus-1)); printf ' CPU %d–%-3d│' "$start" "$end"; start=$((end+1)); done
    echo
    echo "# $bot"
  fi
}

# ---- emit file --------------------------------------------------------------
mkdir -p "$JOB_DIR/cfg" "$JOB_DIR/logs"/{slurm,tb}
out="$JOB_DIR/cfg/sbatch-entry.slurm"; tmp="$out.new"

{
  cat <<EOF
#!/usr/bin/env bash
# AUTOGENERATED from cfg/params.sh with command:'
#     cd $PROJHOME
#     export  JOB_NAME=$JOB_NAME
#     base/mammoth-helper/helper/bin/create  mksbatch.sh 
# This is account agnostic (no -A option or #SBATCH --account)
# Instead, put into your .profile:
#     export SBATCH_ACCOUNT="$ACCOUNT"

# ---- sbatch directives ----------------------------------------------------
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=${JOB_PARTITION}
#SBATCH --nodes=${JOB_NODES}
#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}
#SBATCH --cpus-per-task=${JOB_CPUS_PER_TASK}
EOF

  # GPU lines
  echo "#SBATCH --nodes=$JOB_NODES"
  echo "#SBATCH --ntasks-per-node=$NTASKS_PER_NODE"
  echo "#SBATCH --cpus-per-task=$JOB_CPUS_PER_TASK"
  if (( GPN > 0 )); then
      if [[ -n "${GPUS_PER_TASK:-}" ]]; then
	  echo "#SBATCH --gpus-per-task=$GPUS_PER_TASK"
	  if [[ -n "$GPU_TYPE" ]]; then
	      echo "#SBATCH --gres=gpu:$GPU_TYPE:$GPN"
	  else
	      echo "#SBATCH --gres=gpu:$GPN"
	  fi
      else
	  if [[ -n "$GPU_TYPE" ]]; then
	      echo "#SBATCH --gpus-per-node=$GPU_TYPE:$GPN"
	      echo "#SBATCH --gres=gpu:$GPU_TYPE:$GPN"
	      echo "#SBATCH --hint=nomultithread"   # like your LUMI headers
	  else
	      echo "#SBATCH --gpus-per-node=$GPN"
	      echo "#SBATCH --gres=gpu:$GPN"
	  fi
      fi
  fi
  [[ "${JOB_EXCLUSIVE:-0}" = "1" ]] && echo "#SBATCH --exclusive"
  echo "#SBATCH --time=$JOB_TIME"
  echo "#SBATCH --mem=$JOB_MEM"

  cat <<EOF

# ---- guards for stopping accidental heavy jobs-----------------------------
export GUARD_MAX_NODES=4                       # do not change unless you need to
export GUARD_TIME="${GUARD_TIME:-0-01:00:00}"  # do not change unless you need to
# examples: GUARD_TIME="12:00:00" (12h) or GUARD_TIME="2-00:00:00" (2 days).

# ---- job payload (adjust as needed) ---------------------------------------
export JOB_PATTERN=${JOB_PATTERN:-slurm}
EOF
  cat $PARAMS | egrep '^export'
  cat <<EOF
# export JOB_SCRIPT="..."
# export JOB_ARGS="..."
source base/mammoth-helper/helper/bin/slurm/sbatch-tail.sh

# ---- visualization for a quick review "is this sensible?" -----------------
$(viz_block)
# On LUMI: 7 CPUs per GPU → 4 GPUs use 28 CPUs; the other 4 GPUs can still be scheduled.
# If you want to own the node, export JOB_EXCLUSIVE=1 (or JOB_CPU_SPLIT_POLICY=used) and take more.
EOF
} > "$tmp"

# ---- (Optional) One-line sanity to stderr -----------------------------------
printf '[mk] system=%s usable=%d cores, policy=%s, gpn=%d -> cpt=%d, mem=%s\n' \
  "$JOB_SYSTEM" "$usable_cores" "$JOB_CPU_SPLIT_POLICY" "$GPN" "$JOB_CPUS_PER_TASK" "$JOB_MEM" >&2

if [[ ! -f "$out" || "${OVERWRITE:-0}" = "1" ]]; then
    mv -f "$tmp" "$out"; chmod +x "$out"
    echo "[mk] Wrote $out"
else
    echo "[mk] Kept existing: $out"
    echo "[mk] New version  : $tmp"
fi
