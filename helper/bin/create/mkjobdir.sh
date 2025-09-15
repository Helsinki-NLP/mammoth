#!/usr/bin/bash
set -euo pipefail
prog=${0##*/}
if (( $# < 1 )); then
  printf '❌ expecting a job name argument\n' >&2
  printf 'Usage Example: %s train-L-1n1g10m-{en,es}-test\n' "$prog" >&2
  exit 2
fi
: "${PROJHOME:?❌ PROJHOME not set}"
JOB_NAME=$1
echo "making a job directory under \$PROJHOME/data/"


# set -euo pipefail
: "${PROJHOME:?❌ PROJHOME not set}"      || exit 1
: "${1:?❌ expecting a jobname argument}" 
JOB_NAME="$1"

validate_job_name() {
  local re='^([a-z][a-z0-9_]*)-([LPRM])-?([0-9]+)n([0-9]+)g([0-9]+[mhd])-(\[[a-z,]*\])(?:-([A-Za-z0-9._-]+))?'
  [[ $JOB_NAME =~ $re ]]
}
# Expect: Task-SystemNodesGpusTime-Lang[-Spec]
diagnose_job_name() {
  local name="$1"
  local -i bad=0

  # Split into parts
  local parts
  IFS='-' read -r -a parts <<<"$name"
  if (( ${#parts[@]} < 5 || ${#parts[@]} > 6 )); then
    printf '❌ %s\n' "Expected 5 or 6 dash-separated fields (got ${#parts[@]}): $name" >&2
    printf '   Format: Task-SystemNodesGpusTime-Lang[-Spec]\n' >&2
    return 1
  fi

  local task="${parts[0]}"
  local system="${parts[1]}"
  local ngt="${parts[2]}"    # NodesGpusTime e.g. 1n1g10m
  local lang="${parts[3]}"
  local spec="${parts[4]:-}"; (( ${#parts[@]} == 6 )) && spec="${parts[5]}"

  # Field-by-field checks with actionable messages
  if [[ ! $task =~ ^([a-z][a-z0-9_]*)$ ]]; then
      printf '• Task invalid: %-20s  (expected: [a-z][a-z0-9_]*)\n' "'$task'" >&2; bad=1
  else
      JOB_TASK=$task
      #echo $JOB_TASK
  fi
  
  if [[ ! $system =~ ^[LPRM]$ ]]; then
      printf '• System invalid: %-18s  (expected: one of L P R M)\n' "'$system'" >&2; bad=1
  else
      case "$system" in
	  L)      JOB_SYSTEM="lumi"  ;;
	  R)      JOB_SYSTEM="roihu" ;;
	  P)      JOB_SYSTEM="puhti" ;;
	  M)      JOB_SYSTEM="mahti" ;;
      esac
  fi
  
  if [[ ! $ngt =~ ^([0-9]+)n([0-9]+)g([0-9]+)([smhd])$ ]]; then
      printf '• NodesGpusTime invalid: %-9s  (expected: <N>n<G>g<T><unit>, e.g. 1n1g10m, 2n8g1h)\n' "'$ngt'" >&2
      printf '  Hints: number of nodes -> n, GPUs -> g, time unit one of s/m/h/d\n' >&2
      bad=1
  else
      JOB_NODES="${BASH_REMATCH[1]}"        # e.g. 1
      JOB_GPUS="${BASH_REMATCH[2]}"         # e.g. 1
      N="${BASH_REMATCH[3]}"         # e.g. 10
      U="${BASH_REMATCH[4]}"         # e.g. m/h/d

      # Convert to total seconds
      local sec
      case "$U" in
	  s) sec=$((N)) ;;
	  m) sec=$((N*60)) ;;
	  h) sec=$((N*3600)) ;;
	  d) sec=$((N*86400)) ;;
      esac

      # Format as D-HH:MM:SS (omit D- when zero days)
      local d=$((sec/86400)); local r=$((sec%86400))
      local h=$((r/3600));    r=$((r%3600))
      local m=$((r/60));      local s=$((r%60))
      
      printf -v JOB_TIME '%d-%02d:%02d:%02d' "$d" "$h" "$m" "$s"
      #echo $JOB_NODES
      #echo $JOB_GPUS
      #echo $JOB_TIME
  fi
  
  if [[ ! $lang =~ ^[a-z:]*$ ]]; then
      printf '• Lang invalid: %-19s  (expected: lowercase, or comma, e.g., en, en:es)\n' "'$lang'" >&2; bad=1
  else
      JOB_LANG=$lang
      #echo $JOB_LANG
  fi
  
  if (( ${#parts[@]} < 5 )); then
      JOB_SPEC=""
  else if [[ ! $spec =~ ^[A-Za-z0-9._-]+$ ]]; then
	   printf '• Spec invalid: %-19s  (allowed chars: A–Z a–z 0–9 . _ -)\n' "'$spec'" >&2; bad=1
       else
	   JOB_SPEC=$spec
	   #echo $JOB_SPEC
       fi
  fi

  if (( bad )); then
    printf '— Example: train-L-1n1g10m-en:es-test\n' >&2
    return 1
  fi
}

if diagnose_job_name "$JOB_NAME"; then
    echo =================================
    echo "JOB_NAME   : $JOB_NAME"
    echo "JOB_TASK   : $JOB_TASK"
    echo "JOB_SYSTEM : $JOB_SYSTEM"
    echo "JOB_NODES  : $JOB_NODES"
    echo "JOB_GPUS   : $JOB_GPUS"
    echo "JOB_TIME   : $JOB_TIME"
    echo "JOB_LANG   : $JOB_LANG"
    echo "JOB_SPEC   : $JOB_SPEC"
    echo =================================
else
  echo "❌ Invalid JOB_NAME. Expected: Task-SystemNodesGpusTime-Lang[-Spec]" >&2
  echo "   Example: train-L-1n1g10m-en:es-test" >&2
  # Task  : train, convert, diag, trans, ...
  # System: C=L(umi), P(uhti), R(oihu), M(ahti)
  # Nodes : -1n -2n 16n etc
  # Gpus  : 1g 2g 4g 8g 1g6 ect
  # Time  : 10m 1h ...
  # Lang  : en:es
  # Spec  : free specifiers
  exit 1
fi
export JOB_NAME
export JOB_DIR=$PROJHOME/data/$JOB_NAME

echo "export JOB_NAME=$JOB_NAME"     > $JOB_DIR/cfg/params.sh
echo "export JOB_TASK=$JOB_TASK"    >> $JOB_DIR/cfg/params.sh
echo "export JOB_SYSTEM=$JOB_SYSTEM">> $JOB_DIR/cfg/params.sh
echo "export JOB_NODES=$JOB_NODES"  >> $JOB_DIR/cfg/params.sh
echo "export JOB_GPUS=$JOB_GPUS"    >> $JOB_DIR/cfg/params.sh
echo "export JOB_TIME=$JOB_TIME"    >> $JOB_DIR/cfg/params.sh
echo "export JOB_LANG=$JOB_LANG"    >> $JOB_DIR/cfg/params.sh
echo "export JOB_SPEC=$JOB_SPEC"    >> $JOB_DIR/cfg/params.sh

BASE_SRC="${BASE_SRC:-$PROJHOME/base}"

log(){ printf '[job:%s] %s\n' "$JOB_NAME" "$*"; }

# --- helpers ---------------------------------------------------------------
ensure_dir(){ [[ -d "$1" ]] || { mkdir -p "$1"; log "mkdir -p $1"; }; }

# Create link if missing; leave existing files/links as-is (no overwrite)
ensure_symlink(){
  local target="$1" link="$2"
  if [[ -L "$link" ]]; then
    # already a symlink; change nothing (idempotent)
    return 0
  elif [[ -e "$link" ]]; then
    log "keep existing (not a symlink): $link"
    return 0
  else
    ln -s "$target" "$link"
    log "ln -s $target -> $link"
  fi
}

# Copy only if destination doesn't exist
ensure_copy(){
  local src="$1" dest="$2"
  [[ -e "$dest" ]] && return 0
  install -D -m 0644 "$src" "$dest"
  log "copied $(basename "$src") -> $dest"
}

# --- make / complement structure ------------------------------------------
ensure_dir "$JOB_DIR"

# base symlink inside the job dir (no overwrite)
if [[ -d "$BASE_SRC" ]]; then
  ensure_symlink "$BASE_SRC" "$JOB_DIR/base"
else
  log "WARN: BASE_SRC not found: $BASE_SRC"
fi

# directories (idempotent)
for d in \
  "$JOB_DIR/cfg" \
  "$JOB_DIR/in" "$JOB_DIR/in/data" "$JOB_DIR/in/models" "$JOB_DIR/in/vocab" \
  "$JOB_DIR/logs" "$JOB_DIR/logs/slurm" "$JOB_DIR/logs/tb" \
  "$JOB_DIR/out" "$JOB_DIR/out/checkpoints" "$JOB_DIR/out/metrics" "$JOB_DIR/out/models" "$JOB_DIR/out/translations"
do
  ensure_dir "$d"
done

# template file into cfg/ (only if missing)
ensure_copy "$BASE_SRC/mammoth-helper/helper/bin/slurm/sbatch-entry.slurm" \
            "$JOB_DIR/cfg/sbatch-entry.slurm"

module load LUMI
module load systools

cd $PROJHOME/data/
tree $JOB_NAME
echo =================================

export JOB_DIR
echo $JOB_DIR
