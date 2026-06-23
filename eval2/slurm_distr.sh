#!/usr/bin/env bash
#
# inf_distr.sh
#
# Purpose
# -------
# Distribute inference commands across Slurm ranks, or propose a Slurm allocation
# when run outside Slurm.
#
# This script has two modes:
#
#   1) Outside Slurm:
#      - reads the planned inference commands from CALLS_FILE,
#      - counts them,
#      - chooses a recommended LUMI allocation from an internal lookup table,
#      - prints the suggested `sbatch` command.
#
#   2) Inside Slurm:
#      - reads the same CALLS_FILE,
#      - determines the current rank and world size from Slurm,
#      - assigns to each rank every S-th command starting from its rank index,
#      - executes the assigned commands sequentially.
#
# The script is agnostic about the command contents. Each line in CALLS_FILE
# is treated as one executable shell command, and commands are assumed to have
# roughly similar runtime.
#
# Typical use
# -----------
# Outside Slurm:
#   bash inf_distr.sh
#
# Inside Slurm:
#   bash inf_distr.sh
#
# In practice, the inside-Slurm mode is usually invoked indirectly by
# `inf_wrapper.sh`.
#
# Environment
# -----------
# Input command file:
#   CALLS_FILE
# Planning mode defaults:
#   MAKESCRIPT
#       Optional but useful in planning mode; used when printing the suggested
#       `sbatch` command.
#
# Slurm runtime variables (inside Slurm):
#   SLURM_JOB_ID
#   SLURM_PROCID
#   SLURM_NTASKS
#   SLURM_LOCALID
#
# Reads
# -----
#   - CALLS_FILE (default: $OUTDIR/calls.out)
#
# Writes / effects
# ----------------
# Outside Slurm:
#   - prints a suggested allocation and `sbatch` command
#
# Inside Slurm:
#   - prints rank assignment information
#   - executes assigned commands
#
# Output
# ------
# No files are written directly by this script.
# In runtime mode, side effects come from the commands read from CALLS_FILE.

if [ -z "${BASH_VERSION:-}" ] || set -o | grep -q '^posix[[:space:]]*on'; then
    echo "Error: run this script with bash, not sh." >&2
    exit 1
fi

set -euo pipefail

: "${MAKESCRIPT:?MAKESCRIPT must be set}"
: "${CALLS_FILE:?CALLS_FILE must be set}"

die() {
    echo "ERROR: $*" >&2
    exit 1
}
have_slurm() {
    [[ -n "${SLURM_JOB_ID:-}" ]]
}
load_calls() {
    [[ -f "$CALLS_FILE" ]] || die "Missing calls file: $CALLS_FILE"
    echo "Loading $CALLS_FILE"
    echo -n "Supervised pair inferences: "
    grep -F ".hyp" $CALLS_FILE | wc -l || true
    echo -n "Zeroshot pair   inferences: "
    grep -F ".0shyp" $CALLS_FILE | wc -l || true
    echo "---"
    mapfile -t CALLS < <(grep '^python' "$CALLS_FILE" || true)
    NCALLS="${#CALLS[@]}"
    (( NCALLS > 0 )) || die "No python calls found in $CALLS_FILE"
}


# min max nodes gpus_per_node
DEVG_PLAN_TABLE=(
  "1    13    1   1"
  "14   21    1   2"
  "22   28    1   3"
  "29   36    1   4"
  "37   44    1   5"
  "45   52    1   6"
  "53   60    1   7"
  "61   64    1   8"
  "65   80    1   5"
  "81   96    1   6"
  "97   112   1   7"
  "113  128   1   8"
  "129  256   2   8"
  "257  512   4   8"
  "513  1024  8   8"
  "1025 2048  16  8"
  "2049 4096  32  8"
)

choose_devg_plan_from_table_devg() {
    local mins_per_wave=4
    local row min max nodes gpn
    for row in "${DEVG_PLAN_TABLE[@]}"; do
        read -r min max nodes gpn <<< "$row"
        if (( NCALLS >= min && NCALLS <= max )); then
            CHOSEN_NODES="$nodes"
            CHOSEN_GPUS_PER_NODE="$gpn"
            CHOSEN_WORLD=$(( nodes * gpn ))
            CHOSEN_BATCHES=$(( (NCALLS + CHOSEN_WORLD - 1) / CHOSEN_WORLD ))
            CHOSEN_RUNTIME_MIN=$(( CHOSEN_BATCHES * mins_per_wave ))
            CHOSEN_COST=$(( CHOSEN_WORLD * CHOSEN_BATCHES ))
            CHOSEN_WASTE=$(( CHOSEN_COST - NCALLS ))
            CHOSEN_PARTITION="dev-g"
            printf -v CHOSEN_TIME '%02d:%02d:00' \
                $(( CHOSEN_RUNTIME_MIN / 60 )) \
                $(( CHOSEN_RUNTIME_MIN % 60 ))
            return 0
        fi
    done
    echo "No dev-g lookup-table plan for NCALLS=$NCALLS" >&2
    return 1
}

# small-g: up to 4 nodes, up to 8 GPUs per node on LUMI-G
# rows: min_calls max_calls nodes gpus_per_node
SMALLG_PLAN_TABLE=(
  "1      64      1   1"
  "65     128     1   2"
  "129    192     1   3"
  "193    256     1   4"
  "257    320     1   5"
  "321    384     1   6"
  "385    448     1   7"
  "449    512     1   8"
  "513   1024     2   8"
  "1025  999999   4   8"
)

choose_smallg_plan_from_table() {
    local mins_per_wave="${1:-4}"
    local row min max nodes gpn

    for row in "${SMALLG_PLAN_TABLE[@]}"; do
        read -r min max nodes gpn <<< "$row"
        if (( NCALLS >= min && NCALLS <= max )); then
            CHOSEN_NODES="$nodes"
            CHOSEN_GPUS_PER_NODE="$gpn"
            CHOSEN_WORLD=$(( nodes * gpn ))
            CHOSEN_BATCHES=$(( (NCALLS + CHOSEN_WORLD - 1) / CHOSEN_WORLD ))
            CHOSEN_RUNTIME_MIN=$(( CHOSEN_BATCHES * mins_per_wave ))
            CHOSEN_COST=$(( CHOSEN_WORLD * CHOSEN_BATCHES ))
            CHOSEN_WASTE=$(( CHOSEN_COST - NCALLS ))
            CHOSEN_PARTITION="small-g"

            # small-g walltime limit is 3 days = 4320 minutes
            if (( CHOSEN_RUNTIME_MIN > 4320 )); then
                echo "small-g plan would exceed 3-day walltime for NCALLS=$NCALLS" >&2
                return 1
            fi

            printf -v CHOSEN_TIME '%02d:%02d:00' \
                $(( CHOSEN_RUNTIME_MIN / 60 )) \
                $(( CHOSEN_RUNTIME_MIN % 60 ))

            return 0
        fi
    done

    echo "No small-g lookup-table plan for NCALLS=$NCALLS" >&2
    return 1
}

plan_outside_slurm() {
    choose_smallg_plan_from_table
    
    echo "Outside Slurm."
    echo "Suggested LUMI allocation:"
    echo "  calls             : $NCALLS"
    echo "  nodes             : $CHOSEN_NODES"
    echo "  GPUs per node     : $CHOSEN_GPUS_PER_NODE"
    echo "  world size        : $CHOSEN_WORLD"
    echo "  waves             : $CHOSEN_BATCHES"
    echo "  expected runtime  : $CHOSEN_TIME"
    echo "  wasted slots      : $CHOSEN_WASTE"
    echo "  partition         : $CHOSEN_PARTITION"
    echo
    echo "sbatch --parsable --partition=$CHOSEN_PARTITION --time=$CHOSEN_TIME --nodes=$CHOSEN_NODES --ntasks=$CHOSEN_WORLD $MAKESCRIPT"
    echo "sbatch --parsable --partition=$CHOSEN_PARTITION --time=$CHOSEN_TIME --nodes=$CHOSEN_NODES --ntasks=$CHOSEN_WORLD $MAKESCRIPT" > "${MAKESCRIPT}"
}


# SLURM-INTERNAL:
assign_rank_calls_for_show() {
    local rank="$1"
    local world="$2"
    local i
    local n=1
    for ((i=rank; i<NCALLS; i+=world)); do
        printf '[rank %d] %d: %s\n' "$rank" "$n" "${CALLS[$i]}"
        ((n++))
    done
}
assign_rank_calls() {
    local rank="$1"
    local world="$2"
    local i
    for ((i=rank; i<NCALLS; i+=world)); do
        printf '%s\n' "${CALLS[$i]}"
    done
    # feed only this rank’s assigned commands into the while loop
}
run_inside_slurm() {
    local rank="${SLURM_PROCID:-0}"
    local world="${SLURM_NTASKS:-1}"
    local local_rank="${SLURM_LOCALID:-0}"
    echo "Inside Slurm: rank=$rank world=$world (local_rank=$local_rank) calls=$NCALLS"    
    echo "This rank will process:"
    assign_rank_calls_for_show "$rank" "$world" 

    # the following implements the calls
    while IFS= read -r cmd; do
        [[ -n "$cmd" ]] || continue    # skip empty lines
	date
        echo "[rank $rank] $cmd"       # print the command before running it
        eval "$cmd"                    # executes the call
    done < <(assign_rank_calls "$rank" "$world")
}
main() {
    load_calls
    if have_slurm; then
        run_inside_slurm
    else
        plan_outside_slurm
    fi
}

main "$@"
