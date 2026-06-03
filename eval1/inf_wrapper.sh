#!/bin/bash
# inf_wrapper.sh
#
# Purpose
# -------
# Run a distributed batch of inference commands inside an existing Slurm job.
#
# This script is a lightweight task wrapper. It assumes that:
#   1) a Slurm allocation already exists,
#   2) the current process runs on a GPU node,
#   3) the command file `calls.out` has already been prepared,
#   4) command distribution across ranks is delegated to `inf_distr.sh`.
#
# The wrapper itself does not decide which commands to run. Instead, it:
#   - checks that it is running under Slurm,
#   - validates that required helper scripts exist in BINDIR,
#   - activates the shared Python virtual environment,
#   - invokes `inf_distr.sh`, which assigns each rank its share of commands.
#
# Typical use
# -----------
# This script is not normally called directly by the user.
# It is intended to be executed from a Slurm batch script, typically under:
#
#   srun singularity exec ... $BINDIR/inf_wrapper.sh
#
# Environment
# -----------
# Required:
#   BINDIR
#       Directory containing this script and `inf_distr.sh`.
#
# Required indirectly / expected from the surrounding Slurm job:
#   SLURM_JOBID or SLURM_JOB_ID
#       Must be set; otherwise the wrapper aborts.
#
# Expected from surrounding workflow:
#   OUTDIR
#       Used indirectly by `inf_distr.sh` when locating `calls.out`.
#
# Reads
# -----
#   - $BINDIR/inf_wrapper.sh
#   - $BINDIR/inf_distr.sh
#   - shared venv activation script
#
# Writes / effects
# ----------------
#   - writes progress messages to stdout/stderr
#   - executes a rank-specific subset of commands from `calls.out`
#
# Output
# ------
# The script itself does not produce files directly.
# The executed commands typically create hypothesis files and logs.


if [ -n "${SLURM_JOBID:-}" ]; then
    echo "Running inside Slurm job: $SLURM_JOBID"
else
    echo "Error: Not running inside a Slurm job. This is a task wrapper."
    exit 1
fi
: "${BINDIR:?BINDIR must be set}"
[[ -d "$BINDIR" ]]                || { echo "Error: BINDIR must be an existing directory: $BINDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_wrapper.sh" ]] || { echo "Error: The current script must be in BINDIR: $BINDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_distr.sh"   ]] || { echo "Error: The distributor must be in BINDIR: $BINDIR" >&2; exit 1; }

job_start_ts=$(date +%s)
echo "starting inf_wrapper.sh at ${job_start_ts}" 

ACTIVATE=/scratch/project_462000964/shared/mammoth-shared/.venv/bin/activate
[ -f "$ACTIVATE"   ] || { echo "Missing activate script: $ACTIVATE" >&2; exit 1; }
source "$ACTIVATE"

############################################################################################
bash $BINDIR/inf_distr.sh        # This executes a proportionate share of calls.out commands
############################################################################################

job_end_ts=$(date +%s)
echo "finishing inf_wrapper.sh at ${job_end_ts}" 
