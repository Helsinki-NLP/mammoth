#!/bin/bash -l

# met_make.sh
#
# Purpose
# -------
# Top-level shell/Slurm entry script for running MT evaluation metrics
# on already generated hypothesis files.
#
# This script is responsible for the metrics phase of the shell-based
# evaluation workflow. It:
#
#   1) ensures that a Python virtual environment with sacrebleu exists,
#   2) ensures that the metric command file `calls.sacre.out` exists,
#      generating it through the inference-planning workflow if needed,
#   3) prevents concurrent resubmission of the same metrics job,
#   4) submits itself as a Slurm job when run outside Slurm,
#   5) executes the metric commands under `srun` when run inside Slurm.
#
# Typical use
# -----------
# Outside Slurm:
#   bash met_make.sh
#
# Inside Slurm:
#   sbatch met_make.sh
#
# The normal user-facing mode is to run it outside Slurm; the script then
# submits itself as a batch job if metric commands are available.
#
# Main inputs
# -----------
# Installation-level:
#   TESTINGDIR
#       Root directory of shared evaluation tools and virtual environments.
#
# Model-level:
#   MODELDIR
#       Model directory containing:
#         - inf_make.sh
#         - inf_out/calls.sacre.out
#         - hypothesis files
#
# Reads
# -----
#   - MODELDIR/inf_make.sh
#   - MODELDIR/inf_out/calls.sacre.out
#   - hypothesis files referenced by calls.sacre.out
#   - TESTINGDIR/inf_make.sh (indirectly, when regenerating calls.sacre.out)
#
# Writes
# ------
#   - TESTINGDIR/venvs/sacre/         Python virtual environment for sacrebleu
#   - MODELDIR/inf_out/calls.sacre.out
#   - score files referenced by calls.sacre.out
#   - submission flag files:
#       met_make.sh.submitted
#       inf_make.sh.submitted
#
# Output
# ------
#   - status and progress messages to stdout/stderr
#   - metric result files (.sacre)
#
# Notes
# -----
# - Outside Slurm, the script acts as a launcher and planner.
# - Inside Slurm, the script acts as the batch payload and executes the
#   commands in calls.sacre.out.
# - The script uses a lock/flag file to prevent duplicate metric jobs.

#SBATCH --output=sacrebleu-%j.out
#SBATCH --error=sacrebleu-%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --account=project_462000964

set -euo pipefail

# Set at installation 
TESTINGDIR=/scratch/project_462000964/shared/testing-shared
# Set for every model directory
MODELDIR=/scratch/project_462000964/members/tiedeman/MARMoT/sandbox/tiedeman/oellm-lg/mammoth-mt/

############################## THIS FILE #####################################
submit_dir="${SLURM_SUBMIT_DIR%/}"
model_dir="${MODELDIR%/}"
if [ "$submit_dir" != "$model_dir" ]; then
    echo "Submit directory mismatch:" >&2
    echo "  SLURM_SUBMIT_DIR=$submit_dir" >&2
    echo "  MODELDIR=$model_dir" >&2
    exit 1
fi
expected_jobname="met_make.sh"
if [ "${SLURM_JOB_NAME:-}" != "$expected_jobname" ]; then
    echo "Expected job name $expected_jobname but was given ${SLURM_JOB_NAME:-<unset>}" >&2
    exit 1
fi
[ -d "$MODELDIR"   ] || { echo "Missing directory: $MODELDIR" >&2; exit 1; }
SCRIPT="$MODELDIR/met_make.sh"
[ -f "$SCRIPT"     ] || { echo "Missing script: $SCRIPT" >&2; exit 1; }
INFMAKE="$MODELDIR/inf_make.sh"
[ -f "$INFMAKE"    ] || { echo "Missing script: $INFMAKE" >&2; exit 1; }
############################ DO NOT TOUCH #####################################
FLAG="${SCRIPT}.submitted"
if [ -e "$FLAG" ]; then
    oldjob=$(cat "$FLAG")
    if squeue -h -j "$oldjob" | grep -q .; then
        if [ ! -n "${SLURM_JOBID:-}" ]; then
            echo "Please wait. The SLURM job $oldjob is still queued to measure the hypotheses."
        else
            echo "How come resubmiting? SLURM job $oldjob is still queued or running."
        fi
        exit 0
    fi
    echo "Removing stale flag for old job $oldjob"
    rm -f "$FLAG"
fi
INFFLAG="${INFMAKE}.submitted"
INFOUT="$MODELDIR/inf_out"
[ -d "$INFOUT"     ] || { echo "Missing directory: $MODELDIR/inf_out" >&2; exit 1; }
VENV="$TESTINGDIR/venvs/sacre"
[ -d "$TESTINGDIR" ] || { echo "Missing directory: $TESTINGDIR" >&2; exit 1; }

ACTIVATE="$VENV/bin/activate"         # produced by    met_make.sh
OUTFILE="$INFOUT/calls.sacre.out"     # produced by    (inf_make.sh @) met_make.sh
# scores/....sacre                    # produced by    sbatch met_make.sh

module load cray-python

# The following ensures 'calls.sacre.out' and then sbatches this to run its commands
if [ -n "${SLURM_JOBID:-}" ]; then
    if [ -s "$OUTFILE" ]; then
       echo "Proceed to srun..."
       [ -f "$ACTIVATE"   ] || { echo "Missing activate script: $ACTIVATE" >&2; exit 1; }
       source "$ACTIVATE"
       trap 'rm -f "$FLAG"' EXIT
       srun bash "$OUTFILE"
    else
       echo "Did not find any commands in $OUTFILE"    
       echo "This means met_make.sh has completed producing the measurements"
    fi
else
    echo "Producing the script: $OUTFILE ..."
    if [ -f "$INFFLAG" ]; then
       echo "Another instance of this script has locked $OUTFILE"
       echo "(if this claim is not true, remove $INFFLAG)"
       exit 1
    else
       touch "$INFFLAG"
    fi
    bash $TESTINGDIR/inf_make.sh --fresh $OUTFILE
    if [ ! -s "$OUTFILE" ]; then
       echo "Found no sacrebleu commands in $OUTFILE"
       exit 0
    fi
    echo "The number of sacrebleu commands:"
    egrep 'sacrebleu' $OUTFILE | wc

    echo "Running the commands from $OUTFILE ..."
    jobid=$(sbatch --parsable "$SCRIPT") || {
       echo "sbatch failed" >&2
       exit 1
    }      
    echo "$jobid" > "$FLAG"
    echo "Submitted $SCRIPT with the current calls.sacre.out as job $jobid"
    echo "After job $jobid has finished, run met_make.sh again to complete"	
    rm "$INFFLAG"
    exit 0
fi
