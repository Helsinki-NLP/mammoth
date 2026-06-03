#!/bin/bash
#SBATCH --job-name=inference-docmt4
#SBATCH --account=project_462000964
#SBATCH --partition=dev-g
#SBATCH --mem-per-gpu=16G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --output=inf_logs/job%j.out
#SBATCH --error=inf_logs/job%j.err

# Given on the command line:
##SBATCH --time=00:60:00
##SBATCH --ntasks=1
##SBATCH --gpus=1

# inf_make_template.sh
#
# Purpose
# -------
# Top-level Slurm batch entry script for Mammoth evaluation.
#
# This script is intended to be copied or adapted for a specific model. It:
#   1) defines the model- and environment-specific paths,
#   2) prepares output directories,
#   3) runs the inference planner,
#   4) either stops after planning (outside Slurm),
#      or launches distributed inference (inside Slurm).
#
# In other words, this script ties together the complete shell-based
# evaluation workflow:
#
#   inf_make_template.sh
#       -> inf_plan.sh
#       -> inf_distr.sh   (planning mode outside Slurm)
#       -> inf_wrapper.sh (execution mode inside Slurm)
#       -> inf_distr.sh   (runtime distribution inside Slurm)
#
# Typical use
# -----------
# Planning only (outside Slurm):
#   bash inf_make_template.sh
#
# Fresh planning:
#   bash inf_make_template.sh --fresh
#
# Inside Slurm:
#   sbatch [resource options] inf_make_template.sh
#
# Environment and configuration
# -----------------------------
# This script exports the key path variables needed by the lower-level tools:
#   MAMMOTH
#   BINDIR
#   MAKESCRIPT
#   DATADIR
#   MODEL
#   TRAINCONFIG
#   OUTDIR
#   LOGDIR
#   SCRDIR
#
# It also validates that the expected helper scripts exist in BINDIR.
#
# Reads
# -----
#   - benchmark data under DATADIR
#   - training config under TRAINCONFIG
#   - helper scripts under BINDIR
#
# Writes
# ------
#   - OUTDIR/calls.out
#   - OUTDIR/calls.sacre.out
#   - OUTDIR/calls.comet.out
#   - OUTDIR/*.yaml
#   - hypothesis files under OUTDIR
#   - logs under LOGDIR
#   - metric outputs under SCRDIR
#
# Notes
# -----
# - `--fresh` removes OUTDIR/calls.out before planning.
# - Outside Slurm, the script only plans and prints a suggested allocation.
# - Inside Slurm, it proceeds to `srun singularity exec ... inf_wrapper.sh`.


set -euo pipefail

export MAMMOTH=/scratch/project_462000964/shared/mammoth-shared/mammoth-dev/mammoth
export BINDIR=/scratch/project_462000964/shared/testing-shared/bin
export MAKESCRIPT="$BINDIR/$(basename "$0")"
export DATADIR=/scratch/project_462000964/shared/testing-shared/data

MODELS=/scratch/project_xxxxxxxxxxx/members/jaedoe/models
export MODEL=$MODELS/docmt-4pivots/mammoth/  # end with /
export TRAINCONFIG="${MODEL}train.yaml"
export OUTDIR="${MODEL}inf_out"
export LOGDIR="${MODEL}inf_logs"
export SCRDIR="${MODEL}inf_scores"
mkdir -p $SCRDIR $OUTDIR $LOGDIR

: "${BINDIR:?BINDIR must be set}"
: "${LOGDIR:?LOGDIR must be set}"
: "${SCRDIR:?SCRDIR must be set}"
[[ -d "$BINDIR"                        ]] || { echo "Error: BINDIR must be an existing directory: $BINDIR" >&2; exit 1; }
[[ -d "$LOGDIR"                        ]] || { echo "Error: LOGDIR must be an existing directory: $LOGDIR" >&2; exit 1; }
[[ -d "$SCRDIR"                        ]] || { echo "Error: SCRDIR must be an existing directory: $SCRDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_make.sh"            ]] || { echo "Error: The template SLURM script must be in BINDIR: $BINDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_plan.sh"            ]] || { echo "Error: The planner must be in BINDIR: $BINDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_extract_yaml.py"    ]] || { echo "Error: The inf_extract_yaml.py must be in BINDIR: $BINDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_wrapper.sh"         ]] || { echo "Error: The wrapper must be in BINDIR: $BINDIR" >&2; exit 1; }
[[ -f "$BINDIR/inf_distr.sh"           ]] || { echo "Error: The distributor must be in BINDIR: $BINDIR" >&2; exit 1; }

# Option --fresh removes $OUTDIR/calls.out file.
fresh=0
args=()
for arg in "$@"; do
    case "$arg" in
        --fresh) fresh=1 ;;
        *) args+=("$arg") ;;
    esac
done
if [ "$fresh" -eq 1 ]; then
    echo "Removing the file    $OUTDIR/calls.out   since --fresh"
    rm -f $OUTDIR/calls.out
fi
set -- "${args[@]}"
bash $BINDIR/inf_plan.sh   # checks the variables and creates calls.out if not yet

if [ -n "${SLURM_JOBID:-}" ]; then
    echo "Proceed to srun..."
else
    echo "Just planning..."
    bash $BINDIR/inf_distr.sh   # plans the SLURM allocations
    exit 0
fi

echo "Launching srun..."
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3  # I guess not really needed in inference
export NCCL_NET_GDR_LEVEL=PHB                  # I guess not really needed in inference
#      --env LD_PRELOAD="/usr/lib/libfabric.so.1 /opt/rocm/lib/librccl.so.1" \
#      This causes a freeing error, possibly due to some build incompability
srun /usr/bin/singularity exec \
     --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
     --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" \
     --env NCCL_DEBUG="INFO" \
     -B /scratch/project_462000964:/scratch/project_462000964:rw \
     /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif \
     $BINDIR/inf_wrapper.sh

