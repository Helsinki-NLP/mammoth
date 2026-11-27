#!/bin/bash
# Run any command (no GPU) as a batch job, inside the container with your venv active.
# Usage:   sbatch cpu_job.sh <command> [args...]
# Example: sbatch cpu_job.sh python prepare_parallel.py --src ... --tgt ...
#          sbatch --time=03:00:00 --mem=64G cpu_job.sh bash prepare_all.sh
# Options given to sbatch on the command line override the #SBATCH lines below.

#SBATCH -A project_xxxxxxxxx
#SBATCH -J cpu_job
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# ==== EDIT THESE (same as in env_setup.sh) ====
PROJECT_DIR=/scratch/project_xxxxxxxxx
VENV=$PROJECT_DIR/$USER/mammoth-venv
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
# ==============================================

echo "Starting at `date`"
echo "Command: $*"
set -e

singularity exec \
    -B "$PROJECT_DIR:$PROJECT_DIR:rw" \
    "$CONTAINER" \
    bash -c 'source "$0/bin/activate" && "$@"' "$VENV" "$@"

echo "Finishing at `date`"
