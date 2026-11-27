#!/bin/bash
# Run any command (no GPU) as a batch job, with the PyTorch module loaded and your venv active.
# Usage:   sbatch cpu_job.sh <command> [args...]
# Example: sbatch cpu_job.sh python prepare_parallel.py --src ... --tgt ...
#          sbatch --time=03:00:00 --mem=64G cpu_job.sh bash prepare_all.sh
# Options given to sbatch on the command line override the #SBATCH lines below.

#SBATCH -A project_xxxxxxx
#SBATCH -J cpu_job
#SBATCH -o ./log/%j.out
#SBATCH -e ./log/%j.err
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00

# ==== EDIT THIS (same as in env_setup.sh) ====
VENV=/scratch/project_xxxxxxx/$USER/mammoth-venv
# =============================================

echo "Starting at `date`"
echo "Command: $*"
set -e

module purge
module load python-pytorch
source "$VENV/bin/activate"
"$@"

echo "Finishing at `date`"
