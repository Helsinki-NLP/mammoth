#!/bin/bash

#SBATCH -A project_xxxxxxxxx
#SBATCH -J translate
#SBATCH -o ./log/translate.%j.out
#SBATCH -e ./log/translate.%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1

# ==== EDIT THESE (same as in env_setup.sh) ====
PROJECT_DIR=/scratch/project_xxxxxxxxx
VENV=$PROJECT_DIR/$USER/mammoth-venv
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
# ==============================================

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e

singularity exec \
    -B "$PROJECT_DIR:$PROJECT_DIR:rw" \
    "$CONTAINER" \
    "$VENV/bin/mammoth_translate" -config inference.yaml

echo "Finishing at `date`"
