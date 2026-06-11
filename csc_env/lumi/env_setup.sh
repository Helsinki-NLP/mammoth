#!/bin/bash
#SBATCH -A project_462000964
#SBATCH -J setup_venv
#SBATCH -o ./logs/setup_venv_%j.out
#SBATCH -e ./logs/setup_venv_%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00

set -e  # stop on any error

echo "Starting setup at $(date)"

# ==== CONFIG ====
PROJECT_PATH=/scratch/project_462000964/shared/mammoth-shared # your project path on HPC
VENV_PATH=$PROJECT_PATH/.venv
REQ_FILE=/scratch/project_462000964/shared/mammoth-shared/mammoth-dev/mammoth/csc_env/lumi/requirements_lumi.txt # path to the dependencies

CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif # path to the container

# ==== DEBUG INFO ====
echo "Project path: $PROJECT_PATH"
echo "Venv path: $VENV_PATH"
echo "Requirements: $REQ_FILE"

# ==== RUN INSIDE CONTAINER ====
singularity exec \
    -B /scratch/project_462000964:/scratch/project_462000964:rw \
    $CONTAINER \
    bash -c "

    set -e

    echo 'Inside container:'
    which python

    # Create venv only if it doesn't exist
    if [ ! -d \"$VENV_PATH\" ]; then
        echo 'Creating virtual environment...'
        python -m venv $VENV_PATH --system-site-packages
    else
        echo 'Venv already exists, skipping creation.'
    fi

    echo 'Upgrading pip...'
    $VENV_PATH/bin/pip install --upgrade pip

    echo 'Installing requirements...'
    $VENV_PATH/bin/pip install -r $REQ_FILE

    echo 'Verifying installation...'
    $VENV_PATH/bin/python -c \"import torch; print('Torch version:', torch.__version__)\"

    echo 'Done inside container.'
"

echo "Finished at $(date)"