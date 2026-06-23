#!/bin/bash

#SBATCH -A project_xxxx 
#SBATCH -J training
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e


singularity exec \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" \
    -B /scratch/project_462000964/members:/scratch/project_462000964/members:rw \
    -B /scratch/project_462000964/shared:/scratch/project_462000964/shared:ro \
    /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif \
    /scratch/project_462000964/members/wangchao/.venv/bin/python /scratch/project_462000964/members/wangchao/mammoth/train.py \
    -config single_node_train.yaml

echo "Finishing at `date`"

