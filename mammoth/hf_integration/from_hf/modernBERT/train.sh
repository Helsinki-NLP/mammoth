#!/bin/bash

#SBATCH -A project_462000964
#SBATCH -J train
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e


singularity exec \
    -B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
    -B /scratch/project_462000964/shared:/scratch/project_462000964/shared:rw \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
    /scratch/project_462000964/members/wangchao/.venv/bin/python /scratch/project_462000964/members/wangchao/mammoth/train.py \
    -config train.yaml

echo "Finishing at `date`"
