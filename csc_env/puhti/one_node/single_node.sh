#!/bin/bash

#SBATCH -A project_
#SBATCH -J train
#SBATCH -o ./log/training.%j.out
#SBATCH -e ./log/training.%j.err
#SBATCH --partition=gpu  
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e

# export NCCL_DEBUG=INFO
# export NCCL_NVML_DISABLE=1
/scratch/project_2005099/members/chaowang/.venv/bin/python /scratch/project_2005099/members/chaowang/mammoth/train.py -config /scratch/project_2005099/members/chaowang/training/one_node/train.yaml

echo "Finishing at `date`"
