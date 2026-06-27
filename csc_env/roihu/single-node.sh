#!/bin/bash

#SBATCH -A project_2017852
#SBATCH -J train
#SBATCH -o ./log/training.%j.out
#SBATCH -e ./log/training.%j.err
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e

module purge
module load python-pytorch/2.10
export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia
/scratch/project_2017852/mammoth-shared/.venv/bin/python /scratch/project_2017852/mammoth-shared/mammoth_pytorch/mammoth/train.py -config single_node.yaml

echo "Finishing at `date`"
