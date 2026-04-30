#!/bin/bash

#SBATCH -A project_2017852
#SBATCH -J train
#SBATCH -o ./log/training.%j.out
#SBATCH -e ./log/training.%j.err
#SBATCH --partition=gpupilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:gh200:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e

module purge
module load pytorch
export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia
/scratch/project_2017852/mammoth-shared/.venv/bin/python /scratch/2017852/mammoth-shared/mammoth/train.py -config train.yaml

echo "Finishing at `date`"
