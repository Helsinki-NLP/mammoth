#!/usr/bin/env bash
# train-sbatch-puhti-2n4g-10m.slurm
#SBATCH --job-name=mammoth-puhti-2n4g-10m
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=gpu                 # multi-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Two-node, 4 GPUs/node; IB fabric
source "$PROJHOME/bin/sbatch-tail.sh"
