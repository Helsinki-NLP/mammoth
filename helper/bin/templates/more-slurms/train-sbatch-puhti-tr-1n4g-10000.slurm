#!/usr/bin/env bash
# mammoth-puhti-tr-1n4g-10000.slurm
#SBATCH --job-name=mammoth-puhti-tr-1n4g-10000
#SBATCH --account="$ACCOUNT"
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40             # whole CPU budget on Puhti GPU node
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --mem=180G
# Puhti torchrun: 1 task/node; torchrun spawns 4 GPU workers
export PATTERN=torchrun
source "$PROJHOME/bin/sbatch-tail.sh"
