#!/usr/bin/env bash
# mammoth-puhti-tr-2n4g-40000.slurm
#SBATCH --job-name=mammoth-puhti-tr-2n4g-40000
#SBATCH --account="$ACCOUNT"
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --mem=180G
# Puhti torchrun multi-node: 1 task per node; torchrun spawns 4 workers per node
export PATTERN=torchrun
source "$PROJHOME/bin/sbatch-tail.sh"
