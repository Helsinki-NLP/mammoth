#!/usr/bin/env bash
# train-sbatch-lumi-1n4g-1h.slurm
#SBATCH --job-name=mammoth-lumi-1n4g-1h
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=small-g            # ≤4 nodes often use small-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:mi250:4
#SBATCH --time=01:00:00
#SBATCH --mem=240G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "$PROJHOME/bin/sbatch-tail.sh"
