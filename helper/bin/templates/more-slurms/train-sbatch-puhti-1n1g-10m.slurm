#!/usr/bin/env bash
# train-sbatch-puhti-1n1g-10m.slurm
#SBATCH --job-name=mammoth-puhti-1n1g-10m
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=45G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# ┌──────────┐
# │ Task 0   │ GPU 0 │ CPU 0–9
# └──────────┘
source "$PROJHOME/bin/sbatch-tail.sh"
