#!/usr/bin/env bash
# train-sbatch-puhti-1n4g-10m.slurm
#SBATCH --job-name=mammoth-puhti-1n4g-10m
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10             # 40 / 4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4                   # or gpu:v100:4 to pin type
#SBATCH --time=00:10:00
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Puhti 1 node / 4 GPUs / 10 CPUs per task
# ┌──────────┬──────────┬──────────┬──────────┐
# │ Task 0   │ Task 1   │ Task 2   │ Task 3   │
# │ GPU  0   │ GPU  1   │ GPU  2   │ GPU  3   │
# │ CPU 0–9  │ 10–19    │ 20–29    │ 30–39    │
# └──────────┴──────────┴──────────┴──────────┘
source "$PROJHOME/bin/sbatch-tail.sh"
