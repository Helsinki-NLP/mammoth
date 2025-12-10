#!/usr/bin/env bash
# train-sbatch-lumi-1n4g-10m.slurm
#SBATCH --job-name=mammoth-lumi-1n4g-10m
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=dev-g              # short sanity queue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16             # 64 / 4
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:mi250:4
#SBATCH --time=00:10:00
#SBATCH --mem=240G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# LUMI 1 node / 4 GPUs / 16 CPUs per task
# ┌──────────┬──────────┬──────────┬──────────┐
# │ Task 0   │ Task 1   │ Task 2   │ Task 3   │
# │ GPU  0   │ GPU  1   │ GPU  2   │ GPU  3   │
# │ CPU 0–15 │ 16–31    │ 32–47    │ 48–63    │
# └──────────┴──────────┴──────────┴──────────┘
source "$PROJHOME/bin/sbatch-tail.sh"
