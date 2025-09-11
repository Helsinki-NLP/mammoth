#!/usr/bin/env bash
# mammoth-lumi-tr-1n4g-01000.slurm
#SBATCH --job-name=mammoth-lumi-tr-1n4g-01000
#SBATCH --account="$ACCOUNT"
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56             # give torchrun the whole CPU budget per node
#SBATCH --gpus-per-node=4              # torchrun will spawn 4 workers
#SBATCH --time=00:10:00
#SBATCH --mem=240G
# LUMI torchrun: 1 task/node; torchrun spawns 4 GPU workers
# ┌──────────────────────────┐
# │ Task 0                   │ → torchrun → 4 procs → GPUs 0–3
# │ CPU 0–55                 │
# └──────────────────────────┘
export PATTERN=torchrun
source "$PROJHOME/bin/sbatch-tail.sh"
