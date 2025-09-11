#!/usr/bin/env bash
# train-sbatch-lumi-1n1g-10m.slurm
#SBATCH --job-name=mammoth-lumi-1n1g-1m
#SBATCH --account=project_462000964
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7              # ~1/8 of node
##SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:mi250:1             # aways request typed GPUs
#SBATCH --time=00:60:00
#SBATCH --mem-per-gpu=60G              # choose ONE memory flag
##SBATCH --mem=60G                     # never combine --mem with --mem-per-*
##SBATCH --mem-per-cpu=8G              # 7 CPUs → 56G for the task
##SBATCH --mem=0                       # (or omit): take all host RAM on the node
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
# ┌──────────┐
# │ Task 0   │ GPU 0 │ CPU 0–6
# └──────────┘
source "$PROJHOME/bin/sbatch-tail.sh"
