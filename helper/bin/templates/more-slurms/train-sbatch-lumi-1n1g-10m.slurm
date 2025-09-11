#!/usr/bin/env bash
# train-sbatch-lumi-1n1g-10m.slurm
#SBATCH --job-name=mammoth-lumi-1n1g-10m
#SBATCH --account=project_462000964
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7              # ~1/8 of node
#SBATCH --gpus-per-task=1
##SBATCH --gres=gpu:mi250:1
#SBATCH --time=00:10:00
#SBATCH --mem=60G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# ┌──────────┐
# │ Task 0   │ GPU 0 │ CPU 0–6
# └──────────┘
source "$PROJHOME/bin/sbatch-tail.sh"
