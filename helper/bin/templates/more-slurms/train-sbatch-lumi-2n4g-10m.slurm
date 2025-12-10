#!/usr/bin/env bash
# train-sbatch-lumi-2n4g-10m.slurm
#SBATCH --job-name=mammoth-lumi-2n4g-10m
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:mi250:4
#SBATCH --time=00:10:00
#SBATCH --mem=240G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# ╔══════════════════════╗     ╔══════════════════════╗
# ║       Node 1         ║     ║       Node 2         ║
# ║ Task0..3 ↔ GPU0..3   ║     ║ Task4..7 ↔ GPU0..3   ║
# ╚══════════════════════╝     ╚══════════════════════╝
source "$PROJHOME/bin/sbatch-tail.sh"
