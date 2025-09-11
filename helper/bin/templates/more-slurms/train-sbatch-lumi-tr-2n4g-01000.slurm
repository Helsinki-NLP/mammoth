#!/usr/bin/env bash
# mammoth-lumi-tr-2n4g-01000.slurm
#SBATCH --job-name=mammoth-lumi-tr-2n4g-01000
#SBATCH --account="$ACCOUNT"
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=4
#SBATCH --time=00:10:00
#SBATCH --mem=240G
#SBATCH --hint=nomultithread
# LUMI torchrun multi-node: 1 task/node; 4 GPU workers per node (Slingshot CXI fabric)
# ╔══════════════════════╗     ╔══════════════════════╗
# ║       Node 1         ║     ║       Node 2         ║
# ║ torchrun spawns 4    ║     ║ torchrun spawns 4    ║
# ║ GPU workers (0–3)    ║     ║ GPU workers (0–3)    ║
# ╚══════════════════════╝     ╚══════════════════════╝
export PATTERN=torchrun
source "$PROJHOME/bin/sbatch-tail.sh"
