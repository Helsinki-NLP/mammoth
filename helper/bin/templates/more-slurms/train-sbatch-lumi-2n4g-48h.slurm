#!/usr/bin/env bash
# train-sbatch-lumi-2n4g-48h.slurm
#SBATCH --job-name=mammoth-lumi-2n4g-48h
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=standard-g         # verify site max walltime
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:mi250:4
#SBATCH --time=48:00:00
#SBATCH --mem=240G
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "$PROJHOME/bin/sbatch-tail.sh"
