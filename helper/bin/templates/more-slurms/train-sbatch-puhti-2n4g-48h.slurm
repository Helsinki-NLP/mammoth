#!/usr/bin/env bash
# train-sbatch-puhti-2n4g-48h.slurm
#SBATCH --job-name=mammoth-puhti-2n4g-48h
#SBATCH --account="$ACCOUNT"
#SBATCH --partition=gpu                # check site max walltime
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --mem=180G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
source "$PROJHOME/bin/sbatch-tail.sh"
