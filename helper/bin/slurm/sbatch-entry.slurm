#!/usr/bin/env bash

# Template for creating this local file:
#    base/mammoth-helper/helper/bin/slurm/sbatch-entry.slurm
# Usage:
#    sbatch -J $JOB_NAME -A "$ACCOUNT" -o logs/%x-%j.out -e logs/%x-%j.err sbatch-entry.slurm
#
# Copy this file to your local experiment directory with name "job_name".
# Assume *local* directory names: mammoth, models, venv (in sbatch-tail), bin/slurm, logs.
# Use a create script to symlink these directories to your experiment directory.

#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7              # ~1/8 of node
#SBATCH --gpus-per-task=1
##SBATCH --gres=gpu:mi250:1
#SBATCH --time=00:10:00
#SBATCH --mem=60G

# ┌──────────┐
# │ Task 0   │ GPU 0 │ CPU 0–6
# └──────────┘

export JOB_PATTERN=slurm              # slurm or torchrun
export JOB_SCRIPT=mammoth/hf2mammoth.py
export JOB_ARGS=vgaraujov/bart-base-translation-en-es models/converted_model --src-lang en --tgt-lang es

export GUARD_MAX_NODES=4                       # do not change unless you need to
export GUARD_TIME="${GUARD_TIME:-0-01:00:00}"  # do not change unless you need to
# examples: GUARD_TIME="12:00:00" (12h) or GUARD_TIME="2-00:00:00" (2 days).

source base/mammoth-helper/helper/bin/slurm/sbatch-tail.sh

