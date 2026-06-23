#!/bin/bash

#SBATCH -A project_
#SBATCH -J training
#SBATCH -o ./log/training.%j.out
#SBATCH -e ./log/training.%j.err
#SBATCH --partition=dev-g     
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-00:30:00
#SBATCH --gres=gpu:1 

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e


singularity exec \
    -B $your_path_in_lumi:$your_path_in_lumi:rw \
    /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif \
    $your_path_in_lumi/.venv/bin/python $path_to_mammoth/translate.py \
    -config csc_env/inference.yaml

echo "Finishing at `date`"

