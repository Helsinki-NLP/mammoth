#!/bin/bash

#SBATCH -A project_
#SBATCH -J training
#SBATCH -o ./log/training.%j.out
#SBATCH -e ./log/training.%j.err
#SBATCH --partition=standard-g     
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-4:00:00
#SBATCH --gres=gpu:1 

echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e


singularity exec \
    -B $your_path_in_lumi:$your_path_in_lumi:rw \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
    $your_path_in_lumi/.venv/bin/python $path_to_mammoth/train.py \
    -config $path_to_mammoth/training_ft.yaml

echo "Finishing at `date`"
