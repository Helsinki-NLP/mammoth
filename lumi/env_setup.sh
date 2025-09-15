#!/bin/bash
# This script sets up the Python virtual environment and installs necessary packages
# for the PyTorch project on the LUMI supercomputer.

# Create virtual environment in your project space
singularity exec \
    -B $your_path_in_lumi:rw \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
    python -m venv $your_path_in_lumi/.venv --system-site-packages

# Install packages (this preserves PyTorch from container)
singularity exec \
    -B $your_path_in_lumi:$your_path_in_lumi:rw \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
    $your_path_in_lumi/.venv/bin/pip install -r $path_to_mammoth/lumi/requirements_lumi.txt


# Example usage:

# Create virtual environment in your project space
# singularity exec \
#     -B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
#     /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
#     python -m venv /scratch/project_462000964/members/wangchao/.venv --system-site-packages

# Install packages (this preserves PyTorch from container)
# singularity exec \
#     -B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
#     /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
#     /scratch/project_462000964/members/wangchao/.venv/bin/pip install -r /scratch/project_462000964/members/wangchao/mammoth/lumi/requirements_lumi.txt