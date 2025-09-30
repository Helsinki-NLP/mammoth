# LUMI Supercomputer Quick Start Guide

This guide provides quick setup and usage instructions for running MAMMOTH on the LUMI supercomputer.

## Prerequisites

- Access to LUMI supercomputer
- Project allocation for compute resources
- Basic familiarity with SLURM job scheduling

## Environment Setup

### 1. Set Up Python Virtual Environment

Use the provided setup script to create a virtual environment with PyTorch:

```bash
# Navigate to your project directory on LUMI
cd /scratch/project_YOUR_PROJECT_ID/project/directory/

# Clone MAMMOTH repository
git clone -b feat/hf_integration https://github.com/Helsinki-NLP/mammoth.git
cd mammoth

# Copy and customize the environment setup script
cp lumi/env_setup.sh ./setup_env.sh

# Edit the script to set your specific paths
# Replace $your_path_in_lumi with your actual project path
# Replace $path_to_mammoth with your mammoth directory path
nano setup_env.sh

# Run the setup
source setup_env.sh
```

## Training on LUMI

### 1. Prepare Training Script

```bash
# Edit the script to set your project ID and paths
cp lumi/env_setup.sh ./training.sh
nano training.sh
```

Update the following in `training.sh`:

- Replace path placeholders with your actual paths
- Adjust resource requirements (nodes, memory, time) as needed

### 2. Submit Training Job

```bash
# Submit the training job
sbatch training.sh

# Check job status
squeue -u YOUR_USERNAME

```

## Translation/Inference on LUMI

### 1. Prepare Translation Script

```bash
# Edit the script to set your project ID and paths
cp lumi/translation.sh ./translation.sh
nano translation.sh
```

### 2. Submit Translation Job

```bash
# Submit the translation job
sbatch translation.sh
```

## Container Information

MAMMOTH on LUMI uses the official PyTorch container (latest by 15-Sep-2025):

- Container: `/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`
- Includes: PyTorch, ROCm support, Python 3.12
- Additional dependencies installed via `requirements_lumi.txt`

For more detailed information, refer to the main README.md and LUMI documentation.
