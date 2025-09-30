# LUMI Supercomputer Quick Start Guide

This guide provides quick setup and usage instructions for running MAMMOTH on the LUMI supercomputer.


## Environment Setup

Use the provided setup script to create a virtual environment with PyTorch:

```bash
# Navigate to your project directory on LUMI
cd /scratch/project_YOUR_PROJECT_ID/

# Clone MAMMOTH repository
git clone -b feat/hf_integration https://github.com/Helsinki-NLP/mammoth.git
cd mammoth

# Copy and customize the environment setup script
cp csc_env/lumi/env_setup.sh ./setup_env.sh

# Edit the script to set your specific paths
# Replace $your_path_in_lumi with your actual project path
# Replace $path_to_mammoth with your mammoth directory path
nano setup_env.sh

# Run the setup
source setup_env.sh
```

The setup script will:
- Load the LUMI PyTorch container (ROCm 6.2.4, Python 3.12, PyTorch 2.7.1)
- Create a Python virtual environment
- Install MAMMOTH and required dependencies

## Training on LUMI

### Single-Node Training

For single-node training with up to 8 GPUs.

#### 1. Prepare Configuration

```bash
# Copy and customize the single-node configuration
cp csc_env/lumi/single_node_train.yaml ./my_train.yaml
nano my_train.yaml
```

Update in `my_train.yaml`:
- Data paths (`path_src`, `path_tgt`, `path_valid_src`, `path_valid_tgt`)
- Vocabulary paths (`src_vocab`, `tgt_vocab`, `src_subword_model`, `tgt_subword_model`)
- Model save path (`save_model`)
- Task configuration as needed

#### 2. Prepare SLURM Script

```bash
# Copy and customize the single-node training script
cp csc_env/lumi/single_node_train.sh ./my_train.sh
nano my_train.sh
```

Update in `my_train.sh`:
- `#SBATCH -A project_XXXXXX` - Your project number
- Mount paths (line 22) - Replace `$your_path_in_lumi` with your actual path
- Python and config paths (lines 24-25) - Replace `$your_path_in_lumi` and `$path_to_mammoth`

#### 3. Submit Job

```bash
mkdir -p log
sbatch my_train.sh
```

### Multi-Node Distributed Training

For distributed training across multiple nodes. Example: 2 nodes × 4 GPUs = 8 GPUs total.

#### 1. Prepare Configuration

```bash
# Copy and customize the multi-node configuration
cp csc_env/lumi/multi_node_train.yaml ./my_multinode_train.yaml
nano my_multinode_train.yaml
```

Key configuration points:
- `world_size: 8` - Total GPUs across all nodes
- `n_nodes: 2` - Number of nodes
- `node_gpu: "0:0"` - Task assignment (format: `"node:gpu"`)
- Update all data paths and vocabulary paths
- Adjust `save_model` path

#### 2. Prepare SLURM Script

```bash
# Copy and customize the multi-node training script
cp csc_env/lumi/multi_node_train.sh ./my_multinode_train.sh
nano my_multinode_train.sh
```

Update in `my_multinode_train.sh`:
- `#SBATCH -A project_XXXXXX` - Your project number
- `#SBATCH --nodes=2` - Adjust if using different node count
- `#SBATCH --mail-user` - Your email address
- Script creation path (line 32)
- Mount paths (lines 55-57)
- Virtual environment and config paths (line 59, line 42)

#### 3. Submit Job

```bash
mkdir -p log
sbatch my_multinode_train.sh
```

#### 4. Monitor Training

```bash
# Check job status
squeue -u $USER

# View live output
tail -f log/training.<job_id>.out

# Check for errors
tail -f log/training.<job_id>.err
```

## Translation/Inference on LUMI

```bash
# Copy and customize the translation script
cp csc_env/lumi/translate.sh ./my_translate.sh
nano my_translate.sh

# Update paths and submit
sbatch my_translate.sh
```

## Container Information

MAMMOTH on LUMI uses the official PyTorch container (latest as of Sept 2025):

- **Container**: `/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`
- **Includes**: PyTorch 2.7.1, ROCm 6.2.4, Python 3.12 (as of Sept 2025)
- **GPU**: AMD MI250X GPUs
- **Additional dependencies**: Installed via `requirements_lumi.txt`

## Hardware Specifications

- **GPU**: AMD MI250X (each GCD has 64GB HBM2e memory)
- **Nodes**: Pre-exascale system with thousands of GPU nodes
- **GPUs per node**: 8× MI250X GCDs (4× MI250X packages, 2 GCDs per package)
- **Interconnect**: HPE Slingshot for high-speed inter-node communication

For more information, refer to the [main README](../../README.md) and [LUMI documentation](https://docs.lumi-supercomputer.eu/).