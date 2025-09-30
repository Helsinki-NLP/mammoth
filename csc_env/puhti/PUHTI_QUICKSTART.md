# Puhti Supercomputer Quick Start Guide

This guide provides quick setup and usage instructions for running MAMMOTH on the Puhti supercomputer.

## Environment Setup

```bash
# Navigate to your project directory on Puhti
cd /scratch/project_YOUR_PROJECT_ID/

# Clone MAMMOTH repository
git clone -b feat/hf_integration https://github.com/Helsinki-NLP/mammoth.git
cd mammoth

# Load PyTorch module and create virtual environment
module purge
module load pytorch/2.1  # Stick to this version for multi-node training (driver/library compatibility as of Sept 2025)
python -m venv .venv
source .venv/bin/activate

# Install MAMMOTH and dependencies
cd mammoth
pip install -r csc_env/puhti/requirements_puhti.txt
```

## Training on Puhti

### Single-Node Training

For single-node training with up to 4 GPUs (You can use pytorch/2.7 for single-node, but stick to pytorch/2.1 for multi-node as of Sept 2025).

#### 1. Prepare Configuration

```bash
# Copy and customize the single-node configuration
cp csc_env/puhti/single_node.yaml ./my_train.yaml
nano my_train.yaml
```

Update in `my_train.yaml`:
- Data paths (`path_src`, `path_tgt`, `path_valid_src`, `path_valid_tgt`)
- Vocabulary paths (`src_vocab`, `tgt_vocab`)
- Model save path (`save_model`)
- Task configuration as needed

#### 2. Prepare SLURM Script

```bash
# Copy and customize the single-node training script
cp csc_env/puhti/single_node.sh ./my_train.sh
nano my_train.sh
```

Update in `my_train.sh`:
- `#SBATCH -A project_XXXXXX` - Your project number
- `#SBATCH --mail-user` - Your email address
- Python path (line 25)
- Config path (line 25)

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
cp csc_env/puhti/multi_node_train.yaml ./my_multinode_train.yaml
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
cp csc_env/puhti/multi_node_train.sh ./my_multinode_train.sh
nano my_multinode_train.sh
```

Update in `my_multinode_train.sh`:
- `#SBATCH -A project_XXXXXX` - Your project number
- `#SBATCH --nodes=2` - Adjust if using different node count
- `#SBATCH --mail-user` - Your email address
- Script creation path (line 32)
- Virtual environment path (line 36)
- Config path (line 44)

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

## Architecture Details

The multi-node setup uses:
- **Shared encoder**: Synchronized across all GPUs via all-reduce
- **Language-specific decoders**: Each GPU maintains its own decoder
- **Task distribution**: `node_gpu` parameter assigns tasks to specific `"node:gpu"` pairs

An example showing two-node training detailed architecture and communication patterns is available in [distributed_training_flowchart.md](distributed_training_flowchart.md).

## Hardware Specifications

- **GPU**: NVIDIA V100 (32GB memory)
- **Driver version**: 535.261.03 (as of Sept 2025)
- **CUDA version**: 12.2 (as of Sept 2025)
- **Nodes**: Up to 80 GPU nodes available
- **GPUs per node**: 4× V100 GPUs
- **Interconnect**: High-speed InfiniBand for inter-node communication



**Note**: Ensure you're using `pytorch/2.1` - other versions may have compatibility issues with multi-node training.

For more information, refer to the [main README](../../README.md) and [Puhti documentation](https://docs.csc.fi/computing/systems-puhti/).