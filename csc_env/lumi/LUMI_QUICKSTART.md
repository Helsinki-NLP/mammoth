# LUMI Supercomputer Quick Start Guide

This guide provides quick setup and usage instructions for running MAMMOTH on the LUMI supercomputer.


## Environment Setup

Use the provided setup script to create a virtual environment with PyTorch:

```bash
# Navigate to your project directory on LUMI
cd /scratch/project_YOUR_PROJECT_ID/

# Clone MAMMOTH repository
git clone -b feat/hf_integration_lumi https://github.com/Helsinki-NLP/mammoth.git
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
- Install Mammoth required dependencies (but not Mammoth itself as current run does not require installing Mammoth to lib path)

## Training on LUMI

### Single-Node Training

For single-node training with up to 8 GPUs:

- Check `csc_env/lumi/single_node/single_node_train.yaml` for training configuration template.
- Check `csc_env/lumi/single_node/single_node_train.sh` for slurm job script template.
- Use `sbatch single_node_train.sh` to submit the job to LUMI system.

### Multi-Node Distributed Training

For distributed training across multiple nodes. Example: 2 nodes × 4 GPUs = 8 GPUs total.

- Check `csc_env/lumi/two_nodes/two_nodes.yaml` for training configuration template.
Key configuration points:
- `world_size: 8` - Total GPUs across all nodes
- `n_nodes: 2` - Number of nodes
- `node_gpu: "0:0"` - Task assignment (format: `"node:gpu"`)

- Check `csc_env/lumi/two_nodes/two_nodes.sh` for slurm job script template.
- Use `sbatch multi_node_train.sh` to submit the job to LUMI system.


### And template recipes for 4-node and 8-node training are in:
`csc_env/lumi/four_nodes` and `csc_env/lumi/eight_nodes`

## Translation/Inference on LUMI

- Check `csc_env/lumi/inference.yaml` for inferencing configuration template
Note: 
Please align the model settings as possible as you can. Copying the model training settings (model architectures, x-transformer opts etc) to the inference config file is preferred, because Mammoth is building a new such model before loading the trained weights for the inferencing. 
- Check `csc_env/lumi/inference.yaml` for slurm script template.

## Container Information

MAMMOTH on LUMI uses the official PyTorch container (latest as of Sept 2025):

- **Container**: `/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif`
- **Includes**: PyTorch 2.7.1, ROCm 6.2.4, Python 3.12 (as of Sept 2025)
- **GPU**: AMD MI250X GPUs
- **Additional dependencies**: Installed via `requirements_lumi.txt`


For more information, refer to the [main README](../../README.md) and [LUMI documentation](https://docs.lumi-supercomputer.eu/).