
# LUMI & Roihu Quickstart

This guide covers how to set up and run MAMMOTH training jobs on two CSC supercomputers:

- **[LUMI](https://www.lumi.csc.fi/public/)** — AMD MI250X GPU cluster
- **[Roihu](https://docs.csc.fi/computing/systems-roihu/)** — NVIDIA GH200 GPU cluster

Both use SLURM for job scheduling. The main difference is that LUMI requires a Singularity container while Roihu runs with native modules.

---

## LUMI (AMD MI250X)

### Environment Setup

LUMI requires running inside a Singularity container. A shared container image and virtual environment are maintained at:

```
Container: /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
```

To set up the virtual environment for the first time, use the setup script:

```bash
sbatch csc_env/lumi/env_setup.sh
```

This creates a venv inside the container with all dependencies. Edit the script first to set `PROJECT_PATH` and `VENV_PATH` to your project's paths.

### Single-Node Training

Use `csc_env/lumi/single_node/single_node_train.sh` as a template. Key SLURM options:

```
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # from 1-8
```

The script runs training inside the container:

```bash
singularity exec \
    -B /scratch/<your_project>:/scratch/<your_project>:rw \
    /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif \
    <Path to Python> mammoth/train.py \
    -config train.yaml
```

> `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` optional flag, for peak throughput (incurs a one-time warmup cost)

### Multi-Node Training

Use `csc_env/lumi/multi_nodes/two_nodes.sh` as a template. Key SLURM options for 2 nodes × 4 GPUs:

```
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=28 # each GPU takes 7 CPUs, total 56 CPUs for gres=gpu:8
#SBATCH --gres=gpu:4 # set from 1-8
```

The script auto-detects the master node and launches training via `srun`:

```bash
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=9973

srun singularity exec ... /path/to/wrapper_script.sh
```

Inside the wrapper, each node gets its rank from `$SLURM_PROCID`:

```bash
python mammoth/train.py \
    -config two_nodes.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT}
```

Your config must match the allocation:

```yaml
n_nodes: 2
world_size: 8
gpu_ranks: [0, 1, 2, 3]
```

### Inference (LUMI)

Use `csc_env/translate.sh` as a template. Edit the SLURM options as needed (inference only supports one GPU at a time):

```bash
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
```

The script runs translation inside the container:

```bash
singularity exec \
    -B /scratch/<your_project>:/scratch/<your_project>:rw \
    /appl/local/laifs/containers/.../lumi-multitorch-full-....sif \
    <Path to Python> mammoth/translate.py \
    -config inference.yaml
```

Use the provided `csc_env/inference.yaml` as a starting point. Key fields to fill in:

```yaml
model: /path/to/checkpoints/  # directory path to checkpoint (no .pt extension)
src: /path/to/input.txt                             # source sentences, one per line
output: /path/to/output.txt                         # translation output
task_id: task_en_fi                                  # must match a task from training
beam_size: 5                                         # beam search width
batch_size: 32
batch_type: sents
```
---

## Roihu (NVIDIA GH200)

### Environment Setup

Roihu does not require a container. Load the PyTorch module and activate the shared virtual environment:

```bash
module purge
module load python-pytorch/2.10
source <Path to Python venv> # activate the virtual environment
```

Set these environment variables before any training run:

```bash
export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia
```

To set up the shared venv for the first time, create it with `--system-site-packages` so it inherits PyTorch and core ML dependencies from the loaded module, then install the extras:

```bash
module load python-pytorch/2.10
python -m venv .venv --system-site-packages
source <Path to Python venv>/bin/activate # use your path to the newly created venv
pip install -r <Path to Mammoth>/mammoth/csc_env/requirements.txt
```

### Single-Node Training

Use `csc_env/roihu/single-node.sh` as a template. Key SLURM options:

```
#SBATCH --partition=gpupilot (only available during Roihu pilot phase, subject to change)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=120G
#SBATCH --gres=gpu:gh200:1 # from 1-4
```

Submit with:

```bash
sbatch csc_env/roihu/single-node.sh
```

### Multi-Node Training

Use `csc_env/roihu/multi-nodes.sh` as a template. Each node has 4 GH200 GPUs, so a 2-node job uses 8 GPUs total.

Key SLURM options:

```
#SBATCH --partition=gpupilot
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=288
#SBATCH --mem=0 # Use up all available memory from one node 
#SBATCH --gres=gpu:gh200:4
```

The script auto-detects the master node and runs training via `srun`:

```bash
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=9973

srun ./multinode_train_script.sh
```

Inside the wrapper, each node gets its rank from `$SLURM_PROCID`:

```bash
python train.py \
    -config multi_node_train.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT}
```

### Inference (Roihu)

On Roihu, run translation directly without a container:

```bash
module purge
module load python-pytorch/2.10
source <Path to Python venv>

python mammoth/translate.py \
    -config inference.yaml \
    --gpu_ranks 0
```

Or submit as a SLURM job (single GPU is usually enough):

```bash
#SBATCH --partition=gpupilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=0-00:30:00
```

Use the same `csc_env/inference.yaml` template as LUMI. Fill in `model`, `src`, `output`, and `task_id`. 

---

## Monitoring Jobs

```bash
squeue -u $USER                         # list your running jobs
scontrol show job <job_id>              # job details
tail -f ./log/training.<job_id>.out    # live stdout
tail -f ./log/training.<job_id>.err    # live stderr
```
