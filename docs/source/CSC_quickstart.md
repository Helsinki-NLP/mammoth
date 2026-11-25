# LUMI & Roihu Quickstart

This guide covers how to set up and run MAMMOTH training jobs on two CSC supercomputers:

- **[LUMI](https://www.lumi.csc.fi/public/)** — AMD MI250X GPU cluster
- **[Roihu](https://docs.csc.fi/computing/systems-roihu/)** — NVIDIA GH200 GPU cluster

Both use SLURM for job scheduling. The main difference is that LUMI runs inside a Singularity container, while Roihu uses a PyTorch module.

On both, you install MAMMOTH with pip into your own virtual environment, on top of the PyTorch that the cluster provides (the container's ROCm build on LUMI, the module's CUDA build on Roihu). The job templates are in [`csc_env/`](https://github.com/Helsinki-NLP/mammoth/tree/main/csc_env). Each template starts with a few variables to edit (project, venv path, container); use the same values in all of them.

Packages installed with `pip install --user` live in `~/.local` and are loaded before those of the container or venv. An old `transformers` there breaks MAMMOTH with errors such as `cannot import name 'PreTrainedModel' from 'transformers'`. The templates set `PYTHONNOUSERSITE=1` to ignore them; do the same in your own scripts (see Troubleshooting in [Installation](install.md)).

---

## LUMI (AMD MI250X)

### Environment Setup

LUMI runs MAMMOTH inside a Singularity container that provides Python and a ROCm build of PyTorch:

```
/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
```

Create your venv once with `csc_env/lumi/env_setup.sh`. Edit `PROJECT_DIR`, `VENV` and the `#SBATCH -A` project first, then submit it:

```bash
mkdir -p log
sbatch csc_env/lumi/env_setup.sh
```

Inside the container, the script runs:

```bash
python -m venv --system-site-packages $VENV   # inherit the container's PyTorch
$VENV/bin/pip install mammoth-nlp
```

`--system-site-packages` keeps the container's PyTorch: pip only installs MAMMOTH and the packages the container lacks. The script prints the torch, transformers and mammoth versions with their paths, and stops with an error if pip replaced the container's PyTorch. To install a specific version, set `MAMMOTH_PKG=mammoth-nlp==<version>`; to work on MAMMOTH itself, set `MAMMOTH_PKG="-e /path/to/your/mammoth/checkout"`.

To keep pip from installing or upgrading any dependency, set `REQ_FILE` to a requirements file of your own, e.g. [`csc_env/requirements.txt`](https://github.com/Helsinki-NLP/mammoth/blob/main/csc_env/requirements.txt). The script then installs MAMMOTH with `--no-deps`, installs that file, and runs `pip check`, which fails if a package MAMMOTH needs (such as `transformers>=5.24`) is missing from both the container and the file.

### Single-Node Training

Use `csc_env/lumi/single_node.sh` as a template. Key SLURM options:

```
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 # from 1-8
```

The script runs training inside the container:

```bash
singularity exec \
    --env PYTHONNOUSERSITE=1 \
    -B $PROJECT_DIR:$PROJECT_DIR:rw \
    $CONTAINER \
    $VENV/bin/mammoth_train -config single_node.yaml
```

The venv does not need to be activated: `$VENV/bin/mammoth_train` runs with the venv's Python.

> `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` optional flag, for peak throughput (incurs a one-time warmup cost)

### Multi-Node Training

Use `csc_env/lumi/multi_nodes.sh` as a template. Key SLURM options for 2 nodes × 4 GPUs:

```
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=28 # each GPU takes 7 CPUs, total 56 CPUs for gres=gpu:8
#SBATCH --gres=gpu:4 # set from 1-8
```

The script finds the master node and starts one process per node with `srun`, inside the container:

```bash
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=9973

srun singularity exec --env VENV=$VENV --env PYTHONNOUSERSITE=1 ... $CONTAINER ./two_nodes_wrapper_script.sh
```

Inside the wrapper, each node gets its rank from `$SLURM_PROCID`:

```bash
$VENV/bin/mammoth_train \
    -config ${SLURM_SUBMIT_DIR}/multi_nodes.yaml \
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

Use `csc_env/lumi/translate.sh` as a template. Edit the SLURM options as needed (inference only supports one GPU at a time):

```bash
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
```

The script runs translation inside the container:

```bash
singularity exec \
    --env PYTHONNOUSERSITE=1 \
    -B $PROJECT_DIR:$PROJECT_DIR:rw \
    $CONTAINER \
    $VENV/bin/mammoth_translate -config inference.yaml
```

Copy `csc_env/inference.yaml` next to the script as a starting point. Key fields to fill in:

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

Roihu does not need a container. Python and a CUDA build of PyTorch come from a module:

```bash
module purge
module load python-pytorch/2.10
```

Create your venv once with `csc_env/roihu/env_setup.sh`. Edit `VENV` and the `#SBATCH -A` project first, then submit it:

```bash
mkdir -p log
sbatch csc_env/roihu/env_setup.sh
```

It does the same as the LUMI setup on top of the module: `python -m venv --system-site-packages $VENV`, then `$VENV/bin/pip install mammoth-nlp`, and checks that the module's PyTorch was kept. `MAMMOTH_PKG` and `REQ_FILE` work as on LUMI.

Set these environment variables before any training run (the templates do this):

```bash
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia
```

### Single-Node Training

Use `csc_env/roihu/single-node.sh` as a template. Key SLURM options:

```
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=0G
#SBATCH --gres=gpu:gh200:1 # from 1-4
```

The script loads the module and runs `$VENV/bin/mammoth_train -config single_node.yaml`. Submit with:

```bash
sbatch csc_env/roihu/single-node.sh
```

### Multi-Node Training

Use `csc_env/roihu/multi-nodes.sh` as a template. Each node has 4 GH200 GPUs, so a 2-node job uses 8 GPUs total.

Key SLURM options:

```
#SBATCH --partition=gpularge
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=288
#SBATCH --mem=0 # Use up all available memory from one node
#SBATCH --gres=gpu:gh200:4
```

The script finds the master node and starts one process per node with `srun`:

```bash
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
MASTER_PORT=9973
export MASTER_NODE MASTER_PORT VENV

srun ./multinode_train_script.sh
```

Inside the wrapper, each node gets its rank from `$SLURM_PROCID`:

```bash
$VENV/bin/mammoth_train \
    -config ./multi_nodes.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip $MASTER_NODE \
    --master_port $MASTER_PORT
```

### Inference (Roihu)

Submit as a SLURM job (single GPU is usually enough), for example:

```bash
#SBATCH --partition=gpumedium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --time=0-00:30:00

module purge
module load python-pytorch/2.10
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia
$VENV/bin/mammoth_translate -config inference.yaml
```
