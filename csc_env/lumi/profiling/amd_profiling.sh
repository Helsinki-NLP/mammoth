#!/bin/bash

#SBATCH -A project_462000964
#SBATCH -J profiling
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:4


echo "Starting at `date`"

set -eux

# Get the master node (first node in the job allocation)
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

# Generate a random port for distributed training communication
MASTER_PORT=9973

echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"
echo "Node list: ${SLURM_JOB_NODELIST}"

# Create profiling output directory
PROFILING_DIR="/scratch/project_462000964/members/wangchao/profiling"
mkdir -p ${PROFILING_DIR}

# Create the training script that will be executed on each node
cat > /scratch/project_462000964/members/wangchao/profiling/amd_profiler.sh << 'EOF'
#!/bin/bash
source /scratch/project_462000964/members/wangchao/.venv/bin/activate

echo "Node ${SLURM_NODEID} (Process ${SLURM_PROCID}) starting training"
echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"

export TOKENIZERS_PARALLELISM=False

# ROCm profiling environment variables
export HSA_ENABLE_SDMA=0  # Disable SDMA for more accurate profiling
export ROCM_PATH=/opt/rocm  # Adjust if ROCm is installed elsewhere in container

# Create profiling output directory if it doesn't exist
mkdir -p /scratch/project_462000964/members/wangchao/profiling
export TMPDIR=/scratch/project_462000964/members/wangchao/profiling/tmp
mkdir -p $TMPDIR

# Check if rocprofv3 is available
if ! command -v rocprofv3 &> /dev/null; then
    echo "ERROR: rocprofv3 command not found. Profiling unavailable."
    echo "Note: rocprofv3 requires ROCm 6.0+. Try 'module load rocm/6.2.4' or newer."
    exit 1
fi

echo "Using rocprofv3 for profiling"
echo "ROCm version:"
cat /opt/rocm/.info/version || echo "ROCm version file not found"
echo "GPU info:"
rocm-smi --showproductname || echo "rocm-smi not available"

# Set PYTHONPATH for ROCTx Python bindings
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH

# Verify ROCTx is available
python -c "import roctx; print('ROCTx Python bindings available')" || \
    echo "WARNING: ROCTx Python bindings not found. Profiling markers may not work."

# Change to profiling directory before running rocprofv3
cd /scratch/project_462000964/members/wangchao/profiling

# Create per-rank output directory
RANK_DIR="/scratch/project_462000964/members/wangchao/profiling/rank_${SLURM_PROCID}"
mkdir -p ${RANK_DIR}

rocprofv3 \
    --marker-trace \
    --output-format csv \
    --output-dir ${RANK_DIR} \
    -- python /scratch/project_462000964/members/wangchao/mammoth/train.py \
        -config /scratch/project_462000964/members/wangchao/profiling/two_nodes.yaml \
        --node_rank ${SLURM_PROCID} \
        --master_ip ${MASTER_NODE} \
        --master_port ${MASTER_PORT}
EOF

chmod +x /scratch/project_462000964/members/wangchao/profiling/amd_profiler.sh

# Execute the training script on each node using singularity

srun /usr/bin/singularity exec \
--env MASTER_NODE="${MASTER_NODE}" \
--env MASTER_PORT="${MASTER_PORT}" \
--env PROFILING_DIR="${PROFILING_DIR}" \
-B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
-B /scratch/project_462000964/shared:/scratch/project_462000964/shared:ro \
/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
/scratch/project_462000964/members/wangchao/profiling/amd_profiler.sh

echo "Finishing at `date`"
echo "Profiling results saved to: ${PROFILING_DIR}"
