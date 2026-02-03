#!/bin/bash

#SBATCH -A project_2001970
#SBATCH -J 2nodes
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=gputest
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=10
#SBATCH --mem=256G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:v100:4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=chao.wang@helsinki.fi

echo "Starting at `date`"

set -eux

# Get the master node (first node in the job allocation)
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

# Generate a random port for distributed training communication
MASTER_PORT=9973

echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"
echo "Node list: ${SLURM_JOB_NODELIST}"

# Create the training script that will be executed on each node
cat > /scratch/project_2001970/members/chao/trainings/01122025/multinode_train_script.sh << 'EOF'
#!/bin/bash

module purge
module load pytorch
source /scratch/project_2001970/members/chao/.venv/bin/activate
cd /scratch/project_2001970/members/chao

echo "Node ${SLURM_NODEID} starting training"
echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"

export TOKENIZERS_PARALLELISM=False

# Enhanced NCCL debugging for distributed communication profiling
# export NCCL_DEBUG_SUBSYS=COLL  # Detailed collective operations
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Better memory profiling

# Create profiling output directory if it doesn't exist
mkdir -p /scratch/project_2001970/members/chao/trainings/01122025/profiling
export TMPDIR=/scratch/project_2001970/members/chao/trainings/01122025/profiling/tmp
mkdir -p $TMPDIR

# Check if nsys is available
if ! command -v nsys &> /dev/null; then
    echo "ERROR: nsys command not found. Please load the appropriate module."
    echo "Available profiling modules might include: nvhpc, cuda-nsight-systems, etc."
    exit 1
fi

echo "nsys version: $(nsys --version)"
echo "Starting profiling with 120 second delay..."
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Enhanced profiling with comprehensive tracing
nsys profile \
    -t nvtx,cuda,cublas,cudnn,osrt \
    -o /scratch/project_2001970/members/chao/trainings/01122025/profiling/node_${SLURM_PROCID} \
    --stats=true \
    --force-overwrite true \
    --delay 120 \
    --duration 30 \
    --capture-range=cudaProfilerApi \
    --cuda-memory-usage=true \
    --backtrace=dwarf \
    --sample=cpu \
    python mammoth/train.py \
        -config /scratch/project_2001970/members/chao/trainings/01122025/train.yaml \
        --node_rank ${SLURM_PROCID} \
        --master_ip $MASTER_NODE \
        --master_port $MASTER_PORT \
EOF

chmod +x /scratch/project_2001970/members/chao/trainings/01122025/multinode_train_script.sh

# Execute the training script on each node directly with environment variables
export MASTER_NODE="${MASTER_NODE}"
export MASTER_PORT="${MASTER_PORT}"
export NCCL_DEBUG=INFO

srun /scratch/project_2001970/members/chao/trainings/01122025/multinode_train_script.sh

echo "Finishing at `date`"