#!/bin/bash

#SBATCH -A project_462000964
#SBATCH -J test
#SBATCH -o /scratch/project_462000964/members/wangchao/training/two_nodes/log/training.%j.out
#SBATCH -e /scratch/project_462000964/members/wangchao/training/two_nodes/log/training.%j.err
#SBATCH --partition=small-g
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:4
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
cat > /scratch/project_462000964/members/wangchao/training/two_nodes/multinode_train_script.sh << 'EOF'
#!/bin/bash
source /scratch/project_462000964/members/wangchao/.venv/bin/activate
cd /scratch/project_462000964/members/wangchao/

echo "Node ${SLURM_NODEID} starting training"
echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"

python mammoth/train.py \
    -config /scratch/project_462000964/members/wangchao/training/two_nodes/train.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT} \
EOF

chmod +x /scratch/project_462000964/members/wangchao/training/two_nodes/multinode_train_script.sh

# Execute the training script on each node using singularity

srun /usr/bin/singularity exec \
--env MASTER_NODE="${MASTER_NODE}" \
--env MASTER_PORT="${MASTER_PORT}" \
-B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
-B /scratch/project_462000964/shared:/scratch/project_462000964/shared:ro \
-B /dev/shm:/dev/shm:rw \
/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
/scratch/project_462000964/members/wangchao/training/two_nodes/multinode_train_script.sh

echo "Finishing at `date`"
