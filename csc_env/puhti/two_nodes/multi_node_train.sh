#!/bin/bash

#SBATCH -A project_
#SBATCH -J mammoth_multinode
#SBATCH -o ./log/training.%j.out
#SBATCH -e ./log/training.%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
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
cat > /scratch/project_2005099/members/chaowang/training/two_nodes/multinode_train_script.sh << 'EOF'
#!/bin/bash
module purge
module load pytorch/2.1
source /scratch/project_2005099/members/chaowang/.venv/bin/activate
cd /scratch/project_2005099/members/chaowang/

echo "Node ${SLURM_NODEID} starting training"
echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"

python mammoth/train.py \
    -config /scratch/project_2005099/members/chaowang/training/two_nodes/train.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT} \
EOF

chmod +x /scratch/project_2005099/members/chaowang/training/two_nodes/multinode_train_script.sh

# Execute the training script on each node directly with environment variables
export MASTER_NODE="${MASTER_NODE}"
export MASTER_PORT="${MASTER_PORT}"
export NCCL_DEBUG=INFO

srun /scratch/project_2005099/members/chaowang/training/two_nodes/multinode_train_script.sh

echo "Finishing at `date`"
