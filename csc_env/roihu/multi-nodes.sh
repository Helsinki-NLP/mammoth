#!/bin/bash

#SBATCH -A project_2001194
#SBATCH -J 2nodes
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=6
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=40
#SBATCH --mem=0
#SBATCH --time=12:00:00
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
cat > /scratch/project_2001194/chao/test_shared_env/two_nodes/multinode_train_script.sh << 'EOF'
#!/bin/bash

module purge
module load pytorch
source /scratch/project_2001194/mammoth-shared/.venv/bin/activate


echo "Node ${SLURM_NODEID} starting training"
echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"

export TOKENIZERS_PARALLELISM=False

python /scratch/project_2001194/mammoth-shared/mammoth_dev/mammoth/train.py \
    -config /scratch/project_2001194/chao/test_shared_env/two_nodes/multi_node_train.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip $MASTER_NODE \
    --master_port $MASTER_PORT \
EOF

chmod +x /scratch/project_2001194/chao/test_shared_env/two_nodes/multinode_train_script.sh

# Execute the training script on each node directly with environment variables
export MASTER_NODE="${MASTER_NODE}"
export MASTER_PORT="${MASTER_PORT}"

srun /scratch/project_2001194/chao/test_shared_env/two_nodes/multinode_train_script.sh

echo "Finishing at `date`"