#!/bin/bash

#SBATCH -A project_2017852
#SBATCH -J 2nodes
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=gpularge
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:gh200:4
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
cat > ./multinode_train_script.sh << 'EOF'
#!/bin/bash

module purge
load python-pytorch/2.10
source /scratch/project_2017852/mammoth-shared/.venv/bin/activate


echo "Node ${SLURM_NODEID} starting training"
echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"

export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia

python /scratch/project_2017852/mammoth-shared/mammoth_pytorch/mammoth/train.py \
    -config <training_config.yaml> \
    --node_rank ${SLURM_PROCID} \
    --master_ip $MASTER_NODE \
    --master_port $MASTER_PORT \
EOF

chmod +x ./multinode_train_script.sh

# Execute the training script on each node directly with environment variables
export MASTER_NODE="${MASTER_NODE}"
export MASTER_PORT="${MASTER_PORT}"
export MAMMOTH_PLATFORM=nvidia

srun ./multinode_train_script.sh

echo "Finishing at `date`"