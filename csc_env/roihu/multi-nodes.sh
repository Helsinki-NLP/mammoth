#!/bin/bash

#SBATCH -A project_xxxxxxx
#SBATCH -J 2nodes
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=gpularge
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=288
#SBATCH --mem=0
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:gh200:4

# ==== EDIT THIS (same as in env_setup.sh) ====
VENV=/scratch/project_xxxxxxx/$USER/mammoth-venv
# =============================================

echo "Starting at `date`"

set -eux

# Get the master node (first node in the job allocation)
MASTER_NODE=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)

# Port for distributed training communication
MASTER_PORT=9973

echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"
echo "Node list: ${SLURM_JOB_NODELIST}"

# Create the training script that will be executed on each node
cat > ./multinode_train_script.sh << 'EOF'
#!/bin/bash

module purge
module load python-pytorch/2.10

echo "Node ${SLURM_NODEID} starting training"
echo "Master node: $MASTER_NODE"
echo "Master port: $MASTER_PORT"

"${VENV}/bin/mammoth_train" \
    -config ./multi_nodes.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip $MASTER_NODE \
    --master_port $MASTER_PORT
EOF

chmod +x ./multinode_train_script.sh

# srun passes these environment variables to the script on each node
export MASTER_NODE MASTER_PORT VENV
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=False
export MAMMOTH_PLATFORM=nvidia

srun ./multinode_train_script.sh

echo "Finishing at `date`"
