#!/bin/bash

#SBATCH -A project_xxxxxxxxx
#SBATCH -J test
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:4

# ==== EDIT THESE (same as in env_setup.sh) ====
PROJECT_DIR=/scratch/project_xxxxxxxxx
VENV=$PROJECT_DIR/$USER/mammoth-venv
CONTAINER=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif
# ==============================================

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
cat > ./two_nodes_wrapper_script.sh << 'EOF'
#!/bin/bash
echo "Node ${SLURM_NODEID} starting training"
echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"

"${VENV}/bin/mammoth_train" \
    -config ./multi_nodes.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT}
EOF

chmod +x ./two_nodes_wrapper_script.sh

# Execute the training script on each node using singularity
srun /usr/bin/singularity exec \
    --env MASTER_NODE="${MASTER_NODE}" \
    --env MASTER_PORT="${MASTER_PORT}" \
    --env VENV="${VENV}" \
    -B "$PROJECT_DIR:$PROJECT_DIR:rw" \
    "$CONTAINER" \
    ./two_nodes_wrapper_script.sh

echo "Finishing at `date`"
