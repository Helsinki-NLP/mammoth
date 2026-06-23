#!/bin/bash

#SBATCH -A project_462000964
#SBATCH -J test
#SBATCH -o ./log/train/%j.out
#SBATCH -e ./log/train/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:30:00
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

# Create the training script that will be executed on each node
cat > /scratch/project_462000964/members/wangchao/training/081225/two_nodes_wrapper_script.sh << 'EOF'
#!/bin/bash
source /scratch/project_462000964/members/wangchao/.venv/bin/activate
cd /scratch/project_462000964/members/wangchao/

echo "Node ${SLURM_NODEID} starting training"
echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"

python mammoth/train.py \
    -config /scratch/project_462000964/members/wangchao/training/081225/two_nodes.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT} \
EOF

chmod +x /scratch/project_462000964/members/wangchao/training/081225/two_nodes_wrapper_script.sh

# Execute the training script on each node using singularity

srun /usr/bin/singularity exec \
--env MASTER_NODE="${MASTER_NODE}" \
--env MASTER_PORT="${MASTER_PORT}" \
-B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
-B /scratch/project_462000964/shared:/scratch/project_462000964/shared:ro \
/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260319_153422/lumi-multitorch-full-u24r64f21m43t29-20260319_153422.sif \
/scratch/project_462000964/members/wangchao/training/081225/two_nodes_wrapper_script.sh

echo "Finishing at `date`"
