#!/bin/bash
module purge
module load pytorch
source /scratch/project_2005099/members/chaowang/.venv27/bin/activate
cd /scratch/project_2005099/members/chaowang/

echo "Node ${SLURM_NODEID} starting training"
echo "Master node: ${MASTER_NODE}"
echo "Master port: ${MASTER_PORT}"

python mammoth/train.py \
    -config /scratch/project_2005099/members/chaowang/training/four_nodes/train.yaml \
    --node_rank ${SLURM_PROCID} \
    --master_ip ${MASTER_NODE} \
    --master_port ${MASTER_PORT} \
