#!/bin/bash

#SBATCH -A project_462000964
#SBATCH -J convert
#SBATCH -o ./log/convert/%j.out
#SBATCH -e ./log/convert/%j.err
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=0-00:15:00
#SBATCH --gres=gpu:1


echo "Starting at `date`"

# stops the script when encountering an error
# (useful if running several commands in the same script)
set -e


singularity exec \
    --env FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE" \
    -B /scratch/project_462000964/members/wangchao:/scratch/project_462000964/members/wangchao:rw \
    -B /scratch/project_462000964/shared:/scratch/project_462000964/shared:rw \
    /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
    /scratch/project_462000964/members/wangchao/.venv/bin/python /scratch/project_462000964/members/wangchao/mammoth/mammoth/hf_integration/from_hf/modernBERT/BERT2mammoth.py \
    /scratch/project_462000964/members/wangchao/training/hf_models/modernbert \
    ./converted_modernbert/ \
    --tgt-tokenizer fi /scratch/project_462000964/shared/hplt_bilingual/fi-en.tmx/tgt_tokenizer/tokenizer.json

echo "Finishing at `date`"

