#!/bin/bash -l
#SBATCH --job-name=2025-08-21_enfi_1n1g-30m
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=project_xxxxxx
#SBATCH --output=%x/slurm/logs/%x_%j.out
#SBATCH --error=%x/slurm/logs/%x_%j.err

# --- Polut (lyhyet ja johdonmukaiset) ---
export PROJDATA="${PROJDATA:-/path/to/projdata}"
export EXP_ID="${EXP_ID:-2025-08-21_enfi_base_v1}"
export EXP_DIR="$PROJDATA/exp/$EXP_ID"

# Luo perusrakenne (ajon ulkopuolella riittää tehdä kerran, mutta varmuuden vuoksi)
mkdir -p "$EXP_DIR/slurm/logs" "$EXP_DIR/artifacts/checkpoints" \
         "$EXP_DIR/artifacts/tensorboard" "$EXP_DIR/artifacts/translations" \
         "$EXP_DIR/artifacts/metrics" "$EXP_DIR/tmp"

# --- Modulit/ympäristö (Python 3.10 oma venv) ---
module load LUMI
module load partition/G
module load rocm
# aktivoi oma Python 3.10 -ympäristösi:
source /path/to/venv/bin/activate

# ROCm/MIOPEN-välimuisti levyn sijaan /tmp:iin
export MIOPEN_USER_DB_PATH="$EXP_DIR/tmp/miopen-$SLURM_JOB_ID"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# --- Ajo ---
srun python -u -m mammoth.train -config "$EXP_DIR/cfg/config.yml" -tasks tasks
