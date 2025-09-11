

export PROJDATA="${PROJDATA:-/path/to/projdata}"
export EXP_ID="${EXP_ID:-2025-08-21_enfi_base_v1}"
export EXP_DIR="$PROJDATA/exp/$EXP_ID"

# Luo perusrakenne (ajon ulkopuolella riittää tehdä kerran, mutta varmuuden vuoksi)
mkdir -p "$EXP_DIR/slurm/logs" "$EXP_DIR/artifacts/checkpoints" \
         "$EXP_DIR/artifacts/tensorboard" "$EXP_DIR/artifacts/translations" \
         "$EXP_DIR/artifacts/metrics" "$EXP_DIR/tmp"

