# Start lightweight GPU monitor by calling the following function:
MON_PID=""
if [[ -n "${OUTPUT_DIR-}" ]]; then
  MON_OUT="${OUTPUT_DIR}/gpu_load-${SLURM_NODEID}.log"
else
  MON_OUT="id-${SLURM_JOB_ID}/gpu_load-${SLURM_NODEID}.log"
fi
start_monitor() {
  if [[ "$SYSTEM" == "puhti" || "$SYSTEM" == "mahti" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi dmon -s mu -d 5 -o TD > "$MON_OUT" &
      MON_PID=$!
    fi
  else # LUMI / ROCm
    if command -v rocm-smi >/dev/null 2>&1; then
      # sample every 5s: util, power, temp, vram
      while true; do
        rocm-smi --showuse --showtemp --showpower --showmemuse || true
        sleep 5
      done > "$MON_OUT" &
      MON_PID=$!
    fi
  fi
}
stop_monitor() { [[ -n "$MON_PID" ]] && kill "$MON_PID" 2>/dev/null || true; }

trap stop_monitor EXIT  # clean with trap

start_monitor  # Start monitor (comment out if you don’t want it)

