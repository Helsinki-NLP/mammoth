echo comms-setup.sh...

set -euo pipefail
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
: "${SYSTEM:?❌ SYSTEM is not set}"
: "${SLURM_JOB_ID:?❌ SLURM_JOB_ID is not set}"
: "${SLURM_JOB_NODELIST:?❌ SLURM_JOB_NODELIST is not set}"

# ---------- cluster-specific comm knobs (minimal, safe) ----------

if [[ "$SYSTEM" == "lumi" ]]; then
  # Slingshot/CXI + ROCm: quiet & correct HMEM
  export FI_PROVIDER="${FI_PROVIDER:-cxi}"
  export FI_HMEM="${FI_HMEM:-rocr}"
  export FI_LOG_LEVEL="${FI_LOG_LEVEL:-warn}"
  export FI_LOG_PROV="${FI_LOG_PROV:-cxi}"
  # Bootstrap interface for RCCL (choose a single HSN to avoid noise)
  export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-hsn0}"
  # Keep DMABUF off unless you validated your stack
  unset NCCL_DMABUF_ENABLE  # intentionally unset by default
  export NCCL_NET_GDR_LEVEL=PHB
  export RCCL_ENABLE_OFI=1
  export PLUGIN_DIR=$PROJHOME/rccl-lib3.10       # This has symlinks to /opt/aws-ofi-rccl/librccl-net.so
  export LD_LIBRARY_PATH=$PROJHOME/rccl-lib3.10:$LD_LIBRARY_PATH

  # Cray HPE recommended: can sometimes be detrimental for performance at small
  # scale but useful to improve stability at large scale.
  #
  # Quoted by Samuel from AMD: 18 Aug 2025
  # export  FI_MR_CACHE_MONITOR=userfaultfd
  # export  FI_CXI_DEFAULT_CQ_SIZE=131072
  # export  FI_CXI_RX_MATCH_MODE=software
  # export  FI_CXI_RDZV_PROTO=alt_read
fi

echo ==============================================
echo " SLURM_JOB_ID              : $SLURM_JOB_ID"
echo " SLURM_JOB_NODELIST        : $SLURM_JOB_NODELIST"
echo " FI_PROVIDER (cxi)         : $FI_PROVIDER"
echo " FI_HMEM (rocr)            : $FI_HMEM"
echo " FI_LOG_LEVEL (warn)       : $FI_LOG_LEVEL"
echo " FI_LOG_PROV (cxi)         : $FI_LOG_PROV"
echo " PLUGIN_DIR                : $PLUGIN_DIR"
echo " RCC_ENABLE_OFI            : $RCCL_ENABLE_OFI"
echo " LD_LIBRARY_PATH           : $LD_LIBRARY_PATH"
echo " NCCL_SOCKET_IFNAME (hsn0) : $NCCL_SOCKET_IFNAME"
echo " NCCL_NET_GDR_LEVEL        : $NCCL_NET_GDR_LEVEL"
echo ==============================================



###############################################################################
# --- RCCL net plugin presence sanity check -----------------------------------
###############################################################################
# We want BOTH names available because different RCCL builds may dlopen either:
#   - librccl-net-ofi.so  (OFI provider plugin)
#   - librccl-net.so      (generic/legacy name some builds still look for)
#
# Search order: PLUGIN_DIR (if set) first, then LD_LIBRARY_PATH entries.
# Accept either name; accept regular file OR symlink (even dangling).
_rccl_find_in_path() {
  local lib="$1" d
  local -a dirs=()
  [[ -n "${PLUGIN_DIR:-}" ]] && dirs+=("$PLUGIN_DIR")
  if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    IFS=: read -r -a d <<<"$LD_LIBRARY_PATH"
    dirs+=("${d[@]}")
  fi

  shopt -s nullglob
  for d in "${dirs[@]}"; do
    [[ -z "$d" ]] && d=.
    for cand in "$d/$lib" "$d/$lib".*; do
      # present if regular file OR symlink (even if target not yet mounted)
      if [[ -f "$cand" || -L "$cand" ]]; then
        printf '%s\n' "$cand"
        return 0
      fi
    done
  done
  return 1
}

_missing=0
declare -A _FOUND=()
for _lib in librccl-net-ofi.so librccl-net.so; do
  if _path="$(_rccl_find_in_path "$_lib")"; then
    note=""
    if [[ -L "$_path" && ! -e "$_path" ]]; then
      note=" (dangling now; will resolve inside the container)"
    fi
    echo "OK[rccl-plugin]: found $_lib → $_path$note"
    _FOUND["$_lib"]="$_path"   # <- fixed typo
  else
    echo "❌ MISSING: $_lib not found in \$PLUGIN_DIR/\$LD_LIBRARY_PATH." >&2
    _missing=1
  fi
done

if (( _missing )); then
  echo "HINT:" >&2
  echo "  • Put both names in one dir (e.g. \$PLUGIN_DIR) and prepend it to LD_LIBRARY_PATH:" >&2
  echo "      export PLUGIN_DIR=/path/to/rccl-plugins" >&2
  echo "      export LD_LIBRARY_PATH=\"\$PLUGIN_DIR:\${LD_LIBRARY_PATH:-}\"" >&2
  echo "  • If you only have one of them, add a symlink so both names exist in the same dir:" >&2
  echo "      cd \$PLUGIN_DIR; ln -s librccl-net-ofi.so librccl-net.so   # or vice versa" >&2
  echo "  • If symlinks point into the container (e.g. /opt/aws-ofi-rccl), run this check *inside*" >&2
  echo "    the container (or bind-mount that path) so the target exists at runtime." >&2
  exit 1
fi


# Optional: smoke-test dlopen so we fail early on missing deps
if command -v python3 >/dev/null 2>&1; then
    echo ============================================================================
    python3 - <<'PY'
import ctypes, sys, os
libs = []
# Take names from env to keep the bash search result ordering if you want;
# otherwise, just retest by names—bash already printed locations above.
for name in ("librccl-net-ofi.so", "librccl-net.so"):
    # try LD paths implicitly first; if that fails, fall back to explicit search
    try:
        ctypes.CDLL(name)  # rely on loader search path
        print(f"OK[dlopen]: {name}")
    except OSError as e:
        print(f"❌ dlopen failed for {name}: {e}", file=sys.stderr)
        sys.exit(1)
PY
    echo ============================================================================
fi

# --- end RCCL plugin sanity ---------------------------------------------------

