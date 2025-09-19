echo module-loads.sh...
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
is_sourced()  { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
require_vars() {
  local v
  for v; do
    # ${!v-} expands to empty if unset (safe with set -u)
    if [[ -z "${!v-}" ]]; then
      printf '❌ %s must be set\n' "$v" >&2
      return 1
    fi
  done
}
require_vars JOB_NODE_KIND || { is_sourced && return 1 || exit 1; }

check_under() {
  local base="$1"; shift
  local name f missing=0
  echo "Testing software integrity under $base:"
  for name in "$@"; do
      f="$base/$name"
      [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; missing=1; }
      echo "- found file $name" 
  done
  (( missing == 0 )) 
}

# --- find this file's directory (absolute, symlinks resolved) ---
if [ -n "${BASH_SOURCE-}" ]; then
  THIS=${BASH_SOURCE[0]}                     # bash (sourced or executed)
elif [ -n "${ZSH_VERSION-}" ]; then
  eval 'THIS=${(%):-%N}'                     # zsh (sourced or executed)
elif (eval 'test -n "${.sh.file-}"') 2>/dev/null; then
  eval 'THIS=${.sh.file}'                    # ksh93
else
  echo 1>&2 "Unsupported shell. Please use bash, ksh93 or zsh."
  exit 2
  # _src=$0                                   # POSIX sh (only reliable when executed)
fi
MYDIR=$(cd -P -- "$(dirname -- "$THIS")" && pwd) || { echo "cannot resolve MYDIR" >&2; return 1 2>/dev/null || exit 1; }
echo "This file is in the directory $MYDIR"
HELPER=$(cd -P -- "$MYDIR/../.." && pwd) || { echo "cannot resolve BASE" >&2; is_sourced && return 1 || exit 1; }
echo "The helper functions' base is $HELPER"
check_under "$HELPER" \
   bin/modules/pytorch-rocm-mammoth/6.0.lua \
   bin/modules/load-pytorch-rocm-mammoth.txt \
   bin/wrappers/python \
   images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
   lib/README.md || { is_sourced && return 1 || exit 1; }

# Manual override (required) of node kind detection:
#  export JOB_NODE_KIND=gpu (or cpu / login) if you ever need to force a stack.
# Detection logic order:
#  1. Respect JOB_NODE_KIND if set.
#  2. Use SLURM_JOB_PARTITION hints when available.
#  3. Fall back to device presence (/dev/kfd or /dev/dri/renderD* ⇒ GPU).
#  4. If in a Slurm job with no GPU devices ⇒ CPU compute.
#  5. Otherwise ⇒ login node.
# The checks are conservative and won’t require extra tooling before modules are loaded.
#
# Auto-senses the node type (login / CPU compute / GPU compute) on LUMI and Puhti
# and loads a sensible module stack. On LUMI, the pytorch is inside an custom module
# that loads a CSC-built container.  On Puhti, uses the standard 'pytorch' module.

detect_node_kind () {
  # 1) Manual override (normalize to lowercase, validate, and echo)
  if [[ -n "${JOB_NODE_KIND:-}" ]]; then
    local override="${JOB_NODE_KIND,,}"   # bash lowercase
    case "$override" in
      gpu|cpu|login) echo "$override"; return ;;
      *) echo "Invalid JOB_NODE_KIND='$JOB_NODE_KIND' (use gpu|cpu|login)" >&2;
	 is_sourced && return 1 || exit 1 ;;
    esac
  fi

  # 2) From SLURM partition
  if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
    local part="${SLURM_JOB_PARTITION:-}"
    part="${part%%,*}"; part="${part,,}"
    case "$part" in
       gputest|gpu|gpusmall|gpumedium|small-g|dev-g|debug) echo "gpu"; return ;;
    esac
  fi

  # 3) Device presence → GPU
  if [[ -e /dev/kfd ]] || compgen -G "/dev/dri/renderD*" >/dev/null || command -v nvidia-smi >/dev/null; then
    echo "gpu"; return
  fi

  # 4) In a SLURM job but no GPU devices → CPU
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "cpu"; return
  fi

  # 5) Fallback
  echo "login"
}

module --force purge

NODE_KIND="$(detect_node_kind)"  # in fact will get its value from JOB_NODE_KIND
echo "LUMI node kind detected: ${NODE_KIND} (SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:--})"

# Helper: load first module that exists (Lmod)
load_first_available () {
    for m in "$@"; do
        if module -q is-avail "$m"; then
            module -q load "$m"
            echo "Loaded: $m"
            return 0
        fi
    done
    return 1
}

if [[ "${SYSTEM:-}" == "lumi" ]]; then
    # --- Common module paths ---
    echo ...paths...
    module -q use /appl/local/containers/ai-modules # AI-bindings
    module    use "$HELPER/bin/modules"             # pytorch-rocm-mammoth
    find $HELPER/bin/modules |egrep '\.lua'

    # --- Base env (recommended by CSC; safe on all nodes) ---
    echo ...CrayEnv...
    module -q load CrayEnv
    echo ...LUMI...
    module -q load LUMI

    # --- Load partition stack based on detection -----------------------------
    case "$NODE_KIND" in
        login)
	    echo ...partition/L...
            module -q load partition/L      # Works on login & compute
            ;;
        cpu)
	    echo ...partition/C...
            module -q load partition/C      # LUMI-C CPU compute nodes
            ;;
        gpu)
	    echo ...partition/G...
            module -q load partition/G      # LUMI-G AMD GPU compute nodes
            ;;
        *)
            echo "Could not determine node kind. Defaulting to partition/L."
	    echo ...partition/L...
            module -q load partition/L
            ;;
    esac

    # --- Your workload-specific modules --------------------------------------
    echo "...systools..."
    module -q load systools                  # 'tree', etc. (optional)
    echo "...singularity-AI-bindings..."
    module -q load singularity-AI-bindings   # Needed for AI container bindings

    echo =================INPUT========================
    echo " PATH                      : $PATH"
    echo " LD_LIBRARY_PATH           : $LD_LIBRARY_PATH"
    echo " FI_PROVIDER (cxi)         : $FI_PROVIDER"
    echo " FI_HMEM (rocr)            : $FI_HMEM"
    echo " FI_LOG_LEVEL (warn)       : $FI_LOG_LEVEL"
    echo " FI_LOG_PROV (cxi)         : $FI_LOG_PROV"
    echo " PLUGIN_DIR                : $PLUGIN_DIR"
    echo " RCC_ENABLE_OFI            : $RCCL_ENABLE_OFI"
    echo " NCCL_SOCKET_IFNAME (hsn0) : $NCCL_SOCKET_IFNAME"
    echo " NCCL_NET_GDR_LEVEL        : $NCCL_NET_GDR_LEVEL"
    echo ==============================================
    
    echo "...pytorch-rocm-mammoth (uses lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif)..."
    module    load pytorch-rocm-mammoth      # Lazy PyTorch (ROCm) module

    $PROJHOME/venv/mammoth-hf/bin/python $HELPER/bin/slurm/torch-test.py

elif [[ "${SYSTEM:-}" == "puhti" ]]; then
    # --- Common module paths (Puhti) ---
    # Keep these generic; they will only load if present.
    module -q use /appl/local/containers/ai-modules 2>/dev/null || true  # error is ok
    # shellcheck source=../modules
    module -q use "$HELPER/modules"  2>/dev/null || true   # error is ok

    # --- Base env (Puhti typically uses CSC defaults; load if available) ---
    load_first_available csc csc-env puhti

    # Try multiple aliases for partition modules to be cross-cluster friendly.
    case "$NODE_KIND" in
        login) load_first_available partition/L partition/login partition/l ;;
        cpu)   load_first_available partition/C partition/cpu partition/c partition/puhti-cpu || load_first_available partition/L ;;
        gpu)   load_first_available partition/G partition/gpu partition/g partition/puhti-gpu || load_first_available partition/L ;;
        *)     echo "Could not determine node kind. Defaulting to partition/L."; load_first_available partition/L ;;
    esac

    # Workload-specific
    load_first_available singularity-AI-bindings apptainer-AI-bindings
    load_first_available pytorch
    load_first_available systools

else
    echo "module loads for ${SYSTEM:-<unset>} are not yet specified in module-load.sh"
    echo "please add them in this script!"
    { is_sourced && return 1 || exit 1; }
fi

echo ============================================================================
module --redirect list |egrep '\)'
echo ============================================================================
