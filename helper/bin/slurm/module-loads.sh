
echo module-loads...
# (c) 2025 Anssi Yli-Jyrä, CC-BY

# usage: source module-loads.sh 
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }

LOCAL_PROJHOME=$PROJHOME
# LOCAL_PROJHOME=/project/$ACCOUNT/members/$USER

# Manual override: export NODE_KIND=gpu (or cpu / login) if you ever need to force a stack.
# Detection logic order:
#  1. Respect NODE_KIND if set.
#  2. Use SLURM_JOB_PARTITION hints when available.
#  3. Fall back to device presence (/dev/kfd or /dev/dri/renderD* ⇒ GPU).
#  4. If in a Slurm job with no GPU devices ⇒ CPU compute.
#  5. Otherwise ⇒ login node.
# The checks are conservative and won’t require extra tooling before modules are loaded.
#
# Auto-sense node type (login / CPU compute / GPU compute) on LUMI and Puhti
# and load a sensible module stack. On Puhti, use the standard 'pytorch' module.

detect_node_kind () {
  # 1) Manual override (normalize to lowercase, validate, and echo)
  if [[ -n "${NODE_KIND:-}" ]]; then
    local override="${NODE_KIND,,}"   # bash lowercase
    case "$override" in
      gpu|cpu|login) echo "$override"; return ;;
      *) echo "Invalid NODE_KIND='$NODE_KIND' (use gpu|cpu|login)" >&2; return 1 ;;
    esac
  fi

  # 2) From SLURM partition (safer than '*g*' / '*c*')
  if [[ -n "${SLURM_JOB_PARTITION:-}" ]]; then
    local part="${SLURM_JOB_PARTITION,,}"
    if [[ "$part" == *gpu* || "$part" =~ (^|[^a-z])g($|[^a-z]) ]]; then
      echo "gpu"; return
    elif [[ "$part" == *cpu* || "$part" =~ (^|[^a-z])c($|[^a-z]) ]]; then
      echo "cpu"; return
    fi
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
    module -q use /appl/local/containers/ai-modules     # AI-bindings
    module -q use "${LOCAL_PROJHOME:-$HOME}/bin/modules" # pytorch-rocm-mammoth (fallback to $HOME if unset)

    # --- Base env (recommended by CSC; safe on all nodes) ---
    module -q load CrayEnv
    module -q load LUMI

    NODE_KIND="$(detect_node_kind)"
    echo "LUMI node kind detected: ${NODE_KIND} (SLURM_JOB_PARTITION='${SLURM_JOB_PARTITION:--}')"

    # --- Load partition stack based on detection -----------------------------
    case "$NODE_KIND" in
        login)
            module -q load partition/L      # Works on login & compute
	    module -q load systools         # 'tree', etc. (optional)
            ;;
        cpu)
            module -q load partition/C      # LUMI-C CPU compute nodes
            ;;
        gpu)
            module -q load partition/G      # LUMI-G AMD GPU compute nodes
            ;;
        *)
            echo "Could not determine node kind. Defaulting to partition/L."
            module -q load partition/L
            ;;
    esac

    # --- Your workload-specific modules --------------------------------------
    module -q load singularity-AI-bindings   # Needed for AI container bindings
    module -q load pytorch-rocm-mammoth      # Lazy PyTorch (ROCm) module


elif [[ "${SYSTEM:-}" == "puhti" ]]; then
    # --- Common module paths (Puhti) ---
    # Keep these generic; they will only load if present.
    module -q use /appl/local/containers/ai-modules 2>/dev/null || true
    module -q use "${LOCAL_PROJHOME:-$HOME}/bin/modules" 2>/dev/null || true

    # --- Base env (Puhti typically uses CSC defaults; load if available) ---
    load_first_available csc csc-env puhti

    NODE_KIND="$(detect_node_kind)"
    echo "Puhti node kind detected: ${NODE_KIND} (SLURM_JOB_PARTITION='${SLURM_JOB_PARTITION:--}')"

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
    exit 1
fi

echo ============================================================================
module --redirect list |egrep '\)'
echo ============================================================================
