check_under() {
  local base="$1"; shift
  local name f missing=0
  [ -n "${DEBUG:-}" ] && echo "Testing software integrity under $base:"
  for name in "$@"; do
      f="$base/$name"
      [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; missing=1; }
      [ -n "${DEBUG:-}" ] && echo "- found file $name" 
  done
  # (( missing == 0 ))
  (( missing == 0 )) || exit 1
}

log() {
    printf '%s %s\n' "[$BASENAME]" "$*";
}

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

# load first module that exists (Lmod)
load_first_available () {
    for m in "$@"; do
        if module -q is-avail "$m"; then
            module -q load "$m"
            [ -n "${DEBUG:-}" ] && echo "Loaded: $m"
            return 0
        fi
    done
    return 1
}

print_gpu_env () {
    SING_FLAGS=${SING_FLAGS-}
    PATH=${PATH-}
    LD_LIBRARY_PATH=${LD_LIBRARY_PATH-}
    
	echo "The following shows that these variables have been set by the module"
	echo "   SING_IMAGE                    : ${SING_IMAGE-}"
	if [[ -n "${SING_FLAGS}" ]]; then
	    echo -n "   SING_FLAGS                    : "
	    printf '%s\n' "${SING_FLAGS-}" \
		| sed -E ' s/[[:space:]]+-/\n                                -/g' | sed '/^$/d'
	fi
	if [[ -n "${PATH}" ]]; then
	    echo -n "  +PATH (helper)                 : "
	    printf '%s\n' "${PATH-}" \
		| sed -E ' s/:/\n                                 : /g' | sed '/^$/d' | sed '/^ PYTHONPATH *: *$/d' 
	fi
	echo -n "  +LD_LIBRARY_PATH (helper)      : "
	printf '%s\n' "${LD_LIBRARY_PATH-}" \
	    | sed -E ' s/:/\n                                 : /g' | sed '/^$/d' | sed '/^ PYTHONPATH *: *$/d'
	echo
	echo "   NCCL_SOCKET_IFNAME (hsn)      : ${NCCL_SOCKET_IFNAME-} (-> use only high speed network)"
	echo "   MIOPEN_DISABLE_CACHE (1)      : ${MIOPEN_DISABLE_CACHE-} (disabled; enable later)"
	echo "   MIOPEN_USER_DB_PATH ()        : ${MIOPEN_USER_DB_PATH-} (disabled; enable later)"
	# Don’t disable caching globally unless you must. Better:
	# enable cache but put it on node-local tmp so you avoid
	# home/project FS churn and still get speedups.
	# Do node-locally:
	#    export MIOPEN_DISABLE_CACHE=0
	#    export MIOPEN_USER_DB_PATH="${TMPDIR:-/tmp}/miopen-userdb-$USER"
	#    export MIOPEN_CUSTOM_CACHE_DIR="${TMPDIR:-/tmp}/miopen-cache-$USER"
	echo "   CXI_FORK_SAFE (1)             : ${CXI_FORK_SAFE-} (CXI forksafety for multi-node)"
	echo "   CXI_FORK_SAFE_HP (1)          : ${CXI_FORK_SAFE_HP-}"
	echo "   FI_CXI_DISABLE_CQ_HUGETLB (1) : ${FI_CXI_DISABLE_CQ_HUGETLB-} (for for safety)"
	echo "   NCCL_ENABLE_DMABUF_SUPPORT (1): ${NCCL_ENABLE_DMABUF_SUPPORT-} (set in pytorch)"
	echo "   SLURM_MPI_TYPE (pmix)         : ${SLURM_MPI_TYPE-} (leave unset; prefer PMIx over PMI2)"
	echo "   FI_PROVIDER (cxi)             : ${FI_PROVIDER-} (force libfabric to Slingshot)"
	echo "   FI_HMEM (rocr)                : ${FI_HMEM-} (enable ROCr GPU memory registry)"
	echo "   FI_LOG_LEVEL (warn)           : ${FI_LOG_LEVEL-} (quiet CXI/libfabric logs)"
	echo "   FI_LOG_PROV (cxi)             : ${FI_LOG_PROV-} (only CXI provider logs)"
	echo "   RCCL_ENABLE_OFI               : ${RCCL_ENABLE_OFI-} (do not set this; prevents auto detect)"
	echo "   NCCL_SOCKET_IFNAME (hsn0,..)  : ${NCCL_SOCKET_IFNAME-} (keep high-speed)"
	# echo "   NCCL_NET_GDR_LEVEL            : ${NCCL_NET_GDR_LEVEL-} (no effect on RCCL/ROCm MI250; safe to omit)"
	# python $HELPER/bin/slurm/torch-test.py would also print on LUMI
	#   sys.executable            : /opt/miniconda3/envs/pytorch/bin/python
	#   sys.version               : 3.12.11
	#   torch                     : 2.7.1+rocm6.2.4
	#   HIP version               : 6.2.41134-65d174c3e
	#   CUDA version              : None
	#   CUDA avail?               : False    
}
