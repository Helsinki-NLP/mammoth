# Module loader for MAMMOTH (c) 2025 Anssi Yli-Jyrä; CC-NC-BY
echo module-loads.sh...
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }

# find this file's directory (absolute, symlinks resolved):
if [ -n "${BASH_SOURCE-}" ]; then
  THIS=${BASH_SOURCE[0]}                     # bash (sourced or executed)
elif [ -n "${ZSH_VERSION-}" ]; then
  eval 'THIS=${(%):-%N}'                     # zsh (sourced or executed)
elif (eval 'test -n "${.sh.file-}"') 2>/dev/null; then
  eval 'THIS=${.sh.file}'                    # ksh93
else # POSIX does not support directory detection of sourced files
  echo 1>&2 "Unsupported shell. Please use bash, ksh93 or zsh."
  exit 2
fi
MYDIR=$(cd -P -- "$(dirname -- "$THIS")" && pwd) || { echo "cannot resolve MYDIR" >&2; return 1 2>/dev/null || exit 1; }
HELPER=$(cd -P -- "$MYDIR/.." && pwd) || { echo "cannot resolve BASE" >&2; is_sourced && return 1 || exit 1; }
if [ -n "${DEBUG-}" ]; then
    echo "This file is in the directory $MYDIR"
    echo "The helper functions' base is $HELPER"
fi

. $MYDIR/common.sh
# require_vars JOB_NODE_KIND || { is_sourced && return 1 || exit 1; }
check_under "$HELPER" \
	    modules/pytorch-rocm-mammoth/6.0.lua \
	    modules/pytorch-rocm-mammoth/load-pytorch-rocm-mammoth.txt \
	    wrappers/python || { is_sourced && return 1 || exit 1; }

module --force purge

if [[ "${SYSTEM:-}" == "lumi" ]]; then

    [ -n "${DEBUG:-}" ] && echo ...use module paths...
    module -q use /appl/local/containers/ai-modules # AI-bindings
    module    use "$HELPER/modules"                 # pytorch-rocm-mammoth

    [ -n "${DEBUG:-}" ] && echo ...CrayEnv...
    module -q load CrayEnv

    [ -n "${DEBUG:-}" ] && echo ...LUMI...
    module -q load LUMI

    # C = LUMI-C CPU compute nodes (default)
    # G = LUMI-G AMD GPU compute nodes
    # L = Works on login & compute
    NODE_KIND=G 
    [ -n "${DEBUG:-}" ] && echo ...partition/${NODE_KIND}...
    module -q load partition/${NODE_KIND}

    [ -n "${DEBUG:-}" ] && echo "...systools..."
    module -q load systools                  # 'tree', etc. (optional)

    if [ -n "${DEBUG-}" ]; then
	echo "...singularity-AI-bindings..."
	echo "   The 'singularity-AI-bindings' module is the glue that makes PyTorch (and your MAMMOTH"
        echo "   training) run inside the supported ROCm container with the right"
	echo "   Slingshot/CXI networking and GPU libraries. Without it, things often"
	echo "   “work” but are slow/fragile (TCP fallback, plugin not found, odd CXI"
	echo "   warnings)."
	# LUMI provides containers for AI applications in /appl/local/containers,
	# in the sif-images, tested-containers and easybuild-sif-images subdirectories.
	#
	# These containers require access to the Slingshot network for good RCCL and MPI
	# performance, and most users also want transparent access to their files.
	# 
	# The bindings set up by this module take care of this.
	# 
	# Note that the precise bindings may be different for some containers, and may
	# also change over time as the MPI and libfabric libraries on the system change.
	# This module is certainly not a generic solution for all containers, but worked
	# for the PyTorch, Tensorflow, JAX, rocm and mpi4py containers in
	# /appl/local/containers when developed in early 2025.
	# 
	# The module should not be used with any other container binding module.
	echo
	echo "The following shows that these variables are not yet set when we enter to the module"
	echo "   CXI_FORK_SAFE             : ${CXI_FORK_SAFE-}"
	echo "   FI_CXI_DISABLE_CQ_HUGETLB : ${FI_CXI_DISABLE_CQ_HUGETLB-}"
	echo -n "   SINGULARITY_BIND          : "
        printf '%s\n' "${SINGULARITY_BIND-}" \
            | sed -E ' s/,/,\n                               /g' | sed '/^$/d'
	echo
	echo
	module -q load singularity-AI-bindings   # Needed for AI container bindings
	echo
	echo "The following shows that these variables have been set by the module"
	echo "   CXI_FORK_SAFE             : ${CXI_FORK_SAFE-}"
	echo "   FI_CXI_DISABLE_CQ_HUGETLB : ${FI_CXI_DISABLE_CQ_HUGETLB-}"
	echo -n "   SINGULARITY_BIND          : "
        printf '%s\n' "${SINGULARITY_BIND-}" \
            | sed -E ' s/,/,\n                               /g' | sed '/^$/d'
	echo
	echo
    else
	module -q load singularity-AI-bindings   # Needed for AI container bindings
    fi

    [ -n "${DEBUG:-}" ] && echo "loading pytorch-rocm-mammoth..."
    if [ -n "${DEBUG:-}" ]; then 
	echo
	echo "The following shows that these variables are not yet set when we enter to the module"
	print_gpu_env
	echo
	module    load pytorch-rocm-mammoth      # Lazy PyTorch (ROCm) module
	echo
	print_gpu_env
	echo
    else
	module    load pytorch-rocm-mammoth      # Lazy PyTorch (ROCm) module
    fi	
	
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

if [ -n "${DEBUG:-}" ]; then
    echo
    echo "List of active modules:"
    module --redirect list 2>&1 |egrep '\)'
    echo
fi

