#!/usr/bin/env -S -u BASH_ENV bash --noprofile --norc
echo running task-wrapper.sh...  1>&2

THIS="${BASH_SOURCE[0]}"   # THIS is the wrapper file path (argv0 label to show in ps/top).
BASENAME=$(basename "$0")  # BASENAME becomes the name you invoked the wrapper as (e.g. python, pip).
THISDIR=$(cd -P -- "$(dirname -- "$THIS")" && pwd) || { echo "cannot resolve THISDIR" >&2; return 1 2>/dev/null || exit 1; }
BASEDIR=$(cd -P -- "$THISDIR/.." && pwd)           || { echo "cannot resolve BASEDIR" >&2; return 1 2>/dev/null || exit 1; }

echo "THIS    : $THIS"     1>&2 # helper/bin/slurm/task-wrapper.sh
echo "THISDIR : $THISDIR"  1>&2 # helper/bin/slurm
echo "BASEDIR : $BASEDIR"  1>&2 # helper/bin

source $BASEDIR/../install/common.sh
require_vars SLURM_PROCID SLURM_LOCALID SLURM_NODEID RUN_SCRIPT RUN_ARGS

DEV="GPU (${SLURM_NODEID},${SLURM_LOCALID}) "

###################################################
#             SOFTWARE STACK LOADER               #
###################################################

echo $DEV "Loading the software stack at node ${SLURM_NODEID}" 1>&2
if [ "${SLURM_NODEID:-}" = 0 ]; then
    source $BASEDIR/../install/module-loads.sh 
else
    source $BASEDIR/../install/module-loads.sh 2>/dev/null
fi

# 1) Loads standard modules:
#
#    CrayEnv
#    LUMI
#    partition/G
#    singularity-AI-bindings from /appl/local/containers/ai-modules
#    systool
# 
# 2) Loads module pytorch-rocm-mammoth/6.0 containing a singularity file with pytorch/rocm 
#
# $HELPER                  : /pfs/lustrep1/projappl/project_462000964/members/aylijyra/git/mammoth/helper
# Real binary              : $HELPER/wrappers/python
# SING_IMAGE               : $HELPER//appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif \
# SING_FLAGS                 -B /bin/ip:/bin/ip \
#                            -B /usr/lib64/libmnl.so.0:/usr/lib64/libmnl.so.0 \
#                            -B /opt/cray/libfabric/1.15.2.0/bin/fi_info:/bin/fi_info \
#                            -B /usr/lib64/libcurl.so.4:/usr/lib/libcurl.so.4 \
#                            -B /users:/users \
#                            -B /appl/local/csc/soft/ai/bin\
#                            -B /appl/lumi \
#                            -B /var/lib/project_info \
#
#
# 3) The pytorch-rocm-mammoth/6.0 module sets the ROCm related paths and env variables
# 
# 
#   PATH                          : /pfs/lustrep1/projappl/project_462000964/members/aylijyra/git/mammoth/helper/wrappers \
#                                   /appl/lumi/SW/LUMI-24.03/common/EB/systools/24.03-2/bin \
#                                   /appl/lumi/SW/LUMI-24.03/common/EB/syslibs/24.03-static-ncurses-6.5/bin \
#                                   /appl/lumi/SW/LUMI-24.03/common/EB/buildtools/24.03/bin \
#                                   /opt/cray/libfabric/1.15.2.0/bin \
#                                   /usr/local/bin \
#                                   /usr/bin \
#                                   /bin \
#                                   /usr/lib/mit/bin
#   LD_LIBRARY_PATH               : /appl/lumi/SW/LUMI-24.03/common/EB/syslibs/24.03-static-ncurses-6.5/lib \
#                                   /appl/lumi/SW/LUMI-24.03/common/EB/buildtools/24.03/lib \
#                                   /opt/cray/libfabric/1.15.2.0/lib64
#
#   NCCL_SOCKET_IFNAME (hsn)      : hsn0,hsn1,hsn2,hsn3 (-> use only high speed network)
#   MIOPEN_DISABLE_CACHE (1)      : 1 (disabled; enable later)
#   MIOPEN_USER_DB_PATH ()        :  (disabled; enable later)
#   CXI_FORK_SAFE (1)             : 1 (CXI forksafety for multi-node)
#   CXI_FORK_SAFE_HP (1)          : 1
#   FI_CXI_DISABLE_CQ_HUGETLB (1) : 1 (for for safety)
#   NCCL_ENABLE_DMABUF_SUPPORT (1): 1 (set in pytorch)
#   SLURM_MPI_TYPE (pmix)         : pmi2 (leave unset; prefer PMIx over PMI2)
#   FI_PROVIDER (cxi)             : cxi (force libfabric to Slingshot)
#   FI_HMEM (rocr)                : rocr (enable ROCr GPU memory registry)
#   FI_LOG_LEVEL (warn)           : warn (quiet CXI/libfabric logs)
#   FI_LOG_PROV (cxi)             : cxi (only CXI provider logs)
#   RCCL_ENABLE_OFI               :  (do not set this; prevents auto detect)
#
# 4) Loads venv containing mammoth-specific requirements
#
# VENV_DIR                 : /pfs/lustrep1/projappl/project_462000964/members/aylijyra/git/mammoth/helper/venv
#
# The environmet has been created with 'build-venv.sh'
# and possibly extended.  The venv-local packages are:
#
#     ConfigArgParse==1.7.1
#     einx==0.3.0
#     frozendict==2.4.6
#     loguru==0.7.3
#  

###################################################
#          NO GPU VISIBILITY CONSTRAINTS          #
#            Mammoth does not use these           #
###################################################
# Don’t set any *VISIBLE_DEVICES before srun.  Never export
# *VISIBLE_DEVICES in the batch script before srun. Outside srun
# there’s no SLURM_LOCALID, and you’ll end up exposing all GPUs to
# every task (the “bad” case you saw in the LUMI AI course) .
#
# Do not do this (!):
#  export HIP_VISIBLE_DEVICES=$SLURM_LOCALID # only correct inside srun
#  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5   # 0,1,2,3,4,5 lines are fragile/incorrect
#  export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5   # 0,1,2,3,4,5 lines are fragile/incorrect
# LUMI has 8 GCDs; plus you shouldn’t hardcode).
#
# Pick the right env var names once
# case "$SYSTEM" in
#   lumi)           GPU_ENV1=HIP_VISIBLE_DEVICES;  GPU_ENV2=ROCR_VISIBLE_DEVICES ;;
#   puhti|mahti|*)  GPU_ENV1=CUDA_VISIBLE_DEVICES; GPU_ENV2= ;;
# esac
# Derive layout from Slurm: 1 task per GPU vs 1 launcher per node
# TPN=${SLURM_NTASKS_PER_NODE:-1}
# GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_JOB_GPUS_PER_NODE:-${GPUS_PER_NODE:-1}}}
# GPUS_PER_TASK=${SLURM_GPUS_PER_TASK:-1}
# if [ "$TPN" -eq "$GPUS_PER_NODE" ] && [ "$GPUS_PER_TASK" -eq 1 ]; then
#   # ----- Pattern: 1 task per GPU -----
#   # Each Slurm task gets exactly one GPU.  Make only that GPU visible
#   # based on SLURM_LOCALID (0..GPUS_PER_NODE-1).
#   export "$GPU_ENV1"="$SLURM_LOCALID"
#   [[ -n "${GPU_ENV2:-}" ]] && export "$GPU_ENV2"="$SLURM_LOCALID"
#   TASK_PATTERN=1-task-per-GPU
# else
#   # ----- Pattern: 1 launcher per node (Mammoth, torchrun, etc.) -----
#   # The launcher (python, torchrun, Mammoth) spawns per-GPU workers and
#   # sets device/ranks internally.  Expose all local GPUs; do not restrict.
#   [[ -n "${GPU_ENV1:-}" ]] && unset "$GPU_ENV1"
#   [[ -n "${GPU_ENV2:-}" ]] && unset "$GPU_ENV2"
#   TASK_PATTERN=1-task-per-NODE
# fi
# MAMMOTH use one launcher per node (the "else" branch).
# Expose all local GPUs to the launcher, no VISIBILITY.
#
echo $DEV ============================================== 1>&2
echo $DEV " SYSTEM                    : $SYSTEM"         1>&2
echo $DEV " TASK_PATTERN              : $TASK_PATTERN"   1>&2
echo $DEV " CUDA_VISIBLE_DEVICES      : ${CUDA_VISIBLE_DEVICES-}"  1>&2
echo $DEV " HIP_VISIBLE_DEVICES       : ${HIP_VISIBLE_DEVICES-}"   1>&2
echo $DEV " ROCR_VISIBLE_DEVICES      : ${ROCR_VISIBLE_DEVICES-}"  1>&2

###################################################
#   OpenMP settings tied to SLURM_CPUS_PER_TASK   #
#    potentially counterproductive for Mammoth    #
###################################################
# Match OpenMP threads to cpus-per-task (data loaders often separate; this is safe)
#
# if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "${SLURM_CPUS_PER_TASK}" -gt 0 ]]; then
#   export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
#   export OMP_PROC_BIND="${OMP_PROC_BIND:-close}"
#   export OMP_PLACES="${OMP_PLACES:-cores}"
# fi
#
# The “blindly set OMP_NUM_THREADS = SLURM_CPUS_PER_TASK” is not a good match for
# Mammoth’s DDP-style, multi-process design. Mammoth’s parallelism is multi-process,
# not OpenMP.  If you want, you can use conservative defaults:
#
#   : "${OMP_NUM_THREADS:=1}"
#   : "${OMP_PROC_BIND:=close}"
#   : "${OMP_PLACES:=cores}"
#   export OMP_NUM_THREADS OMP_PROC_BIND OMP_PLACES
# 
# echo $DEV ==============================================  
# echo $DEV " OMP_NUM_THREADS           : $OMP_NUM_THREADS"
# echo $DEV " OMP_PROC_BIND             : $OMP_PROC_BIND"
# echo $DEV " OMP_PLACES                : $OMP_PLACES"

###################################################
#            PYTORCH's WORLD                      #
#            not consulted by MAMMOTH             #
###################################################
# PyTorch honors env vars: 
#    MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK, NODE_RANK
#
# RANK or LOCAL_RANK not available since MAMMOTH spawns GPUs
#
NODE_RANK="${SLURM_NODEID:-}"    # used by --node_rank
WORLD_SIZE="${SLURM_NTASKS:-}"   # now just for printing
# But MAMMOTH does not read the WORLD_SIZE environment
# variable when it initializes distributed; it uses opts.world_size
# from its own config and passes it directly to init_process_group.
# Setting export WORLD_SIZE="$SLURM_NTASKS" does not affect MAMMOTH's init.

###################################################
#            MAMMOTHS WORLD                       #
#            not consulted by MAMMOTH             #
###################################################
RUNTIME="python"
RUNTIME_ARGS=("-u")
require_vars MASTER_PORT MASTER_ADDR SLURM_NODEID
POST_ARGS=(--node_rank "${NODE_RANK}" --master_port "${MASTER_PORT}" \
		       --master_ip "${MASTER_ADDR}")
echo $DEV ==============================================    1>&2
echo $DEV " SLURM_JOB_NAME              : $SLURM_JOB_NAME"  1>&2
echo $DEV " SLURM_NODEID  (--node_rank) : $SLURM_NODEID"    1>&2
echo $DEV " SLURM_NTASKS (--world_size) : $SLURM_NTASKS"    1>&2
echo $DEV " MASTER_PORT (--master_port) : $MASTER_PORT"     1>&2
echo $DEV " MASTER_ADDR (--master_addr) : $MASTER_ADDR"     1>&2
echo $DEV ==============================================    1>&2
echo $DEV " RUNTIME           : $RUNTIME"                   1>&2
echo $DEV " RUNTIME_ARGS      : ${RUNTIME_ARGS[@]}"         1>&2
echo $DEV " COMMAND LINE ARGS : $RUN_SCRIPT $RUN_ARGS"      1>&2
echo $DEV " POST_ARGS         : ${POST_ARGS[@]}"            1>&2


###################################################
#           MIOPEN cache per job + node           #
#             Keep this, it’s useful.             #
###################################################
# On LUMI-G (ROCm), PyTorch’s convolutions, layernorms, etc. go through
# MIOpen under the hood.  MIOpen uses a kernel cache to store tuned
# configs; by default that can end up in a shared place and cause:
# - contention,
# - permission issues,
# - or at least noisy cross-job interference.
#
# The following code
# - Uses per-job, per-node cache directory under $TMPDIR (local scratch if available),
# - Avoids collisions across users/jobs/nodes (JOB_ID + NODEID),
# - Keeps this logic inside the srun context (so SLURM_NODEID exists).
#
# Keep this code after module loads since the modules can
# reset these settings due to ignorance of the JOB_ID + NODEID.
# 
# That’s exactly the sort of pattern ROCm users are encouraged to use
# on big HPC systems. Mammoth itself doesn’t know or care about these
# env vars, but the ROCm libraries that Mammoth uses via PyTorch do.
#
# Per-node cache needs SLURM_NODEID (useful on ROCm; harmless elsewhere) ---
BASE=${TMPDIR:-/tmp}   # uses the job’s local scratch if your system provides it.
export MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$BASE/$USER-miopen-${SLURM_JOB_ID:-0}-${SLURM_NODEID:-0}}"
#
# Including both SLURM_JOB_ID and SLURM_NODEID avoids collisions across jobs and nodes.
# Keep this inside srun so SLURM_NODEID exists.
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR:-$MIOPEN_USER_DB_PATH}"
mkdir -p -- "$MIOPEN_USER_DB_PATH" >/dev/null 2>&1 || true
#
echo $DEV ==============================================          1>&2
echo $DEV " MIOPEN_USER_DB_PATH       : $MIOPEN_USER_DB_PATH"     1>&2
echo $DEV " MIOPEN_CUSTOM_CACHE_DIR   : $MIOPEN_CUSTOM_CACHE_DIR" 1>&2

###################################################
#             ADDITIONAL RCCL VARIABLES           #
###################################################
#
if [ "${COMMS_DEBUG:-0}" = 1 ]; then
    # RCCL logging
    export RCCL_DEBUG=INFO                 # WARN < INFO < TRACE
    export RCCL_DEBUG_SUBSYS=INIT,COLL,NET # INIT < COLL < NET < INIT,COLL,NET < TUNING/GRAPH/ENV/ALL
    # With RCCL_DEBUG enabled, all RCCL logs will land in the --error specified file
    # alongside your Python errors. That’s often good enough for a first pass.
    export TORCH_DISTRIBUTED_DEBUG=DETAIL  # optional
    # export ROCPROFILER_DISABLE=1         # Optionally: quiet rocprofiler if it becomes noisy
    # export PLUGIN_DIR=base/mammoth-helper/helper/lib  # This has symlinks to /opt/aws-ofi-rccl/librccl-net.so
    # export LD_LIBRARY_PATH=base/mammoth-helper/helper/lib:$LD_LIBRARY_PATH
    
    echo $DEV ==============================================             1>&2
    echo $DEV " RCCL_DEBUG                : ${RCCL_DEBUG:-}"             1>&2
    echo $DEV " RCCL_DEBUG_SUBSYS         : ${RCCL_DEBUG_SUBSYS:-}"      1>&2
    echo $DEV " TORCH_DISTRIBUTED_DEBUG   : ${TORCH_DISTRIBUTED_DEBUG:-}"  1>&2
fi

###################################################
#              SMI MONITORING                     #
###################################################

.  $THISDIR/smi-monitor.sh

echo $DEV ============================================== 1>&2
echo $DEV " MON_PID       : ${MON_PID:-}"                1>&2
echo $DEV " MON_OUT       : ${MON_OUT:-}"                1>&2
echo $DEV ============================================== 1>&2

###################################################
#        RUNNING THE TASK OF THE CURRENT NODE     #
###################################################

exec ${RUNTIME} ${RUNTIME_ARGS[@]} ${RUN_SCRIPT} ${RUN_ARGS} ${POST_ARGS[@]}
