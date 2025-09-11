(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
echo sbatch-tail...
set -euo pipefail  # make failures fatal and undefined vars errors (optional)

# These come from the .profile or otherwise from the environment.
: "${SYSTEM:?❌ SYSTEM is not set}"
: "${PROJHOME:?❌ PROJHOME is not set}"            
: "${PROJDATA:?❌ PROJDATA is not set}"            
: "${ACCOUNT:?❌ ACCOUNT is not set}"              
: "${SLURM_JOB_NAME:?❌ SLURM_JOB_NAME is not set}"              
echo ==============================================
echo " SYSTEM (from .profile)    : $SYSTEM"
echo " ACCOUNT                   : $ACCOUNT"
echo " USER                      : $USER"
echo " PROJHOME (from .profile)  : $PROJHOME"
echo " PROJDATA (from .profile)  : $PROJDATA"
echo " SLURM_JOB_NAME            : $SLURM_JOB_NAME"
echo ==============================================

export SLURM="$PROJHOME/bin/slurm"
f="$SLURM/distributed-setup.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
f="$SLURM/sanity-checks.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
f="$PROJDATA/exp/$SLURM_JOB_NAME/job-setup.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
f="$SLURM/module-loads.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
f="$SLURM/comms-setup.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
f="$SLURM/local-setup.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
f="$SLURM/task-wrapper.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }

source $SLURM/distributed-setup.sh
# These come from $SLURM/distributed-setup.sh
: "${CPUS_PER_TASK:?❌ CPUS_PER_TASK is not set}"
: "${GPUS_PER_NODE:?❌ GPUS_PER_NODE is not set}"
: "${DISTR_OPS:?❌ DISTR_OPS is not set}"          # this will be used in srun command

source $PROJDATA/exp/$SLURM_JOB_NAME/job-setup.sh # job configurations define software/data
# These come from job-setup.sh
: "${MAMMOTH:?❌ MAMMOTH is not set}"              # $PROJHOME/venv/mammoth
: "${MASTER_ARGS:?❌ MASTER_ARGS is not set}"      # 
: "${TRAIN_SCRIPT:?❌ TRAIN_SCRIPT is not set}"    #
: "${TRAIN_ARGS:?❌ TRAIN_ARGS is not set}"        #
: "${PATTERN:?❌ PATTERN is not set}"              #
: "${EXP_ID:?❌ EXP_ID is not set}"                # Must be equivalent to SLURM_JOB_NAME
: "${GUARD_TIME:?❌ GUARD_TIME is not set}"        #
: "${GUARD_MAX_NODES:?❌ GUARD_MAX_NODES not set}" #

source $SLURM/sanity-checks.sh             # check sbatch sanity
# This comes from $SLURM/sanity-checks.sh
: "${SANITY_CHECKS_OK:?❌ SANITY_CHECKS_OK is not set}"

source $SLURM/module-loads.sh              # load modules

f="$MAMMOTH/bin/activate"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
source $MAMMOTH/bin/activate                       # enter the environment

source $SLURM/comms-setup.sh
# These variables come from $SLURM/comms-setup.sh
: "${MASTER_PORT:?❌ MASTER_PORT is not set}"
: "${MASTER_ADDR:?❌ MASTER_ADDR is not set}"
: "${FI_PROVIDER:?❌ FI_PROVIDER is not set (e.g. cxi)}"
: "${FI_HMEM:?❌ FI_HMEM is not set (e.g., rocr)}"
: "${FI_LOG_LEVEL:?❌ FI_LOG_LEVEL is not set (e.g., warn)}"
: "${FI_LOG_PROV:?❌ FI_LOG_PROV is not set (e.g., cxi)}"
: "${NCCL_SOCKET_IFNAME:?❌ NCCL_SOCKET_IFNAME is not set (e.g., hsn0)}"

# This uses a job-generic task-wrapper.sh to run python scripts either
# with python or torchrun; See job-setup.sh for details

# For Slurm, srun --unbuffered (you already have this in DISTR_OPS)
# reduces output buffering between tasks and the collector. Handy for
# debugging, but it adds overhead—use sparingly on big jobs.

# export DIAG=
# export DIAG=--overlap rocm-smi

# I have to pass the array as arguments as bash cannot export arrays
srun $DISTR_OPS $SLURM/task-wrapper.sh ${RUN_SCRIPT} ${RUN_ARGS[@]}

