echo "Starting sbatch-tail.sh at `date`..."
set -e # stops the script when encountering an error
(return 0 2>/dev/null) || { echo "❌ Please source this script instead of executing it."; exit 1; }
is_sourced()  { [[ "${BASH_SOURCE[0]}" != "$0" ]]; }
require_set() {
  local v
  for v; do
    # ${!v-} expands to empty if unset (safe with set -u)
    if [[ -z "${!v-}" ]]; then
      printf '❌ %s must be set\n' "$v" >&2
      return 1
    fi
  done
}

source base/mammoth-helper/helper/bin/slurm/1-integrity-checks.sh
# Check that scripts and variables are in place

source base/mammoth-helper/helper/bin/slurm/2-distributed-setup.sh
# Complement the #SBATCH parameters
require_set CPUS_PER_TASK GPUS_PER_NODE DISTR_OPS MASTER_PORT \
	    MASTER_ADDR MASTER_ARGS JOB_NODE_KIND || { is_sourced && return 1 || exit 1; };

echo check sanity and GUARDs of the allocations ...
source base/mammoth-helper/helper/bin/slurm/3-sanity-checks.sh
require_set SANITY_CHECKS_OK || { is_sourced && return 1 || exit 1; };

echo load modules and the virtual environment ...
source base/mammoth-helper/helper/bin/slurm/4-module-loads.sh

echo activating the virtual environment ...
source base/venv/mammoth-hf/bin/activate

echo setting the multiprocessor communication settings ...
source base/mammoth-helper/helper/bin/slurm/5-comms-setup.sh
require_set FI_PROVIDER FI_HMEM FI_LOG_LEVEL FI_LOG_PROV NCCL_SOCKET_IFNAME \
     || { is_sourced && return 1 || exit 1; } ;

echo running the srun...
echo "Starting task-wrapper.sh at `date`"
srun $DISTR_OPS base/mammoth-helper/helper/bin/slurm/6-task-wrapper.sh ${RUN_SCRIPT} ${RUN_ARGS}
echo "Finishing the whole sbatch at `date`"


