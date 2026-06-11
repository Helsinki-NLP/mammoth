#!/bin/bash
changecom()dnl # disables m4 comments so #SBATCH lines are expanded; this line is supressed

#SBATCH --partition=dev-g
#SBATCH --mem-per-gpu=60G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --output=__LOGDIR__/job%j.out
#SBATCH --error=__LOGDIR__/job%j.err
#SBATCH --job-name=__JOB_NAME__
#SBATCH --account=__ACCOUNT__
#SBATCH --chdir=__MODELDIR__

# This have to be give as command line arguments:
##SBATCH --time=__TIME__
##SBATCH --ntasks=__NTASKS__
##SBATCH --gpus=__GPUS__
##SBATCH --nodes=__NODES__

set -euo pipefail

export ACCOUNT=__ACCOUNT__
export DISKPROJECT=__DISKPROJECT__
export SIF=__SIF__
export SELFDIR=__SELFDIR__
export OUTDIR=__OUTDIR__
export MAKESCRIPT=__MAKESCRIPT__
export LOGDIR=__LOGDIR__

export BINDIR="${SELFDIR}"
export INF_DISTR="${SELFDIR}/inf_distr.sh"
export INF_WRAPPER="${SELFDIR}/inf_wrapper.sh"
export CALLS_FILE="${OUTDIR}/calls.out"
export INF_SBATCH="${OUTDIR}/inf.sbatch"

[[ -f "${INF_DISTR}"   ]] || { echo "Error: The distributor not found in ${INF_DISTR}" >&2; exit 1; }
if [ -n "${SLURM_JOBID:-}" ]; then
    export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_DEBUG=INFO
    echo "Launching srun..."
    trap 'rm -f "__INF_FLAG__"; touch "__INF_DONE__"' EXIT
    srun /usr/bin/singularity exec \
	 --env NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" \
	 --env NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL}" \
	 --env NCCL_DEBUG="${NCCL_DEBUG}" \
         -B  "/scratch/${DISKPROJECT}:/scratch/${DISKPROJECT}:rw" \
	 "${SIF}" "${INF_WRAPPER}"
else
    bash "${INF_DISTR}"
fi
