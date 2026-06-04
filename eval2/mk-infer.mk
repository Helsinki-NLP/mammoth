# -----------------------------------------------------------------------------
# Inference submission placeholders
# -----------------------------------------------------------------------------

.PHONY: clean-inference-lock inference-submit run-inference 

mk-infer: inference-submit

clean-inference-lock: dirs check-in-model-dir 
> @set -euo pipefail
> @if [ -e "$(INF_FLAG)" ]; then \
>   oldjob="$$(cat "$(INF_FLAG)")"; \
>   if squeue -h -j "$$oldjob" | grep -q .; then \
>     echo "❌ Inference job $$oldjob is still queued/running."; \
>     exit 1; \
>   fi; \
>   echo "Removing stale inference flag for job $$oldjob"; \
>   rm -f "$(INF_FLAG)"; \
> fi
> @echo "✅ Verified that no other inference job is running"


inference-plan-slurm: $(MAKESCRIPT)
> if [ ! -s "$(INFERENCE_CALLS)" ]; then \
>   echo "No inference commands to submit."; \
>   echo "Either all hypothesis outputs exist, or inference planning is still a placeholder."; \
>   exit 0; \
> fi
> echo "Planning SLURM distribution..."
> export OUTDIR="$(OUTDIR)"; export MAKESCRIPT="$(MAKESCRIPT)"; \
>   bash "$(BINDIR)/inf_distr.sh"


$(MAKESCRIPT): check-in-model-dir
	@set -eu; \
	mkdir -p "$(dir $(MAKESCRIPT))"; \
	cat > "$(MAKESCRIPT)" <<EOF
#!/bin/bash
#SBATCH --job-name=inference
#SBATCH --account=$(PROJECT)
#SBATCH --partition=small-g
#SBATCH --mem-per-gpu=16G
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=$(LOGDIR)/job%j.out
#SBATCH --error=$(LOGDIR)/job%j.err
#SBATCH --chdir=$(MODELDIR)

set -euo pipefail
echo "Launching srun..."

trap 'rm -f "$(INF_FLAG)"' EXIT
export SIF=/appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB
srun /usr/bin/singularity exec \
     --env NCCL_SOCKET_IFNAME="\${NCCL_SOCKET_IFNAME}" \
     --env NCCL_NET_GDR_LEVEL="\${NCCL_NET_GDR_LEVEL}" \
     --env NCCL_DEBUG="INFO" \
     -B /scratch/project_462000964:/scratch/project_462000964:rw \
     "$(SIF)" \
     "$(BINDIR)/inf_wrapper.sh"
EOF
	@chmod +x "$(MAKESCRIPT)"
	@echo "✅ Wrote $(MAKESCRIPT)"



inference-submit: clean-inference-lock inference-plan
> @set -euo pipefail
> jobid="$$(sbatch --parsable \
> echo "$$jobid" > "$(INF_FLAG)"
> echo "Submitted inference job $$jobid"
> echo "After job $$jobid finishes, run: make metrics"

run-inference: check-in-model-dir
> @set -euo pipefail
> if [ "$${SLURM_JOB_NAME:-}" != "inference" ]; then \
>   echo "run-inference must execute inside SLURM" >&2; \
>   echo "Expected SLURM job name inference but got $${SLURM_JOB_NAME:-<unset>}" >&2; \
>   exit 1; \
> fi
> if [ ! -s "$(INFERENCE_CALLS)" ]; then \
>   echo "No commands in $(INFERENCE_CALLS)"; \
>   exit 0; \
> fi
> trap 'rm -f "$(INF_FLAG)"' EXIT
> echo "Launching distributed inference..."
> export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
> export NCCL_NET_GDR_LEVEL=PHB
> srun /usr/bin/singularity exec \
>      --env NCCL_SOCKET_IFNAME="$$NCCL_SOCKET_IFNAME" \
>      --env NCCL_NET_GDR_LEVEL="$$NCCL_NET_GDR_LEVEL" \
>      --env NCCL_DEBUG="INFO" \
>      -B /scratch/project_462000964:/scratch/project_462000964:rw \
>      "$(SIF)" \
>      "$(BINDIR)/inf_wrapper.sh"
> echo "Current simple fallback:"

