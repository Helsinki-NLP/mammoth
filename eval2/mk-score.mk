# -----------------------------------------------------------------------------
# Metric command planning and submission
# -----------------------------------------------------------------------------

.PHONY: ensure-sacre-venv metrics-plan metrics-submit run-sacre clean-metrics-lock

ensure-sacre-venv: check-in-model-dir
> @set -euo pipefail
> module load cray-python
> echo "Ensuring sacrebleu virtual environment..."
> if [ ! -f "$(SACRE_ACTIVATE)" ]; then \
>   python3 -m venv "$(SACRE_VENV)"; \
>   source "$(SACRE_ACTIVATE)"; \
>   python -m pip install --upgrade pip; \
>   pip install sacrebleu; \
>   echo "Built $(SACRE_VENV)"; \
> else \
>   source "$(SACRE_ACTIVATE)"; \
> fi

clean-metrics-lock:
> @set -euo pipefail
> if [ -e "$(MET_FLAG)" ]; then \
>   oldjob="$$(cat "$(MET_FLAG)")"; \
>   if squeue -h -j "$$oldjob" | grep -q .; then \
>     echo "Metrics job $$oldjob is still queued/running."; \
>     exit 0; \
>   fi; \
>   echo "Removing stale metrics flag for job $$oldjob"; \
>   rm -f "$(MET_FLAG)"; \
> fi

metrics-plan: check-in-model-dir dirs ensure-sacre-venv
> @set -euo pipefail
> if [ -f "$(SACRE_FLAG)" ]; then
>   echo "Another process has locked metric planning: $(SACRE_FLAG)" >&2
>   echo "(if this claim is not true, remove $(SACRE_FLAG))" >&2
>   exit 1
> fi
> touch "$(SACRE_FLAG)"
> trap 'rm -f "$(SACRE_FLAG)"' EXIT
> bash "$(TESTINGDIR)/inf_make.sh" --fresh "$(SACRE_CALLS)"
> if [ ! -s "$(SACRE_CALLS)" ]; then \
>   echo "Found no sacrebleu commands in $(SACRE_CALLS)"
>   exit 0
> fi
> echo -n "The number of sacrebleu commands: "
> egrep 'sacrebleu' "$(SACRE_CALLS)" | wc -l

metrics-submit: check-in-model-dir dirs clean-metrics-lock metrics-plan
> @set -euo pipefail
> echo "Producing the script: $(OUTFILE) ..."
> if [ -f "$(INFFLAG)" ]; then
>   echo "Another instance has locked $(OUTFILE)"
>   echo "If this is not true, remove $(INFFLAG)"
>   exit 1
> fi
> touch "$(INFFLAG)"
> trap 'rm -f "$(INFFLAG)"' EXIT
> bash "$(TESTINGDIR)/inf_make.sh" --fresh "$(OUTFILE)"
> if [ ! -s "$(SACRE_CALLS)" ]; then \
>   echo "Found no sacrebleu commands in $(SACRE_CALLS) to submit"
>   exit 0; 
> fi
> jobid="$$(sbatch --parsable \
>   --job-name=met_make.sh \
>   --output=sacrebleu-%j.out \
>   --error=sacrebleu-%j.err \
>   --partition=small \
>   --nodes=1 \
>   --ntasks=1 \
>   --cpus-per-task=1 \
>   --time=04:00:00 \
>   --mem=8G \
>   --account=$(PROJECT) \
>   --chdir="$(MODELDIR)" \
>   --wrap='make -f Makefile run-sacre')" || { echo "sbatch failed" >&2; exit 1; }
> echo "$$jobid" > "$(MET_FLAG)"
> echo "Submitted metrics job $$jobid"
> echo "After job $$jobid has finished, run: make met"

run-sacre: check-in-model-dir
> @set -euo pipefail
> if [ "$${SLURM_JOB_NAME:-}" != "met_make.sh" ]; then
>   echo "Expected SLURM job name met_make.sh but got $${SLURM_JOB_NAME:-<unset>}" >&2;
>   exit 1;
> fi
> module load cray-python
> if [ -s "$(SACRE_CALLS)" ]; then
>   [ -f "$(SACRE_ACTIVATE)" ] || { echo "Missing activate script: $(SACRE_ACTIVATE)" >&2; exit 1; }
>   source "$(SACRE_ACTIVATE)"
>   trap 'rm -f "$(MET_FLAG)"' EXIT
>   srun bash "$(SACRE_CALLS)"
> else
>   echo "Did not find any commands in $(SACRE_CALLS)"
>   echo "This means measurements are already complete."
> fi
