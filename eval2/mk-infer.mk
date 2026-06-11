# -----------------------------------------------------------------------------
# Inference planning, approval, submission, and status
# -----------------------------------------------------------------------------

.PHONY: mk-infer inference-plan inference-status inference-complete run-infer \
        clean-inference-lock clean-infer-plan fresh-infer

INF_DONE     := $(MODELDIR)/inference.done
INF_FLAG     := $(MODELDIR)/inference.submitted

INF_CALLS    := $(OUTDIR)/calls.out
INF_SCRIPT   := $(OUTDIR)/inf.slurm
INF_SBATCH   := $(OUTDIR)/inf.sbatch

INF_TEMPLATE := $(SELFDIR)/inf_templ.slurm
INF_DISTR    := $(SELFDIR)/inf_distr.sh

clean-inference-lock: 
> @set -euo pipefail; \
> if [ -e "$(INF_FLAG)" ]; then \
>   oldjob="$$(cat "$(INF_FLAG)")"; \
>   if squeue -h -j "$$oldjob" | grep -q .; then  \
>     echo "mk-infer.mk: ❌ Inference job $$oldjob is still queued/running for model:"; \
>     echo "mk-infer.mk:       $(MODELDIR)"; \
>     echo "mk-infer.mk:    I need to quit to avoid running a conflicting job."; \
>     exit 1; \
>   fi; \
>   echo "mk-infer.mk: Removing stale inference flag for job $$oldjob"; \
>   rm -f "$(INF_FLAG)"; \
> fi; \
> echo "mk-infer.mk: ✅ Verified that no other inference job is running"

$(INF_SCRIPT): $(INF_TEMPLATE) $(SELFDIR)/mk-infer.mk mk-yamls clean-inference-lock 
> @ncalls="$$(wc -l < $(OUTDIR)/calls.out)"; \
> echo "mk-infer.mk: ✅ Inference slurm template found $<"; \
> echo "mk-infer.mk:    Creating a script for job inf_$(FIRST_GOAL)_$${ncalls}_tasks..."; \
> m4 \
>   -D__SIF__="$(SIF)" \
>   -D__ACCOUNT__="$(JOBPROJECT)" \
>   -D__DISKPROJECT__="$(DISKPROJECT)" \
>   -D__SELFDIR__="$(SELFDIR)" \
>   -D__OUTDIR__="$(OUTDIR)" \
>   -D__LOGDIR__="$(LOGDIR)" \
>   -D__JOB_NAME__="inf_$(FIRST_GOAL)_$${ncalls}_tasks" \
>   -D__MAKESCRIPT__="$(INF_SCRIPT)" \
>   -D__MODELDIR__="$(MODELDIR)" \
>   "$<" >"$@"; \
> echo "mk-infer.mk: ✅ Job-specific inference slurm script now at:";\
> echo "mk-infer.mk:    $@"; \
> chmod +x "$@"

$(INF_SBATCH): $(INF_CALLS) $(INF_SCRIPT) mk-yamls clean-inference-lock 
> @export OUTDIR="${OUTDIR}"; \
> bash $(INF_SCRIPT); \
> echo "mk-infer.mk: ✅ Created the sbatch command:"; \
> cat $(INF_SBATCH) | sed 's/^/mk-infer.mk:    /'

mk-infer: $(INF_CALLS) $(INF_SCRIPT) $(INF_SBATCH) 
> @echo "mk-infer.mk: ✅ Inference planning complete."; \
> echo "mk-infer.mk:     bin dir: $(SELFDIR)"; \
> echo "mk-infer.mk:    template: $(INF_TEMPLATE)"; \
> echo "mk-infer.mk:      outdir: $(OUTDIR)"; \
> echo "mk-infer.mk:       calls: $(INF_CALLS)"; \
> echo "mk-infer.mk:     account: $(JOBPROJECT)"; \
> echo "mk-infer.mk:      script: $(INF_SCRIPT)"; \
> echo "mk-infer.mk:      sbatch: $(INF_SBATCH)"; \
> echo "mk-infer.mk: ✨ I am happy with slurm run preparations."; \
> echo "mk-infer.mk: ❓ Human approval required before submission:"; \
> cat "$(INF_SBATCH)" | sed 's/^/mk-infer.mk:      /'; \
> echo "mk-infer.mk:    Run: make run-infer"; \
> echo

status-infer:
> @echo "mk-infer.mk: ✅ Checking the status of inference planning:"; \
> if  [ -e "$(INF_FLAG)" ];     then echo "mk-infer.mk:     run flag: $(INF_FLAG)";     else \
>   echo "mk-inder.mk: ❌  run flag: $(INF_FLAG) missing";   exit 1; fi; \
> if  [ -e "$(INF_TEMPLATE)" ]; then echo "mk-infer.mk:     template: $(INF_TEMPLATE)"; else \
>   echo "mk-inder.mk: ❌  template: $(INF_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(INF_CALLS)" ];    then echo "mk-infer.mk:        calls: $(INF_CALLS)";    else \
>   echo "mk-inder.mk: ❌     calls: $(INF_CALLS) missing";  exit 1; fi; \
> if  [ -e "$(INF_SCRIPT)" ];   then echo "mk-infer.mk:       script: $(INF_SCRIPT)";   else \
>   echo "mk-inder.mk: ❌    script: $(INF_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(INF_SBATCH)" ];   then echo "mk-infer.mk:       sbatch: $(INF_SBATCH)";   else \
>   echo "mk-inder.mk: ❌    script: $(INF_SBATCH) missing"; exit 1; fi; \
> cat "$(INF_SBATCH)" | sed 's/^/mk-infer.mk: /'; \
> job=$$(sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p; /^[0-9][0-9]*$$/p' "$(INF_FLAG)" | head -n1); \
> squeue -j "$$job" -o "%.18i %.40j %.10T %.12M %.12l %.30R"; \
> echo "mk-infer.mk: ✨ I am happy with slurm run preparations."; \
> echo

run-infer: clean-inference-lock $(INF_CALLS) $(INF_SCRIPT) $(INF_SBATCH) mk-yamls 
> @set -euo pipefail; \
> cat "$(INF_SBATCH)"; \
> tmp="$$(mktemp)"; \
> bash "$(INF_SBATCH)" > "$$tmp"; \
> cat "$$tmp"; \
> mv "$$tmp" "$(INF_FLAG)"

clean-infer:
> rm -f $(INF_SCRIPT) $(INF_BATCH)




