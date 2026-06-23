# mk-calls (FORCE_PAIRS=1) (ensures the calls.out)
#   |
#   |   status-infer
#   |    | 
#   |    |  clean-lock
#   |    |     | (ensure no process is running; no $INF_FLAG exists)
#   |    V     V
#   |   mk-infer
#   |    |  (run $INF_SBATCH)
#   V    V
# mk-infer-force

.PHONY: mk-infer mk-infer-force clean-inf-lock status-infer

clean-inf-lock: 
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

status-infer: | mk-calls clean-inf-lock 
> @echo "mk-infer.mk: ✅ Checking the status of inference planning:"; \
> if  [ -e "$(INF_TEMPLATE)" ]; then echo "mk-infer.mk:     template: $(INF_TEMPLATE)"; else \
>   echo "mk-infer.mk: ❌  template: $(INF_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(INF_SCRIPT)" ];   then echo "mk-infer.mk:       script: $(INF_SCRIPT)";   else \
>   echo "mk-infer.mk: ❌    script: $(INF_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(CNT_TEMPLATE)" ]; then echo "mk-infer.mk:     template: $(CNT_TEMPLATE)"; else \
>   echo "mk-infer.mk: ❌  template: $(CNT_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(CNT_SCRIPT)" ];   then echo "mk-infer.mk:       script: $(CNT_SCRIPT)";   else \
>   echo "mk-infer.mk: ❌    script: $(CNT_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(INF_CALLS)" ];    then echo "mk-infer.mk:        calls: $(INF_CALLS)";    else \
>   echo "mk-infer.mk: ❌     calls: $(INF_CALLS) missing";  exit 1; fi; \
> if  [ -e "$(INF_SBATCH)" ];   then echo "mk-infer.mk:       sbatch: $(INF_SBATCH)";   else \
>   echo "mk-infer.mk: ❌    script: $(INF_SBATCH) missing"; exit 1; fi; \
> cat "$(INF_SBATCH)" | sed 's/^/mk-infer.mk: /'; \
> job=$$(sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p; /^[0-9][0-9]*$$/p' "$(INF_FLAG)" | head -n1); \
> squeue -j "$$job" -o "%.18i %.40j %.10T %.12M %.12l %.30R"; \
> echo "mk-infer.mk: ✨ I am happy with slurm run preparations."; \
> echo

infer: $(INF_CALLS) $(INF_SCRIPT) $(INF_SBATCH) $(CNT_SCRIPT) | status-infer
> @set -euo pipefail; \
> cat "$(INF_SBATCH)"; \
> infer_out="$$(bash "$(INF_SBATCH)")"; \
> echo "$$infer_out"; \
> infer_job="$$(printf '%s\n' "$$infer_out" | awk '{print $$NF}')"; \
> echo "$$infer_job" > "$(INF_FLAG)"; \
> rm -f "$(INF_DONE)"; \
> echo "mk-infer.mk: ✅ Submitted inference job $$infer_job"; \
> cnt_job="$$(sbatch --parsable --dependency=afterok:$$infer_job "$(CNT_SCRIPT)")"; \
> echo "mk-control.mk: ✅ Submitted continuation job $$cnt_job after inference"

mk-infer-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODELDIR)"; \
>   MODEL="$(MODEL)" FORCE_PAIRS=1 $(FIRST_GOAL) mk-infer

mk-infer: infer
> @true


