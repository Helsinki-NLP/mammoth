# mk-calls (FORCE_PAIRS=1) (ensures the calls.out)
#   |
#   |   status-infer
#   |    | 
#   |    |  clean-inf-lock
#   |    |     | (ensure no process is running; no $INF_FLAG exists)
#   |    V     V
#   |   mk-infer
#   |    |  (run $INF_SBATCH)
#   V    V
# mk-infer-force

.PHONY: mk-infer mk-infer-force status-infer infer infer-score

status-infer: clean-inf-lock | mk-calls 
> @echo "mk-score.mk: ✅ Checking the status of inference planning:"; \
> if  [ -e "$(INF_CALLS)" ];    then \
>   echo "mk-infer.mk:        calls: $(INF_CALLS)"; \
> else \
>   echo "mk-infer.mk: ❌     calls: $(INF_CALLS) missing";  exit 1; fi; \
> if [ ! -s "$(INF_CALLS)" ]; then \
>   echo "mk-infer.mk: ❌ Found no inference commands in $(INF_CALLS) to submit"; \
>   exit 0; \
> fi; \
> echo -n "mk-infer.mk:   The number of inference calls: "; \
> egrep 'python' "$(INF_CALLS)" | wc -l; \
> if  [ -e "$(INF_TEMPLATE)" ]; then echo "mk-infer.mk:     template: $(INF_TEMPLATE)"; else \
>   echo "mk-infer.mk: ❌  template: $(INF_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(INF_SCRIPT)" ];   then echo "mk-infer.mk:       script: $(INF_SCRIPT)";   else \
>   echo "mk-infer.mk: ❌    script: $(INF_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(INF_SBATCH)" ];   then echo "mk-infer.mk:       sbatch: $(INF_SBATCH)";   else \
>   echo "mk-infer.mk: ❌    script: $(INF_SBATCH) missing"; exit 1; fi; \
> cat "$(INF_SBATCH)" | sed 's/^/mk-infer.mk: /'; \
> if  [ -e "$(CNT_TEMPLATE)" ]; then echo "mk-infer.mk:     continuation template: $(CNT_TEMPLATE)"; else \
>   echo "mk-infer.mk: ❌  template: $(CNT_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(CNT_SCRIPT)" ];   then echo "mk-infer.mk:       continuation script: $(CNT_SCRIPT)";   else \
>   echo "mk-infer.mk: ❌    script: $(CNT_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(INF_FLAG)" ];     then \
>     job=$$(sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p; /^[0-9][0-9]*$$/p' "$(INF_FLAG)" | head -n1); \
>     squeue -j "$$job" -o "%.18i %.40j %.10T %.12M %.12l %.30R"; \
> fi; \
> echo "mk-infer.mk: ✨ I am happy with slurm run preparations."; \
> echo

infer: $(INF_CALLS) $(INF_SCRIPT) $(INF_SBATCH) $(CNT_SCRIPT) | status-infer
> @set -euo pipefail; \
> cat "$(INF_SBATCH)"; \
> infer_out="$$(bash "$(INF_SBATCH)")"; \
> echo "$$infer_out"; \
> infer_job="$$(printf '%s\n' "$$infer_out" | awk '{print $$NF}')"; \
> test -n "$$infer_job"; \
> printf '%s\n' "$$infer_job" > "$(INF_FLAG)"; \
> rm -f "$(INF_DONE)"; \
> echo "mk-infer.mk: ✅ Submitted inference job $$infer_job"

infer-score: infer $(INF_FLAG)
> @set -euo pipefail; \
> infer_job="$$(cat "$(INF_FLAG)")"; \
> test -n "$$infer_job"; \
> cnt_job="$$(sbatch --parsable --dependency=afterany:$$infer_job "$(CNT_SCRIPT)")"; \
> echo "mk-control.mk: ✅ Submitted continuation job $$cnt_job after inference"

mk-infer-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODELDIR)"; \
>   MODEL="$(MODEL)" FORCE_PAIRS=1 $(FIRST_GOAL) infer

mk-infer-score-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODELDIR)"; \
>   MODEL="$(MODEL)" FORCE_PAIRS=1 $(FIRST_GOAL) infer-score

mk-infer: infer
> @true

mk-infer-score: infer-score
> @true


