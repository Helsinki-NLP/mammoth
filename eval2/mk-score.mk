# -----------------------------------------------------------------------------
# Metric command planning and submission
# -----------------------------------------------------------------------------

# mk-calls (FORCE_PAIRS=1) (ensures the calls.out)
#   |
#   |   status-score
#   |    | 
#   |    |  clean-met-lock
#   |    |     | (ensure no process is running; no $MET_FLAG exists)
#   |    V     V
#   |   mk-score
#   |    |  (run $MET_SBATCH)
#   V    V
# mk-score-force

.PHONY: mk-score mk-score-force clean-met-lock status-score

clean-met-lock: 
> @set -euo pipefail; \
> if [ -e "$(MET_FLAG)" ]; then \
>   oldjob="$$(cat "$(MET_FLAG)")"; \
>   if squeue -h -j "$$oldjob" | grep -q .; then  \
>     echo "mk-score.mk: ❌ Scoring job $$oldjob is still queued/running for model:"; \
>     echo "mk-score.mk:       $(MODELDIR)"; \
>     echo "mk-score.mk:    I need to quit to avoid running a conflicting job."; \
>     exit 1; \
>   fi; \
>   echo "mk-score.mk: Removing stale scoring flag for job $$oldjob"; \
>   rm -f "$(MET_FLAG)"; \
> fi; \
> echo "mk-score.mk: ✅ Verified that no other scoring job is running"

status-score: | mk-calls slurm clean-met-lock 
> @echo "mk-score.mk: ✅ Checking the status of scoring planning:"; \
> if  [ -e "$(MET_TEMPLATE)" ]; then echo "mk-score.mk:     template: $(MET_TEMPLATE)"; else \
>   echo "mk-score.mk: ❌  template: $(MET_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(MET_SCRIPT)" ];   then echo "mk-score.mk:       script: $(MET_SCRIPT)";   else \
>   echo "mk-score.mk: ❌    script: $(MET_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(CNT_TEMPLATE)" ]; then echo "mk-score.mk:     template: $(CNT_TEMPLATE)"; else \
>   echo "mk-score.mk: ❌  template: $(CNT_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(CNT_SCRIPT)" ];   then echo "mk-score.mk:       script: $(CNT_SCRIPT)";   else \
>   echo "mk-score.mk: ❌    script: $(CNT_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(SACRE_CALLS)" ];    then echo "mk-score.mk:        calls: $(SACRE_CALLS)";    else \
>   echo "mk-score.mk: ❌     calls: $(SACRE_CALLS) missing";  exit 1; fi; \
> if  [ -e "$(MET_SBATCH)" ];   then echo "mk-score.mk:       sbatch: $(MET_SBATCH)";   else \
>   echo "mk-score.mk: ❌    script: $(MET_SBATCH) missing"; exit 1; fi; \
> cat "$(MET_SBATCH)" | sed 's/^/mk-score.mk: /'; \
> job=$$(sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p; /^[0-9][0-9]*$$/p' "$(MET_FLAG)" | head -n1); \
> squeue -j "$$job" -o "%.18i %.40j %.10T %.12M %.12l %.30R"; \
> echo "mk-score.mk: ✨ I am happy with slurm run preparations."; \
> echo

score: $(SACRE_CALLS) $(MET_SCRIPT) $(MET_SBATCH) $(CNT_SCRIPT) | status-score 
> @set -euo pipefail; \
> if [ ! -s "$(SACRE_CALLS)" ]; then \
>   echo "mk-score.mk: ❌ Found no sacrebleu commands in $(SACRE_CALLS) to submit"; \
>   exit 0; \
> fi; \
> echo -n "The number of sacrebleu commands: "; \
> egrep 'sacrebleu' "$(SACRE_CALLS)" | wc -l; \
> cat "$(MET_SBATCH)"; \
> score_out="$$(bash "$(MET_SBATCH)")"; \
> echo "$$score_out"; \
> score_job="$$(printf '%s\n' "$$score_out" | awk '{print $$NF}')"; \
> echo "$$score_job" > "$(MET_FLAG)"; \
> rm -f "$(MET_DONE)"; \
> echo "mk-score.mk: ✅ Submitted scoring job $$score_job"; \
> cnt_job="$$(sbatch --parsable --dependency=afterok:$$score_job "$(CNT_SCRIPT)")"; \
> echo "mk-score.mk: ✅ Submitted continuation job $$cnt_job after scoring"

mk-score-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODELDIR)"; \
>   MODEL="$(MODEL)" FORCE_PAIRS=1 $(FIRST_GOAL) mk-score

mk-score: score
> @true
