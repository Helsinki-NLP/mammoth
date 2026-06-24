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

.PHONY: mk-score mk-score-force status-score score 

status-score: | mk-calls slurm clean-met-lock 
> @echo "mk-score.mk: ✅ Checking the status of scoring planning:"; \
> if  [ -e "$(SACRE_CALLS)" ];    then \
>   echo "mk-score.mk:        calls: $(SACRE_CALLS)";\
> else \
>   echo "mk-score.mk: ❌     calls: $(SACRE_CALLS) missing";  exit 1; fi; \
> if [ ! -s "$(SACRE_CALLS)" ]; then \
>   echo "mk-score.mk: ❌ Found no sacrebleu calls in $(SACRE_CALLS) to submit"; \
>   exit 0; \
> fi; \
> echo -n "The number of sacrebleu commands: "; \
> egrep 'sacrebleu' "$(SACRE_CALLS)" | wc -l; \
> if  [ -e "$(MET_TEMPLATE)" ]; then echo "mk-score.mk:     template: $(MET_TEMPLATE)"; else \
>   echo "mk-score.mk: ❌  template: $(MET_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(MET_SCRIPT)" ];   then echo "mk-score.mk:       script: $(MET_SCRIPT)";   else \
>   echo "mk-score.mk: ❌    script: $(MET_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(CNT_TEMPLATE)" ]; then echo "mk-score.mk:     template: $(CNT_TEMPLATE)"; else \
>   echo "mk-score.mk: ❌  template: $(CNT_TEMPLATE) missing"; exit 1; fi; \
> if  [ -e "$(CNT_SCRIPT)" ];   then echo "mk-score.mk:       script: $(CNT_SCRIPT)";   else \
>   echo "mk-score.mk: ❌    script: $(CNT_SCRIPT) missing"; exit 1; fi; \
> if  [ -e "$(MET_SBATCH)" ];   then echo "mk-score.mk:       sbatch: $(MET_SBATCH)";   else \
>   echo "mk-score.mk: ❌    script: $(MET_SBATCH) missing"; exit 1; fi; \
> cat "$(MET_SBATCH)" | sed 's/^/mk-score.mk: /'; \
> if  [ -e "$(MET_FLAG)" ];     then \
>     job=$$(sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p; /^[0-9][0-9]*$$/p' "$(INF_FLAG)" | head -n1); \
>     squeue -j "$$job" -o "%.18i %.40j %.10T %.12M %.12l %.30R"; \
> fi; \
> echo "mk-score.mk: ✨ I am happy with slurm run preparations."; \
> echo

score: $(SACRE_CALLS) $(MET_SCRIPT) $(MET_SBATCH) $(CNT_SCRIPT) | status-score 
> @set -euo pipefail; \
> cat "$(MET_SBATCH)"; \
> score_out="$$(bash "$(MET_SBATCH)")"; \
> echo "$$score_out"; \
> score_job="$$(printf '%s\n' "$$score_out" | awk '{print $$NF}')"; \
> echo "$$score_job" > "$(MET_FLAG)"; \
> rm -f "$(MET_DONE)"; \
> echo "mk-score.mk: ✅ Submitted scoring job $$score_job"; \
> cnt_job="$$(sbatch --parsable --dependency=afterany:$$score_job "$(CNT_SCRIPT)")"; \
> echo "mk-score.mk: ✅ Submitted continuation job $$cnt_job after scoring"

mk-score-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODELDIR)"; \
>   MODEL="$(MODEL)" FORCE_PAIRS=1 $(FIRST_GOAL) mk-score

mk-score: score
> @true

mk-comet:
> @echo "mk-comet.mk: ✅ COMET evaluation functionality is not currently implemented"
