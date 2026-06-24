# -----------------------------------------------------------------------------
# Continuation control
# -----------------------------------------------------------------------------

.PHONY: continue-eval 

#  continue-eval
#   |  |  |  |
#   |  |  |  \--> run-infer
#   |  |  \-----> run-scores
#   |  \--------> run-comet
#   \-----------> run-viz

continue-eval: clean-inf-lock clean-cnt-lock clean-met-lock
> @set -euo pipefail; \
> echo "mk-contr.mk: 🛠️ Replanning evaluation state for $(MODELDIR)"; \
> echo "mk-contr.mk:    Remove $(TESTCONFIG).{out,err}"; \
> rm -f $(TESTCONFIG).out $(TESTCONFIG).err; \
> $(MAKE) -C "$(SELFDIR)" --no-print-directory $(FIRST_GOAL) refresh-calls; \
> inf_calls=0; sacre_calls=0; comet_calls=0; \
> [ -f "$(INF_CALLS)" ]   && inf_calls="$$(grep -c . "$(INF_CALLS)" || true)"; \
> [ -f "$(SACRE_CALLS)" ] && sacre_calls="$$(grep -c 'sacrebleu' "$(SACRE_CALLS)" || true)"; \
> [ -f "$(COMET_CALLS)" ] && comet_calls="$$(grep -c . "$(COMET_CALLS)" || true)"; \
> echo "mk-contr.mk:    inference calls: $$inf_calls"; \
> echo "mk-contr.mk:    sacre calls:     $$sacre_calls"; \
> echo "mk-contr.mk:    comet calls:     $$comet_calls"; \
> if [ "$$inf_calls" -gt 0 ]; then \
>   echo "mk-contr.mk: ✅ Next step: mk-infer"; \
>   $(MAKE) -C "$(SELFDIR)" --no-print-directory $(FIRST_GOAL) mk-infer-score; \
> elif [ "$$sacre_calls" -gt 0 ]; then \
>   echo "mk-contr.mk: ✅ Next step: mk-score"; \
>   $(MAKE) -C "$(SELFDIR)" --no-print-directory $(FIRST_GOAL) mk-score; \
> elif [ "$$comet_calls" -gt 0 ]; then \
>   echo "mk-contr.mk: ✅ Next step: mk-comet"; \
>   $(MAKE) -C "$(SELFDIR)" --no-print-directory $(FIRST_GOAL) mk-comet; \
> elif [ ! -e "$(MODELDIR)/viz.done" ]; then \
>   echo "mk-contr.mk: ✅ Next step: mk-viz"; \
>   $(MAKE) -C "$(SELFDIR)" --no-print-directory $(FIRST_GOAL) mk-viz; \
> else \
>   echo "mk-contr.mk: ✅ Evaluation pipeline complete."; \
> fi

.PHONY: continue-all

continue-all:
> @set -u; \
> rc=0; \
> for a in $(MODEL_ALIASES); do \
>   eval "model_dir=\$${MODEL_$$a}"; \
>   inf_flag="$$model_dir/inference.submitted"; \
>   if [ -f "$$inf_flag" ]; then \
>     jobid="$$(cat "$$inf_flag" 2>/dev/null || true)"; \
>     if [ -n "$$jobid" ] && squeue -h -j "$$jobid" 2>/dev/null | grep -q .; then \
>       echo "mk-contr.mk: ⏭️  Skipping $$a because inference job $$jobid is still running/queued"; \
>       continue; \
>     fi; \
>   fi; \
>   echo "mk-contr.mk: 🛠️ continue-eval for $$a"; \
>   if ! $(MAKE) --no-print-directory -C "$(SELFDIR)" "$$a" continue-eval; then \
>     echo "mk-contr.mk: ❌ continue-eval failed for $$a"; \
>     rc=1; \
>   fi; \
> done; \
> exit $$rc
