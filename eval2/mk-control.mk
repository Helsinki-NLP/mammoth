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

continue-eval: mk-shared
> @set -euo pipefail; \
> echo "mk-control.mk: 🛠️ Replanning evaluation state for $(MODELDIR)"; \
> $(MAKE) --no-print-directory MODELDIR="$(MODELDIR)" $(FIRST_GOAL) refresh-calls; \
> inf_calls=0; sacre_calls=0; comet_calls=0; \
> [ -f "$(INF_CALLS)" ]   && inf_calls="$$(grep -c . "$(INF_CALLS)" || true)"; \
> [ -f "$(SACRE_CALLS)" ] && sacre_calls="$$(grep -c 'sacrebleu' "$(SACRE_CALLS)" || true)"; \
> [ -f "$(COMET_CALLS)" ] && comet_calls="$$(grep -c . "$(COMET_CALLS)" || true)"; \
> echo "mk-control.mk:    inference calls: $$inf_calls"; \
> echo "mk-control.mk:    sacre calls:     $$sacre_calls"; \
> echo "mk-control.mk:    comet calls:     $$comet_calls"; \
> if [ "$$inf_calls" -gt 0 ]; then \
>   echo "mk-control.mk: ✅ Next step: run-infer"; \
>   $(MAKE) --no-print-directory MODELDIR="$(MODELDIR)" mk-infer; \
> elif [ "$$sacre_calls" -gt 0 ]; then \
>   echo "mk-control.mk: ✅ Next step: run-scores"; \
>   $(MAKE) --no-print-directory MODELDIR="$(MODELDIR)" mk-scores; \
> elif [ "$$comet_calls" -gt 0 ]; then \
>   echo "mk-control.mk: ✅ Next step: run-comet"; \
>   $(MAKE) --no-print-directory MODELDIR="$(MODELDIR)" mk-comet; \
> elif [ ! -e "$(MODELDIR)/viz.done" ]; then \
>   echo "mk-control.mk: ✅ Next step: run-viz"; \
>   $(MAKE) --no-print-directory MODELDIR="$(MODELDIR)" mk-viz; \
> else \
>   echo "mk-control.mk: ✅ Evaluation pipeline complete."; \
> fi

