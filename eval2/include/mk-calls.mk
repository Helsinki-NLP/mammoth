.PHONY: refresh-calls calls clean-calls mk-calls mk-calls-force clean-calls clean

calls: $(INF_CALLS) $(TESTCONFIG).out
> @echo   "mk-calls.mk:    Logged to $(TESTCONFIG).out"; \
> echo    "mk-calls.mk:    Created   $(OUTDIR)/*.yaml"; \
> echo -n "mk-calls.mk:    The number of yaml files is "; \
> find "$(OUTDIR)" -maxdepth 1 -type f -name '*.yaml' | wc -l; \
> echo "mk-calls.mk: ✅ Lines in prepared out files:";\
> wc -l $(OUTDIR)/*.out 2>/dev/null | sed 's/^/mk-calls.mk:    /'
> @echo "mk-calls.mk: ✨ I am happy with the yaml files and the planned inference calls."; \
> echo

clean-calls:
> rm -f $(INF_CALLS) $(SACRE_CALLS) $(COMET_CALLS) $(TESTCONFIG).out $(TESTCONFIG).err
> @echo "mk-calls.mk: ✨ Cleaning of calls done."; \
> echo

clean: clean-calls 
> rm -f $(OUTDIR)/*.yaml 
> @echo "mk-calls.mk: ✨ Cleaning of inference yaml files done."; \
> echo

mk-calls: calls 
> @true

mk-calls-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" FORCE_PAIRS=1 $(FIRST_GOAL) calls

$(INF_CALLS): $(TESTCONFIG).out
> @test -f "$(INF_CALLS)"

$(TESTCONFIG).out: $(TRAINCONFIG) $(MODELDIR)/mammoth.selected $(PRS_DONE)
> @set -euo pipefail
> @echo "mk-calls.mk: 🛠️ Creating tentative testing tasks..."
> @export MAMMOTH="$$(cat "$(MODELDIR)/mammoth.selected")"; \
>   export LOGDIR="$(LOGDIR)"; \
>   export SCRDIR="$(SCRDIR)"; \
>   export MODEL="$(MODEL)"; \
>   export TRAINCONFIG="$(TRAINCONFIG)"; \
>   export OUTDIR="$(OUTDIR)"; \
>   export DATADIR="$(DATADIR)"; \
>   export ZEROSHOTPAIRS="$(ZEROSHOTPAIRS)"; \
>   export SUPERVISEDPAIRS="$(SUPERVISEDPAIRS)"; \
> module load cray-python; \
> $(VIEWPYTHON) "$(SELFDIR)/bin/inf_plan.py" >"$(TESTCONFIG).err" 2>&1 || true; \
> if [ -f "$(TESTCONFIG).err" ] && grep -Fq 'All stages of planning completed' "$(TESTCONFIG).err"; then \
>    mv "$(TESTCONFIG).err" "$(TESTCONFIG).out"; \
> else \
>    rm -f "$(TESTCONFIG).out"; \
>    echo "mk-calls.mk: ❌ Planning failed."; \
>    echo "mk-calls.mk:    The last 20 lines from $(TESTCONFIG).err:"; \
>    echo -----------------------------------; \
>    tail -20 "$(TESTCONFIG).err"; \
>    echo -----------------------------------; \
>    exit 1 ; \
> fi


