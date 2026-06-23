.PHONY: refresh-calls calls clean-calls mk-calls mk-calls-force clean-calls clean

calls: $(INF_CALLS) $(TESTCONFIG).out
> @echo "mk-calls.mk: ✅ Logged the inference YAML preparation to:";\
> echo "mk-calls.mk:    $(TESTCONFIG).out";\
> echo "mk-calls.mk: ✅ Created yaml files:";\
> ls $(OUTDIR)/*yaml 2>/dev/null | wc | sed 's/^/mk-calls.mk:    /';\
> echo "mk-calls.mk: ✅ Lines in prepared call files:";\
> wc -l $(OUTDIR)/*.out 2>/dev/null | sed 's/^/mk-calls.mk:    /'
> @echo "mk-calls.mk: ✨ I am happy with the yaml files and the planned inference calls."; \
> echo "mk-calls.mk: ❓ Do you want to clean and rebuild thoses files? Say: "; \
> echo "mk-calls.mk:       make clean-yamls  # clean .input files, yamls and commands, but leave and use manual selections to rebuild"; \
> echo

clean-calls:
> rm -f $(INF_CALLS) $(SACRE_CALLS) $(COMET_CALLS) $(TESTCONFIG).out $(TESTCONFIG).err
> @echo "mk-calls.mk: ✨ Cleaning of calls done."; \
> echo "mk-calls.mk: ❓ Do you want to rebuild thoses files? Say: "; \
> echo "mk-calls.mk:       make mk-calls  or  make mk-calls-force"; \
> echo

clean: clean-calls 
> rm -f $(OUTDIR)/*.yaml 
> @echo "mk-calls.mk: ✨ Cleaning of inference yaml files done."; \
> echo "mk-calls.mk: ❓ Do you want to rebuild thoses files? Say: "; \
> echo "mk-calls.mk:       make mk-calls  or  make mk-calls-force"; \
> echo

mk-calls: calls 
> @true

mk-calls-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODELDIR)"; \
>   MODEL="$(MODEL)" FORCE_PAIRS=1 $(FIRST_GOAL) calls

.SECONDARY: $(INF_CALLS) $(TESTCONFIG).out

$(INF_CALLS) $(TESTCONFIG).out: refresh-calls
> @true

refresh-calls: $(TRAINCONFIG) $(MODELDIR)/mammoth.selected | require-pairs 
> @set -euo pipefail
> @echo "mk-calls.mk: 🛠️ Creating tentative testing tasks..."
> @echo -n "mk-calls.mk:    In $(OUTDIR) the number of files is "
> @(ls $(OUTDIR)/*.yaml 2>/dev/null || true) | wc -l
> @export MAMMOTH="$$(cat "$(MODELDIR)/mammoth.selected")"; \
>   echo "mk-calls.mk: ✅ Using $${MAMMOTH}"; \
>   export LOGDIR="$(LOGDIR)"; \
>   export SCRDIR="$(SCRDIR)"; \
>   export MODEL="$(MODEL)"; \
>   export TRAINCONFIG="$(TRAINCONFIG)"; \
>   export OUTDIR="$(OUTDIR)"; \
>   export DATADIR="$(DATADIR)"; \
>   export ZEROSHOTPAIRS="$(ZEROSHOTPAIRS)"; \
>   export SUPERVISEDPAIRS="$(SUPERVISEDPAIRS)"; \
> module load cray-python; \
> $(VIEWPYTHON) "$(SELFDIR)/inf_plan.py" >"$(TESTCONFIG).err" 2>&1 || true; \
> if [ -f "$(TESTCONFIG).err" ] && grep -Fq 'All stages of planning completed' "$(TESTCONFIG).err"; then \
>    mv "$(TESTCONFIG).err" "$(TESTCONFIG).out"; \
>    echo "mk-calls.mk:    Logged to $(TESTCONFIG).out"; \
>    echo "mk-calls.mk:    Created   $(OUTDIR)/*.yaml"; \
>    echo -n "mk-calls.mk:    In $(OUTDIR) the number of files is "; \
>    find "$(OUTDIR)" -maxdepth 1 -type f -name '*.yaml' | wc -l; \
> else \
>    echo "mk-calls.mk: ❌ Planning failed."; \
>    echo "mk-calls.mk:    The last 20 lines from $(TESTCONFIG).err:"; \
>    echo -----------------------------------; \
>    tail -20 "$(TESTCONFIG).err"; \
>    echo -----------------------------------; \
>    exit 1 ; \
> fi

