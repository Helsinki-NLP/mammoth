TRAINCONFIG := $(MODELDIR)/train.yaml

ZEROSHOTPAIRSINPUT    := $(MODELDIR)/inf_zeroshot.txt.input
ZEROSHOTPAIRS         := $(MODELDIR)/inf_zeroshot.txt
SUPERVISEDPAIRSINPUT  := $(MODELDIR)/inf_supervised.txt.input
SUPERVISEDPAIRS       := $(MODELDIR)/inf_supervised.txt

PAIR_INPUTS_DONE      := $(MODELDIR)/pairs_input.done

.PHONY: pass-pairs require-pairs 

$(SELFDIR)/inf_pairs.py:
> @[[ -f "$(SELFDIR)/inf_pairs.py" ]] || { echo "mk-pairs.mk: ❌ Missing $(SELFDIR)/inf_pairs.py" >&2; exit 1; }

$(PAIR_INPUTS_DONE): $(TRAINCONFIG) $(VIEWPYTHON) $(SELFDIR)/inf_pairs.py | $(BAS_DONE) 
> @set -euo pipefail; \
> echo "mk-pairs.mk: 🛠️ Building $(ZEROSHOTPAIRSINPUT) and $(SUPERVISEDPAIRSINPUT)..."; \
> module load cray-python; \
> $(VIEWPYTHON) "$(SELFDIR)/inf_pairs.py" "$(TRAINCONFIG)" \
>   --zs-out "$(ZEROSHOTPAIRSINPUT)" \
>   --supervised-pairs-and-quit "$(SUPERVISEDPAIRSINPUT)" >&2; \
> touch "$@"

$(ZEROSHOTPAIRSINPUT) $(SUPERVISEDPAIRSINPUT): $(PAIR_INPUTS_DONE)
> @test -f "$@"

.SECONDARY: $(ZEROSHOTPAIRS) $(SUPERVISEDPAIRS)

$(ZEROSHOTPAIRS): $(ZEROSHOTPAIRSINPUT) | $(BAS_DONE)
> @if [ "$(FORCE_PAIRS)" = 1 ]; then \
>   if [ ! -e "$@" ] || ! cmp -s "$<" "$@"; then \
>     cp -p "$<" "$@"; \
>   else \
>     touch "$@"; \
>   fi; \
>   echo "mk-pairs.mk: ✅ Accepted proposed zero-shot pairs: $@"; \
> else \
>   echo "mk-pairs.mk: ❌ Missing zero-shot pair selection: $@" >&2; \
>   echo "mk-pairs.mk:    Read suggestions from: $<" >&2; \
>   echo "mk-pairs.mk:    To pass the automatic proposals as selections, use target 'mk-pairs-force'" >&2; \
>   exit 1; \
> fi

$(SUPERVISEDPAIRS): $(SUPERVISEDPAIRSINPUT) | $(BAS_DONE)
> @if [ "$(FORCE_PAIRS)" = 1 ]; then \
>   if [ ! -e "$@" ] || ! cmp -s "$<" "$@"; then \
>     cp -p "$<" "$@"; \
>   else \
>     touch "$@"; \
>   fi; \
>   echo "mk-pairs.mk: ✅ Accepted proposed supervised pairs: $@"; \
> else \
>   echo "mk-pairs.mk: ❌ Missing supervised pair selection: $@" >&2; \
>   echo "mk-pairs.mk:    Read suggestions from: $<" >&2; \
>   echo "mk-pairs.mk:    To pass the automatic proposals as selections, use target 'mk-pairs-force'" >&2; \
>   exit 1; \
> fi

$(PRS_DONE): $(ZEROSHOTPAIRS) $(SUPERVISEDPAIRS) | $(BAS_DONE)
> @echo "mk-pairs.mk: ✅ Pair selections exist with line counts:"; \
> wc -l "$(ZEROSHOTPAIRS)" "$(SUPERVISEDPAIRS)" | sed 's/^/mk-pairs.mk:    /'; \
> echo "mk-pairs.mk: ✨ I am happy with the pairs."; \
> test -e "$@" || touch "$@"; \
> if [ "$(ZEROSHOTPAIRS)" -nt "$@" ] || [ "$(SUPERVISEDPAIRS)" -nt "$@" ]; then touch "$@"; fi; \
> echo

mk-pairs: $(PRS_DONE)

mk-pairs-force: 
> @$(MAKE) --no-print-directory -C "$(SELFDIR)" FORCE_PAIRS=1 $(FIRST_GOAL) mk-pairs


