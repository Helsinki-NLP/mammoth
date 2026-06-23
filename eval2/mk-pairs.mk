TRAINCONFIG := $(MODELDIR)/train.yaml

ZEROSHOTPAIRSINPUT    := $(MODELDIR)/inf_zeroshot.txt.input
ZEROSHOTPAIRS         := $(MODELDIR)/inf_zeroshot.txt
SUPERVISEDPAIRSINPUT  := $(MODELDIR)/inf_supervised.txt.input
SUPERVISEDPAIRS       := $(MODELDIR)/inf_supervised.txt

.PHONY: pass-pairs require-pairs 

$(ZEROSHOTPAIRSINPUT): $(TRAINCONFIG) $(VIEWPYTHON) | dirs not-in-slurm $(SELFDIR)/inf_pairs.py
> @set -euo pipefail
> @echo "Building $(ZEROSHOTPAIRSINPUT)..."
> @module load cray-python; $(VIEWPYTHON) "$(SELFDIR)/inf_pairs.py" "$(TRAINCONFIG)" --zs-out "$(ZEROSHOTPAIRSINPUT)" >&2

$(SUPERVISEDPAIRSINPUT): $(TRAINCONFIG) $(VIEWPYTHON) | dirs not-in-slurm $(SELFDIR)/inf_pairs.py
> @set -euo pipefail; \
> echo "Building $(SUPERVISEDPAIRSINPUT)..."; \
> module load cray-python; $(VIEWPYTHON) "$(SELFDIR)/inf_pairs.py" "$(TRAINCONFIG)" --supervised-pairs-and-quit "$(SUPERVISEDPAIRSINPUT)" >&2

.SECONDARY: $(ZEROSHOTPAIRS) $(SUPERVISEDPAIRS)

$(ZEROSHOTPAIRS): $(ZEROSHOTPAIRSINPUT) | mk-shared
> @if [ "$(FORCE_PAIRS)" = 1 ]; then \
>   cp -p "$<" "$@"; \
>   echo "mk-pairs.mk: ✅ Accepted proposed zero-shot pairs: $@"; \
> else \
>   echo "mk-pairs.mk: ❌ Missing zero-shot pair selection: $@" >&2; \
>   echo "mk-pairs.mk:    Read suggestions from: $<" >&2; \
>   echo "mk-pairs.mk:    To pass the automatic proposals as selections, use target 'mk-pairs-force'" >&2; \
>   exit 1; \
> fi

$(SUPERVISEDPAIRS): $(SUPERVISEDPAIRSINPUT) | mk-shared 
> @if [ "$(FORCE_PAIRS)" = 1 ]; then \
>   cp -p "$<" "$@"; \
>   echo "mk-pairs.mk: ✅ Accepted proposed supervised pairs: $@"; \
> else \
>   echo "mk-pairs.mk: ❌ Missing supervised pair selection: $@" >&2; \
>   echo "mk-pairs.mk:    Read suggestions from: $<" >&2; \
>   echo "mk-pairs.mk:    To pass the automatic proposals as selections, use target 'mk-pairs-force'" >&2; \
>   exit 1; \
> fi

require-pairs: $(ZEROSHOTPAIRS) $(SUPERVISEDPAIRS) | mk-shared 
> @echo "mk-pairs.mk: ✅ Pair selections exist with line counts: "; \
> wc -l "$(ZEROSHOTPAIRS)" "$(SUPERVISEDPAIRS)" | sed 's/^/mk-pairs.mk:    /'

