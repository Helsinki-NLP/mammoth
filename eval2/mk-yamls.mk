# -----------------------------------------------------------------------------
# Inference YAML preparation
# -----------------------------------------------------------------------------
# mk-yamls.mk
#
# Purpose
# -------
# Pair extraction and inference-YAML preparation rules for the eval2 workflow.
#
# This Makefile fragment is responsible for the planning stage that turns a
# training configuration into:
#   - suggested zero-shot pair inputs
#   - suggested supervised pair inputs
#   - validated pair-selection files
#   - generated inference YAML files and call lists
#   - a planning log recording the output of `inf_plan.py`
#
# Current behavior
# ----------------
# The fragment defines:
#
#   pair-extractor
#       Verifies that `$(SELFDIR)/inf_pairs.py` exists.
#
#   pair-planner
#       Verifies that `$(SELFDIR)/inf_plan.py` exists.
#
#   require-pairs
#       Requires the final supervised and zero-shot pair-selection files and
#       reports their line counts.
#
#   $(ZEROSHOTPAIRSINPUT)
#       Builds the suggested zero-shot pair list by running:
#         inf_pairs.py TRAINCONFIG --zs-out ...
#
#   $(SUPERVISEDPAIRSINPUT)
#       Builds the suggested supervised pair list by running:
#         inf_pairs.py TRAINCONFIG --supervised-pairs-and-quit ...
#
#   $(ZEROSHOTPAIRS)
#   $(SUPERVISEDPAIRS)
#       Intentionally fail if the final pair-selection files are still missing,
#       while pointing the user to the generated suggestion files.
#
#   $(TESTCONFIG).err
#       Runs `inf_plan.py` with the required environment variables and records
#       its stderr/stdout output through `tee`.
#
#   inference-yamls-ok
#       Checks whether the planning log contains the success marker:
#         "All stages of planning completed"
#
#   inference-yamls
#       Reports successful completion of the YAML-preparation stage.
#
#   clean-inf-inputs
#       Removes the generated pair-input and testing files.
#
#   fresh-yamls
#       Removes the testing file and prints rebuild hints.
#
# Inputs
# ------
# Expected from the including Makefile:
#   TRAINCONFIG
#   VIEWPYTHON
#   SELFDIR
#   MODELDIR
#   OUTDIR
#   DATADIR
#   MAMMOTH
#   LOGDIR
#   SCRDIR
#   MODEL
#   TESTCONFIG
#   ZEROSHOTPAIRSINPUT
#   ZEROSHOTPAIRS
#   SUPERVISEDPAIRSINPUT
#   SUPERVISEDPAIRS
#
# Reads
# -----
#   - `$(SELFDIR)/inf_pairs.py`
#   - `$(SELFDIR)/inf_plan.py`
#   - the training YAML from `$(TRAINCONFIG)`
#   - the selected supervised and zero-shot pair files, once present
#
# Writes / effects
# ----------------
#   - suggested pair files:
#       - `$(ZEROSHOTPAIRSINPUT)`
#       - `$(SUPERVISEDPAIRSINPUT)`
#   - planning log:
#       - `$(TESTCONFIG).err`
#   - generated inference YAML files under `$(OUTDIR)`
#   - generated call lists under `$(OUTDIR)` via `inf_plan.py`
#
# Output
# ------
# The fragment prints progress, success/failure messages, pair counts, and
# rebuild hints.
#
# Notes
# -----
# - The fragment distinguishes between:
#     - suggested pair files (`*.input`)
#     - final pair-selection files
# - The final pair-selection files must exist before `mk-yamls` can complete.
# - `inf_plan.py` is treated as successful only if its log contains the marker:
#     "All stages of planning completed"
# - The planning stage is intended to run outside Slurm.

TRAINCONFIG := $(MODELDIR)/train.yaml

ZEROSHOTPAIRSINPUT    := $(MODELDIR)/inf_zeroshot.txt.input
ZEROSHOTPAIRS         := $(MODELDIR)/inf_zeroshot.txt
SUPERVISEDPAIRSINPUT  := $(MODELDIR)/inf_supervised.txt.input
SUPERVISEDPAIRS       := $(MODELDIR)/inf_supervised.txt


.PHONY: inference-yamls require-pairs clean-inf-inputs view-venv pair-extractor pair-planner fresh-yamls inference-yamls-ok pass-pairs

pair-extractor: 
> @[[ -f "$(SELFDIR)/inf_pairs.py" ]] || { echo "mk-yamls.mk: ❌ Missing $(SELFDIR)/inf_pairs.py" >&2; exit 1; }

pair-planner: 
> @ [[ -f "$(SELFDIR)/inf_plan.py" ]] || { echo "mk-yamls.mk: Missing $(SELFDIR)/inf_plan.py" >&2; exit 1; }

$(ZEROSHOTPAIRSINPUT): $(TRAINCONFIG) $(VIEWPYTHON) | dirs not-in-slurm pair-extractor
> @set -euo pipefail
> @echo "Building $(ZEROSHOTPAIRSINPUT)..."
> @module load cray-python; $(VIEWPYTHON) "$(SELFDIR)/inf_pairs.py" "$(TRAINCONFIG)" --zs-out "$(ZEROSHOTPAIRSINPUT)" >&2

$(SUPERVISEDPAIRSINPUT): $(TRAINCONFIG) $(VIEWPYTHON) | dirs not-in-slurm pair-extractor
> @set -euo pipefail; \
> echo "Building $(SUPERVISEDPAIRSINPUT)..."; \
> module load cray-python; $(VIEWPYTHON) "$(SELFDIR)/inf_pairs.py" "$(TRAINCONFIG)" --supervised-pairs-and-quit "$(SUPERVISEDPAIRSINPUT)" >&2


pass-pairs: $(ZEROSHOTPAIRSINPUT) $(SUPERVISEDPAIRSINPUT)
> @cp -p $(ZEROSHOTPAIRSINPUT)   $(ZEROSHOTPAIRS); \
> cp -p $(SUPERVISEDPAIRSINPUT) $(SUPERVISEDPAIRS); \
> echo "mk-yamls.mk: ✅ Passed the proposed language pair inputs as selections."; \

$(ZEROSHOTPAIRS): $(ZEROSHOTPAIRSINPUT)
> @echo "mk-yamls.mk: ❌ Missing zero-shot pair selection: $@" >&2 ;\
> echo "mk-yamls.mk:    Read suggestions from: $<" >&2; \
> echo "mk-yamls.mk:    To pass the automatic proposals as selections, use target 'pass-pairs'"; \
> exit 1

$(SUPERVISEDPAIRS): $(SUPERVISEDPAIRSINPUT)
> @echo "mk-yamls.mk: ❌ Missing supervised pair selection: $@" >&2; \
> echo "mk-yamls.mk:    Read suggestions from: $<" >&2; \
> echo "mk-yamls.mk:    To pass the automatic proposals as selections, use target 'pass-pairs'"; \
> exit 1

require-pairs: mk-shared $(ZEROSHOTPAIRS) $(SUPERVISEDPAIRS)
> @echo "mk-yamls.mk: ✅ Pair selections exist with line counts: "; \
> wc -l "$(ZEROSHOTPAIRS)" "$(SUPERVISEDPAIRS)" | sed 's/^/mk-yamls.mk:    /'

mk-yamls: require-pairs inference-yamls mk-shared
> @echo "mk-yamls.mk: ✨ I am happy with the yaml files and the planned inference calls."; \
> echo "mk-yamls.mk: ❓ Do you want to clean and rebuild thoses files? Say: "; \
> echo "mk-yamls.mk:       make clean-yamls  # clean .input files, yamls and commands, but leave and use manual selections to rebuild"; \
> echo

$(TESTCONFIG).err: $(TRAINCONFIG) $(MODELDIR)/mammoth.selected | dirs not-in-slurm require-pairs pair-planner
> @set -euo pipefail
> @echo "mk-yamls.mk: 🛠️ Creating tentative testing tasks..."
> @echo -n "mk-yamls.mk:    In $(OUTDIR) the number of files is "
> @(ls $(OUTDIR)/*.yaml 2>/dev/null || true) | wc -l
> @export MAMMOTH="$$(cat "$(MODELDIR)/mammoth.selected")"; \
>     echo "mk-yamls.mk: ✅ Using $${MAMMOTH}"; \
>     export LOGDIR="$(LOGDIR)"; \
>     export SCRDIR="$(SCRDIR)"; \
>     export MODEL="$(MODEL)"; \
>     export TRAINCONFIG="$(TRAINCONFIG)"; \
>     export OUTDIR="$(OUTDIR)"; \
>     export DATADIR="$(DATADIR)"; \
>     export ZEROSHOTPAIRS="$(ZEROSHOTPAIRS)"; \
>     export SUPERVISEDPAIRS="$(SUPERVISEDPAIRS)"; \
> module load cray-python; $(VIEWPYTHON) "$(SELFDIR)/inf_plan.py" 2>&1 | cat >"$(TESTCONFIG).err" || true; \
> echo "mk-yamls.mk:    The last 20 lines from $(TESTCONFIG).err:"; \
> echo -----------------------------------; \
> tail -20 "$(TESTCONFIG).err"; \
> echo -----------------------------------; \
> echo "mk-yamls.mk:    Logged to $(TESTCONFIG).err"; \
> echo "mk-yamls.mk:    Created   $(OUTDIR)/*.yaml"; \
> echo -n "mk-yamls.mk:    In $(OUTDIR) the number of files is "; \
> (ls $(OUTDIR)/*.yaml 2>/dev/null || true) | wc -l; 

inference-yamls-ok: $(TESTCONFIG).err 
> @if [ -f "$(TESTCONFIG).err" ] && grep -Fq 'All stages of planning completed' "$(TESTCONFIG).err"; then \
>               echo -n; \
> else \
>		echo "mk-yamls.mk: ❌ I found an incomplete config file"; \
>               echo "mk-yamls.mk:    $(TESTCONFIG).err"; \
>               rm -f "$(TESTCONFIG).err"; \
>               echo "mk-yamls.mk: ✨ Removed it. Now you can try to remake it."; \
>		exit 1 ; \
> fi

inference-yamls: inference-yamls-ok 
> @echo "mk-yamls.mk: ✅ Logged the inference YAML preparation to:";\
> echo "mk-yamls.mk:    $(TESTCONFIG).err";\
> echo "mk-yamls.mk: ✅ Created yaml files:";\
> ls $(OUTDIR)/*yaml  | wc | sed 's/^/mk-yamls.mk:    /';\
> echo "mk-yamls.mk: ✅ Lines in prepared call files:";\
> wc -l $(OUTDIR)/*.out  | sed 's/^/mk-yamls.mk:    /'

clean-yamls:
> rm -f "$(ZEROSHOTPAIRSINPUT)" "$(SUPERVISEDPAIRSINPUT)" "$(TESTCONFIG)" $(OUTDIR)/mt*yaml
> @echo "mk-yamls.mk: ✨ Cleaning done."; \
> echo "mk-yamls.mk: ❓ Do you want to rebuild thoses files? Say: "; \
> echo "mk-yamls.mk:       make mk-yamls"; \
> echo

