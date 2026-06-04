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
#       Verifies that `$(BINDIR)/inf_pairs.py` exists.
#
#   pair-planner
#       Verifies that `$(BINDIR)/inf_plan.py` exists.
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
#   BINDIR
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
#   - `$(BINDIR)/inf_pairs.py`
#   - `$(BINDIR)/inf_plan.py`
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


.PHONY: inference-yamls require-pairs clean-inf-inputs view-venv pair-extractor pair-planner fresh-yamls inference-yamls-ok

pair-extractor: 
> @[[ -f "$(BINDIR)/inf_pairs.py" ]] || { echo "mk-yamls.mk: ❌ Missing $(BINDIR)/inf_pairs.py" >&2; exit 1; }

pair-planner: 
> @ [[ -f "$(BINDIR)/inf_plan.py" ]] || { echo "mk-yamls.mk: Missing $(BINDIR)/inf_plan.py" >&2; exit 1; }

require-pairs: $(ZEROSHOTPAIRS) $(SUPERVISEDPAIRS)
> @echo "mk-yamls.mk: ✅ Pair selections exist: "
> @echo -n "mk-yamls.mk:    $(ZEROSHOTPAIRS):"
> @echo `wc -l <$(ZEROSHOTPAIRS)` lines
> @echo -n "mk-yamls.mk:    $(SUPERVISEDPAIRS):"
> @echo `wc -l <$(SUPERVISEDPAIRS)` lines

$(ZEROSHOTPAIRSINPUT): $(TRAINCONFIG) $(VIEWPYTHON) | dirs not-in-slurm pair-extractor
> @set -euo pipefail
> @echo "Building $(ZEROSHOTPAIRSINPUT)..."
> @module load cray-python; $(VIEWPYTHON) "$(BINDIR)/inf_pairs.py" "$(TRAINCONFIG)" --zs-out "$(ZEROSHOTPAIRSINPUT)" >&2

$(SUPERVISEDPAIRSINPUT): $(TRAINCONFIG) $(VIEWPYTHON) | dirs not-in-slurm pair-extractor
> @set -euo pipefail
> @echo "Building $(SUPERVISEDPAIRSINPUT)..."
> @module load cray-python; $(VIEWPYTHON) "$(BINDIR)/inf_pairs.py" "$(TRAINCONFIG)" --supervised-pairs-and-quit "$(SUPERVISEDPAIRSINPUT)" >&2

$(ZEROSHOTPAIRS): $(ZEROSHOTPAIRSINPUT)
> @echo "mk-yamls.mk: ❌ Missing zero-shot pair selection: $@" >&2
> @echo "mk-yamls.mk: Read suggestions from: $<" >&2
> @exit 1

$(SUPERVISEDPAIRS): $(SUPERVISEDPAIRSINPUT)
> @echo "mk-yamls.mk: ❌ Missing supervised pair selection: $@" >&2
> @echo "mk-yamls.mk: Read suggestions from: $<" >&2
> @exit 1

mk-yamls: require-pairs inference-yamls
> @echo "mk-yamls.mk: ✨ I am happy."
> @echo "mk-yamls.mk: ❓ Do you want to clean and rebuild thoses files? Say: "
> @echo "mk-yamls.mk:       make fresh-yamls        # to clean  testing.yaml"
> @echo "mk-yamls.mk:       make clean-inf-inputs   # to clean  .input files too"

$(TESTCONFIG).err: $(TRAINCONFIG) #| dirs not-in-slurm require-pairs pair-planner
> @set -euo pipefail
> @echo "mk-yamls.mk: 🛠️ Creating tentative testing tasks..."
> @echo -n "mk-yamls.mk:    In $(OUTDIR) the number of files is "
> @(ls $(OUTDIR)/*.yaml 2>/dev/null || true) | wc -l
> @   export BINDIR="$(BINDIR)"
> @echo -----------------------------------
> @   export MAMMOTH="$(MAMMOTH)"; \
>     export LOGDIR="$(LOGDIR)"; \
>     export SCRDIR="$(SCRDIR)"; \
>     export MODEL="$(MODEL)"; \
>     export TRAINCONFIG="$(TRAINCONFIG)"; \
>     export OUTDIR="$(OUTDIR)"; \
>     export DATADIR="$(DATADIR)"; \
>     export ZEROSHOTPAIRS="$(ZEROSHOTPAIRS)"; \
>     export SUPERVISEDPAIRS="$(SUPERVISEDPAIRS)"; \
> module load cray-python; $(VIEWPYTHON) "$(BINDIR)/inf_plan.py" 2>&1 | tee "$(TESTCONFIG).err" || true
> @echo -n "mk-yamls.mk:    In $(OUTDIR) the number of files is "
> @(ls $(OUTDIR)/*.yaml 2>/dev/null || true) | wc -l
> @echo -----------------------------------
> @echo    "mk-yamls.mk:    Logged to $(TESTCONFIG).err"
> @echo    "mk-yamls.mk:    Created   $(OUTDIR)/*.yaml"

inference-yamls-ok: $(TESTCONFIG).err # | dirs require-pairs
> @if [ -f "$(TESTCONFIG).err" ] && grep -Fq 'All stages of planning completed' "$(TESTCONFIG).err"; then \
>		echo "mk-yamls.mk: ✅ I found a succesfully completed file "; \
>               echo "mk-yamls.mk:    $(TESTCONFIG).err"; \
> else \
>		echo "mk-yamls.mk: ❌ I found an incomplete config file"; \
>               echo "mk-yamls.mk:    $(TESTCONFIG).err"; \
>               rm -f "$(TESTCONFIG).err"; \
>               echo "mk-yamls.mk: ✨ Removed it. Now you can try to remake it."; \
>		exit 1 ; \
> fi

inference-yamls: inference-yamls-ok $(TESTCONFIG).err # | dirs require-pairs
> @echo "mk-yamls.mk: ✅ Inference YAML preparation complete:"
> @echo "mk-yamls.mk:    $(TESTCONFIG).err"
> @echo -n "mk-yamls.mk:    The number of lines in this file: "
> @cat $(TESTCONFIG).err | wc -l
> @echo -n "mk-yamls.mk:    The number of test tasks in this file: "
> @egrep 'test_task:' $(TESTCONFIG).err | wc -l

clean-inf-inputs: 
> rm -f "$(ZEROSHOTPAIRSINPUT)" "$(SUPERVISEDPAIRSINPUT)" "$(TESTCONFIG)"

fresh-yamls:
> rm -f "$(TESTCONFIG)"
> @echo "mk-yamls.mk: ✨ Cleaning done."
> @echo "mk-yamls.mk: ❓ Do you want to rebuild thoses files? Say: "
> @echo "mk-yamls.mk:       make inference-yamls   # to rebuild  testing.yaml"
> @echo "mk-yamls.mk:       make                   # also works"

