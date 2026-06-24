# -----------------------------------------------------------------------------

# Basic checks
# -----------------------------------------------------------------------------
# mk-basic.mk
#
# Purpose
# -------
# Basic checks and common setup rules for the eval2 workflow.
#
# This Makefile fragment provides the basic validation and directory-setup
# targets that the rest of the eval2 pipeline depends on. It verifies that the
# model directory and basic resources exist, ensures that the local output
# directories are present, checks that the workflow is started from the correct
# model directory, and refuses to run the setup phase inside an existing Slurm
# job.
#
# Current behavior
# ----------------
# The fragment defines:
#
#   mk-basic
#       Runs the basic setup sequence:
#         - check-model
#         - dirs
#         - check-in-model-dir
#         - not-in-slurm
#
#   check-model
#       Verifies that:
#         - MODELDIR exists
#         - TESTINGDIR exists
#         - DATADIR exists
#         - TRAINCONFIG exists and is non-empty
#
#   model-status
#       Prints the current values of key derived workflow paths such as:
#         - MODELDIR
#         - TESTCONFIG
#         - INFERENCE_CALLS
#         - SACRE_CALLS
#         - INF_FLAG
#         - MET_FLAG
#
#   check-in-model-dir
#       Verifies that the workflow is being run from the correct model
#       directory:
#         - inside Slurm, it compares SLURM_SUBMIT_DIR with MODELDIR
#         - outside Slurm, it checks PWD and switches to MODELDIR if needed
#
#   dirs
#       Ensures that the model-local directories exist:
#         - OUTDIR
#         - LOGDIR
#         - SCRDIR
#
#   not-in-slurm
#       Fails if the current process is already running inside Slurm.
#
# Inputs
# ------
# The fragment expects the including Makefile to define:
#   MODELDIR
#   TESTINGDIR
#   DATADIR
#   TRAINCONFIG
#   OUTDIR
#   LOGDIR
#   SCRDIR
#   TESTCONFIG
#   INFERENCE_CALLS
#   SACRE_CALLS
#   INF_FLAG
#   MET_FLAG
#
# Reads
# -----
#   - filesystem paths derived by the top-level Makefile
#   - environment variables such as:
#       - PWD
#       - SLURM_JOBID
#       - SLURM_SUBMIT_DIR
#
# Writes / effects
# ----------------
#   - creates OUTDIR, LOGDIR, and SCRDIR if missing
#   - may switch the working directory to MODELDIR when running outside Slurm
#
# Output
# ------
# The fragment prints model-status, validation results, and setup messages to stdout
# or stderr.
#
# Notes
# -----
# - This fragment is intended to be included by the top-level eval2 Makefile.
# - It provides the common preconditions for later pair extraction, planning,
#   inference, and scoring stages.
# - The setup phase is intended to be started outside Slurm.



.PHONY: check-model dirs not-in-slurm mk-basic inspect-model-summary mk-basic ensure-sacre-venv

inspect-model-summary: $(MODELDIR)/model-summary.yaml
> @echo "mk-basic.mk: 🛠️ Short model-file summary for $(MODELDIR):"; \
> cat $(MODELDIR)/model-summary.yaml

$(MODELDIR)/mammoth.selected: $(MODELDIR)/model-summary.yaml
> @set -euo pipefail; \
> echo "mk-basic.mk:    Reading $(MODELDIR)/model-summary.yaml"; \
> if grep -Fq 'use older Mammoth compatible with trained_head_dim=64' "$(MODELDIR)/model-summary.yaml"; then \
>   echo "mk-basic.mk:    The model has a hard-coded head_dim"; \
>   printf '%s\n' "$(MAMMOTH64)" > "$@"; \
> else \
>   echo "mk-basic.mk:    The model uses a computed head_dim"; \
>   printf '%s\n' "$(MAMMOTHDEF)" > "$@"; \
> fi; \
> echo "mk-basic.mk: ✅ Written file $@"

clean-basic:
> rm $(MODELDIR)/model-summary.yaml $(MODELDIR)/mammoth.selected
> @echo "mk-basic.mk: ✨ Cleaning done."

# The folloing sends the names of the model files to the inspector and
# then stores the outputs to a model-summary.yaml file.
$(MODELDIR)/model-summary.yaml: 
> @set -euo pipefail
> @[[ -f "$(INSPECT_MODEL_FILES)" ]] || { echo "mk-basic.mk: ❌ Missing $(INSPECT_MODEL_FILES)" >&2; exit 1; }
> @shopt -s nullglob; \
> files=( "$(MODELDIR)"/*.pt ); \
> keep=(); \
> for f in "$${files[@]}"; do \
>   b="$$(basename "$$f")"; \
>   case "$$b" in \
>     *_optim.pt) ;; \
>     *) keep+=( "$$f" );; \
>   esac; \
> done; \
> if [ "$${#keep[@]}" -eq 0 ]; then \
>   echo "mk-basic.mk: ℹ️ No inspectable .pt model files found in $(MODELDIR)"; \
>   exit 0; \
> fi; \
> module load cray-python; \
> /usr/bin/singularity exec \
>   -B "/scratch/$(DISKPROJECT):/scratch/$(DISKPROJECT):rw" \
>   --env PYTHONPATH="$(MAMMOTHDEF):$${PYTHONPATH:-}" \
>   "$(SIF)" \
>   python3 "$(INSPECT_MODEL_FILES)" --unsafe --depth 2 --top-mods 8 --model-summary --yaml-like "$${keep[@]}" \
>	> $(MODELDIR)/model-summary.yaml

$(MODELDIR) $(TESTINGDIR) $(DATADIR) $(TRAINCONFIG): 
> @set -euo pipefail; \
> [[ -d "$(MODELDIR)"    ]] || { echo "mk-basic.mk: ❌ Missing directory: $(MODELDIR)" >&2; exit 1; }; \
> [[ -d "$(TESTINGDIR)"  ]] || { echo "mk-basic.mk: ❌ Missing directory: $(TESTINGDIR)" >&2; exit 1; }; \
> [[ -d "$(DATADIR)"     ]] || { echo "mk-basic.mk: ❌ Missing directory: $(DATADIR)" >&2; exit 1; }; \
> [[ -s "$(TRAINCONFIG)" ]] || { echo "mk-basic.mk: ❌ Missing training config: $(TRAINCONFIG)" >&2; exit 1; }; \
> echo "mk-basic.mk: ✅ Directories and the config file found"

$(BAS_DONE): $(MODELDIR) $(TESTINGDIR) $(DATADIR) $(TRAINCONFIG) $(MODELDIR)/mammoth.selected $(SACRE_ACTIVATE) $(OUTDIR) $(LOGDIR) $(SCRDIR)
> @MAMMOTHSEL="$$(cat "$(MODELDIR)/mammoth.selected")"; \
> echo "mk-basic.mk:    Using: $$MAMMOTHSEL"; \
> echo "mk-basic.mk:    Ensured sacrebleu virtual environment"; \
> echo "mk-basic.mk:    Ensured the existence of local inf_{out,logs,scores} directories"; \
> echo "mk-pairs.mk: ✨ I am happy with the basics."; \
> touch $(BAS_DONE)

mk-basic: $(BAS_DONE) 
> @echo "mk-basic.mk:    MODELDIR:          $(MODELDIR)"; \
> echo "mk-basic.mk:    TESTCONFIG:        $(TESTCONFIG)"; \
> [[ -z "$${SLURM_JOBID:-}" ]] || { echo "mk-basic.mk: Run make outside SLURM first" >&2; exit 1; }
> @echo "mk-basic.mk: ✅ Running outside SLURM"; \
> echo "mk-basic.mk: ✨ I am happy with the preparations (dirs, venvs, version)."; \
> echo

$(OUTDIR) $(LOGDIR) $(SCRDIR): 
> @mkdir -p "$(OUTDIR)" "$(LOGDIR)" "$(SCRDIR)"

$(SACRE_ACTIVATE): 
> @set -euo pipefail; \
> module load cray-python; \
> if [ ! -f "" ]; then \
>   echo "mk-basic.mk:    Building sacrebleu virtual environment..."; \
>   python3 -m venv "$(SACRE_VENV)"; \
>   source "$(SACRE_ACTIVATE)"; \
>   python -m pip install --upgrade pip; \
>   pip install sacrebleu; \
>   echo "Built scoring venv $(SACRE_VENV)"; \
> fi

