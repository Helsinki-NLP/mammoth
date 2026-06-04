# -----------------------------------------------------------------------------
# Shared checks
# -----------------------------------------------------------------------------
# mk-shared.mk
#
# Purpose
# -------
# Shared checks and common setup rules for the eval2 workflow.
#
# This Makefile fragment provides the basic validation and directory-setup
# targets that the rest of the eval2 pipeline depends on. It verifies that the
# model directory and shared resources exist, ensures that the local output
# directories are present, checks that the workflow is started from the correct
# model directory, and refuses to run the setup phase inside an existing Slurm
# job.
#
# Current behavior
# ----------------
# The fragment defines:
#
#   mk-shared
#       Runs the shared setup sequence:
#         - check-model
#         - dirs
#         - check-in-model-dir
#         - not-in-slurm
#
#   check-model
#       Verifies that:
#         - MODELDIR exists
#         - TESTINGDIR exists
#         - BINDIR exists
#         - DATADIR exists
#         - TRAINCONFIG exists and is non-empty
#
#   status
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
#   BINDIR
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
# The fragment prints status, validation results, and setup messages to stdout
# or stderr.
#
# Notes
# -----
# - This fragment is intended to be included by the top-level eval2 Makefile.
# - It provides the common preconditions for later pair extraction, planning,
#   inference, and scoring stages.
# - The setup phase is intended to be started outside Slurm.



.PHONY: check-model dirs check-in-model-dir not-in-slurm mk-shared

mk-shared: check-model dirs check-in-model-dir not-in-slurm
> @echo "mk-shared.mk: ✨ I am happy."

check-model:
> @set -euo pipefail
> @ [[ -d "$(MODELDIR)"    ]] || { echo "mk-shared.mk: ❌ Missing directory: $(MODELDIR)" >&2; exit 1; }
> @ [[ -d "$(TESTINGDIR)"  ]] || { echo "mk-shared.mk: ❌ Missing directory: $(TESTINGDIR)" >&2; exit 1; }
> @ [[ -d "$(BINDIR)"      ]] || { echo "mk-shared.mk: ❌ Missing directory: $(BINDIR)" >&2; exit 1; }
> @ [[ -d "$(DATADIR)"     ]] || { echo "mk-shared.mk: ❌ Missing directory: $(DATADIR)" >&2; exit 1; }
> @ [[ -s "$(TRAINCONFIG)" ]] || { echo "mk-shared.mk: ❌ Missing training config: $(TRAINCONFIG)" >&2; exit 1; }
> @echo "mk-shared.mk: ✅ Directories and the config file found"

status:
> @echo "mk-shared.mk: MODELDIR:          $(MODELDIR)"
> @echo "mk-shared.mk: TESTCONFIG:        $(TESTCONFIG)"
> @echo "mk-shared.mk: INFERENCE_CALLS:   $(INFERENCE_CALLS)"
> @echo "mk-shared.mk: SACRE_CALLS:       $(SACRE_CALLS)"
> @echo "mk-shared.mk: Inference flag:    $(INF_FLAG)"
> @echo "mk-shared.mk: Metrics flag:      $(MET_FLAG)"

check-in-model-dir: check-model 
> @set -euo pipefail
> @if [ -n "$${SLURM_JOBID:-}" ]; then \
>   submit_dir="$${SLURM_SUBMIT_DIR%/}"; \
>   model_dir="$(MODELDIR)"; \
>   model_dir="$${model_dir%/}"; \
>   if [ "$$submit_dir" != "$$model_dir" ]; then \
>     echo "Submit directory mismatch:" >&2; \
>     echo "  SLURM_SUBMIT_DIR=$$submit_dir" >&2; \
>     echo "  MODELDIR=$$model_dir" >&2; \
>     exit 1; \
>   fi; \
> else \
>   pwd_dir="$${PWD%/}"; \
>   model_dir="$(MODELDIR)"; model_dir="$${model_dir%/}"; \
>   if [ "$$pwd_dir" != "$$model_dir" ]; then \
>     echo "mk-shared.mk: 🛠️ Switching from directory:"; \
>     echo "mk-shared.mk:    $$pwd_dir"; \
>     echo "mk-shared.mk: 🛠️ To the working directory:"; \
>     echo "mk-shared.mk:    $(MODELDIR)"; \
>     cd "$(MODELDIR)"; \
>   fi; \
> fi
> @echo "mk-shared.mk: ✅ We are now in the model dir $(MODELDIR)"

dirs: check-model check-in-model-dir
> @mkdir -p "$(OUTDIR)" "$(LOGDIR)" "$(SCRDIR)"
> @echo "mk-shared.mk: 🛠️ I ensured the existence of local directories: "
> @echo "mk-shared.mk:    $(OUTDIR)"
> @echo "mk-shared.mk:    $(LOGDIR)"
> @echo "mk-shared.mk:    $(SCRDIR)"

not-in-slurm:
> @[[ -z "$${SLURM_JOBID:-}" ]] || { echo "mk-shared.mk: Run make outside SLURM first" >&2; exit 1; }

