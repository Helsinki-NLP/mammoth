# -----------------------------------------------------------------------------
# Inference command calls planning 
# -----------------------------------------------------------------------------

.PHONY: check-inference-env fresh-inference inference-plan 

fresh-inference:
> rm -f "$(INFERENCE_CALLS)"

check-inference-env:
> @set -euo pipefail
> @ : "${BINDIR:?BINDIR must be set}"
> @ : "${LOGDIR:?LOGDIR must be set}"
> @ : "${SCRDIR:?SCRDIR must be set}"
> @ [[ -d "$(BINDIR)"                     ]] || { echo "Missing directory: $(BINDIR)" >&2; exit 1; }
> @ [[ -d "$(LOGDIR)"                     ]] || { echo "Missing directory: $(LOGDIR)" >&2; exit 1; }
> @ [[ -d "$(SCRDIR)"                     ]] || { echo "Missing directory: $(SCRDIR)" >&2; exit 1; }
> @ [[ -f "$(BINDIR)/inf_plan.sh"         ]] || { echo "Missing planner: inf_plan.sh" >&2; exit 1; }
> @ [[ -f "$(BINDIR)/inf_wrapper.sh"      ]] || { echo "Missing wrapper: inf_wrapper.sh" >&2; exit 1; }
> @ [[ -f "$(BINDIR)/inf_distr.sh"        ]] || { echo "Missing distributor: inf_distr.sh" >&2; exit 1; }
> @echo "✅ Found the directories and the component scripts inf_{plan,distr,wrapper}.sh"

inference-plan-lock: 
> @ if [ -f "$(PLAN_FLAG)" ]; then \
>   echo "Another process has locked inference planning: $(PLAN_FLAG)" >&2; \
>   exit 1; \
> fi

inference-plan: inference-yamls check-inference-env inference-plan-lock
> @set -euo pipefail
> @echo "🛠️ Planning inference tasks..."
> @touch "$(PLAN_FLAG)"
> @trap 'rm -f "$(PLAN_FLAG)"' EXIT
> @echo -n "The number of tasks in the testing.yaml:                "
> @egrep 'task:' $(TESTCONFIG) | wc -l
> @echo -n "The number of config files in the output directory:     "
> @ls "$(OUTDIR)/*.yaml" | wc -l
> @echo -n "The number of hypothesis files in the output directory: "
> @ls "$(OUTDIR)/*.hyp" | wc -l
> @export TESTINGDIR="$(TESTINGDIR)"; \
> export MAMMOTH="$(MAMMOTH)"; \
> export VENV="$(VENV)"; \
> export DATADIR="$(DATADIR)"; \
> export BINDIR="$(BINDIR)"; \
> export MODEL="$(MODEL)"; \
> export OUTDIR="$(OUTDIR)"; \
> export LOGDIR="$(LOGDIR)"; \
> export SCRDIR="$(SCRDIR)"; \
> export TRAINCONFIG="$(TRAINCONFIG)"; \
> export ZEROSHOTPAIRS="$(ZEROSHOTPAIRS)"; \
> export SUPERVISEDPAIRS="$(SUPERVISEDPAIRS)"; \
> # bash "$(BINDIR)/inf_plan.sh"
> @echo "PLACEHOLDER: create inference commands in $(INFERENCE_CALLS)"
> @echo "PLACEHOLDER: skip commands whose hypothesis outputs already exist"
> @echo "PLACEHOLDER: plan distribution across ntasks/gpus/time allocation"
> @echo "PLACEHOLDER: write runnable task commands to $(INFERENCE_CALLS)"
> # Example future command:
> # python "$(BINDIR)/make_inference_calls.py" \
> #   --testing-yaml "$(TESTCONFIG)" \
> #   --model "$(MODEL)" \
> #   --outdir "$(OUTDIR)" \
> #   --calls-out "$(INFERENCE_CALLS)"
> #
> # For now, create an empty placeholder if missing:
#> touch "$(INFERENCE_CALLS)"
#> if [ ! -s "$(INFERENCE_CALLS)" ]; then \
#>   echo "No inference commands currently planned in $(INFERENCE_CALLS)"; \
#> fi

