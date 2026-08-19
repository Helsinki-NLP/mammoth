.DEFAULT_GOAL := help
.RECIPEPREFIX := >
.SHELLFLAGS := -lc
SHELL := /bin/bash

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

JOBPROJECT  := project_462001087

# SIF := /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif
SIF := /appl/local/laifs/containers/lumi-multitorch-u24r70f21m50t210-20260513_121430/lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif

DISKPROJECT := project_462001087
SELFDIR     := /scratch/$(DISKPROJECT)/members/aylijyra/dev/mammoth/eval2
MAMMOTHDEF  := /scratch/$(DISKPROJECT)/shared/mammoth-shared/mammoth-dev/mammoth
MAMMOTH64   := /scratch/$(DISKPROJECT)/shared/mammoth-shared/mammoth-64/mammoth
VENV        := /scratch/$(DISKPROJECT)/shared/mammoth-shared/.venv
TESTINGDIR  := /scratch/$(DISKPROJECT)/shared/testing-shared

DATADIR    := $(TESTINGDIR)/data

include $(SELFDIR)/include/mk-model.mk

export $(addprefix MODEL_,$(MODEL_ALIASES))
STATUS_MODELS := $(foreach a,$(MODEL_ALIASES),--model "$(a)=$(MODEL_$(a))")

# Resolve alias used as the first goal, if any.
FIRST_GOAL := $(firstword $(MAKECMDGOALS))
ifneq ($(filter $(FIRST_GOAL),$(MODEL_ALIASES)),)
  MODELDIR := $(MODEL_$(FIRST_GOAL))
  MODEL_ALIAS_USED := 1
else
  MODEL_ALIAS_USED := 0
endif
# If any alias appears later in the goal list, reject it.
LATE_MODEL_GOALS := $(filter $(MODEL_ALIASES),$(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))
ifneq ($(strip $(LATE_MODEL_GOALS)),)
  $(error Put the model alias first, e.g. 'make $(firstword $(LATE_MODEL_GOALS)) mk-yamls')
endif
# If more than one alias appears anywhere, reject it.
ALL_MODEL_GOALS := $(filter $(MODEL_ALIASES),$(MAKECMDGOALS))
ifneq ($(words $(ALL_MODEL_GOALS)),0)
  ifneq ($(words $(ALL_MODEL_GOALS)),1)
    $(error Please specify exactly one model alias. Run 'make list' to see choices.)
  endif
endif
# Prevent workflow targets from running without a model, but allow help/list.
NON_MODEL_GOALS := $(filter-out help venvs list status continue-all continue-all-force test-sif find-sif plot-docmt $(MODEL_ALIASES),$(MAKECMDGOALS))
ifneq ($(strip $(NON_MODEL_GOALS)),)
  ifndef MODELDIR
    $(error MODELDIR is unset. Use 'make list' or 'make <alias> <target>' or 'make MODELDIR=/path/to/model <target>')
  endif
endif
MODELDIR_CLEAN := $(patsubst %/,%,$(MODELDIR))
MODEL := $(MODELDIR_CLEAN)/



OUTDIR     := $(MODELDIR)/inf_out
LOGDIR     := $(MODELDIR)/inf_logs
SCRDIR     := $(MODELDIR)/inf_scores

TESTCONFIG  := $(MODELDIR)/testing.yaml
MAKESCRIPT  := $(MODELDIR)/inf_make.sh

MET_FLAG   := $(MODELDIR)/metrics.submitted
PLAN_FLAG  := $(MODELDIR)/inference-planning.lock
SACRE_FLAG := $(MODELDIR)/metrics-planning.lock

SACRE_VENV := $(TESTINGDIR)/venvs/sacre
PLOT_VENV := $(TESTINGDIR)/venvs/plot
TORCH_VENV := $(TESTINGDIR)/venvs/torch

INF_ACTIVATE    := $(VENV)/bin/activate
SACRE_ACTIVATE  := $(TESTINGDIR)/venvs/sacre/bin/activate
PLOT_ACTIVATE  := $(TESTINGDIR)/venvs/plot/bin/activate
VIEW_ACTIVATE   := $(TESTINGDIR)/venvs/view/bin/activate

PLOTPYTHON  := $(TESTINGDIR)/venvs/plot/bin/python
SACREPYTHON  := $(TESTINGDIR)/venvs/sacre/bin/python
VIEWPYTHON   := $(TESTINGDIR)/venvs/view/bin/python
TORCHPYTHON  := $(TESTINGDIR)/venvs/torch/bin/python
INFPYTHON    := $(VENV)/bin/python

BAS_DONE     := $(MODELDIR)/preps.done
PRS_DONE     := $(MODELDIR)/pairs.done
INF_DONE     := $(MODELDIR)/inference.done
CNT_DONE     := $(MODELDIR)/cnt.done
MET_DONE     := $(MODELDIR)/metrics.done
CMT_DONE     := $(MODELDIR)/comet.done
VIZ_DONE     := $(MODELDIR)/viz.done

INF_FLAG     := $(MODELDIR)/inference.submitted
CNT_FLAG     := $(MODELDIR)/cnt.submitted
MET_FLAG     := $(MODELDIR)/metrics.submitted
CMT_FLAG     := $(MODELDIR)/comet.submitted
VIZ_FLAG     := $(MODELDIR)/viz.submitted

INF_CALLS    := $(OUTDIR)/calls.out
SACRE_CALLS  := $(OUTDIR)/calls.sacre.out
COMET_CALLS  := $(OUTDIR)/calls.comet.out

INF_TEMPLATE := $(SELFDIR)/template/slurm_inf.templ
CNT_TEMPLATE := $(SELFDIR)/template/slurm_cnt.templ
MET_TEMPLATE := $(SELFDIR)/template/slurm_met.templ

INF_SCRIPT   := $(OUTDIR)/inf.slurm
CNT_SCRIPT   := $(OUTDIR)/cnt.slurm
MET_SCRIPT   := $(OUTDIR)/met.slurm

INF_SBATCH   := $(OUTDIR)/inf.sbatch
CMT_SBATCH   := $(OUTDIR)/cmt.sbatch
MET_SBATCH   := $(OUTDIR)/met.sbatch

INSPECT_MODEL_FILES := $(SELFDIR)/bin/inspect-model-files.py

# -----------------------------------------------------------------------------
# Top-level workflow
# -----------------------------------------------------------------------------

.DEFAULT_GOAL := list

# Make alias goals no-op targets so `make docmt4 mk-yamls` works.
.PHONY: all mk-basic mk-yamls mk-infer metrics status $(MODEL_ALIASES) help list 

help: 
> @printf "\n\
eval2 Makefile\n\
\n\
Usage:\n\
  make list\n\
  make <model-alias> <target>\n\
  make MODELDIR=/path/to/model <target>\n\
\n\
The model alias must be specified first.\n\
\n\
Examples:\n\
  make docmt4 mk-basic\n\
  make docmt4 mk-yamls\n\
  make sentmtdenoise mk-yamls\n\
  make MODELDIR=/scratch/.../mammoth mk-yamls\n\
\n\
Main targets:\n\
  mk-shared\n\
  mk-yamls\n\
  status\n\
  clean-inf-inputs\n\
  fresh-yamls\n\n"

list: 
> @$(foreach a,$(MODEL_ALIASES),printf "  %-10s -> %s\n" "$(a)" "$(MODEL_$(a))";)
> @echo Remember to mount the project directories in slurm_*.templ files when using containers

# This is used to inpect the model properties - helps to choose the correct MAMMOTH version
$(MODEL_ALIASES):
> @if [ "$(words $(MAKECMDGOALS))" -eq 1 ]; then \
>   $(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODEL_$@)" inspect-model-summary; \
> fi

include $(SELFDIR)/include/mk-basic.mk # venvs, model inspection, mammoth ver, dirs, (non)slurm detection
include $(SELFDIR)/include/mk-pairs.mk # determine the desired language pairs
include $(SELFDIR)/include/mk-calls.mk # calls.*.out and yaml files
include $(SELFDIR)/include/mk-slurm.mk # slurm scripts and sbatch lines
include $(SELFDIR)/include/mk-infer.mk # run inference calls
include $(SELFDIR)/include/mk-score.mk # run scoring calls
include $(SELFDIR)/include/mk-contr.mk # loop runs until done

all: mk-infer 

.PHONY: status mt-bleu mt-chrf mt-bleu-chrf test-sif find-sif

status:
> @module load cray-python; python3 "$(SELFDIR)/bin/status.py" $(STATUS_MODELS)

mt-bleu:
> @module load cray-python; python "$(SELFDIR)/bin/summarize_sacre.py" --metric BLEU --kind mt $(MODELDIR)/inf_scores/

mt-chrf:
> @module load cray-python; python "$(SELFDIR)/bin/summarize_sacre.py" --metric chrF2 --kind mt $(MODELDIR)/inf_scores/

mt-bleu-chrf:
> @module load cray-python; python "$(SELFDIR)/bin/summarize_sacre.py" --kind mt $(MODELDIR)/inf_scores/

test-sif:
> srun --partition=dev-g --account=$(JOBPROJECT) --time=00:10:00 \
>  --nodes=1 --ntasks=1 --gpus-per-node=1 \
>  --pty bash -lc "set -euxo pipefail; /usr/bin/singularity inspect '$(SIF)'; /usr/bin/singularity exec '$(SIF)' python --version"

find-sif:
> srun --partition=dev-g --account=$(JOBPROJECT) --time=00:05:00 \
>  --nodes=1 --ntasks=1 --gpus-per-node=1 \
>  --pty bash -lc 'set -euo pipefail; find -L /appl/local/laifs/containers -maxdepth 3 -type f \( -name "*multitorch*.sif" -o -name "*torch*.sif" \) -print'

plot-docmt:
> $(PLOTPYTHON) compare.py \
>  --mk-models include/mk-model.mk \
>  --kind mt \
>  --dataset bqtpar \
>  --models docmt4denhalfbase docmt4denhalfsmall docmt4denhalfxl docmt10denhalf \
>  --delta docmt10denhalf docmt4denhalfxl \
>  --outdir cmp_docmt
> cat cmp_docmt/dominance_pretty.txt
> cat cmp_docmt/dominance_percent.txt
> cat cmp_docmt/condorcet.txt

plot-sentmt:
> $(PLOTPYTHON) compare.py   --mk-models include/mk-model.mk   --kind sentmt   \
>   --dataset bqtpar   --models sentmtdenoise sentmthalfsmall sentmthalfxl  \
>   --delta sentmthalfsmall sentmthalfxl   --outdir cmp_sentmt

