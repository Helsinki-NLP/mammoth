.DEFAULT_GOAL := help
.RECIPEPREFIX := >
.SHELLFLAGS := -lc
SHELL := /bin/bash

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DISKPROJECT := project_462000964
JOBPROJECT  := project_462001087

SELFDIR := /scratch/project_462000964/members/aylijyra/dev/mammoth/eval2
INSPECT_MODEL_FILES := /scratch/$(DISKPROJECT)/members/aylijyra/dev/mammoth/eval2/inspect-model-files.py

SIF := /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

MAMMOTHDEF := /scratch/$(DISKPROJECT)/shared/mammoth-shared/mammoth-dev/mammoth
MAMMOTH64  := /scratch/$(DISKPROJECT)/shared/mammoth-shared/mammoth-64/mammoth
VENV       := /scratch/$(DISKPROJECT)/shared/mammoth-shared/.venv

TESTINGDIR := /scratch/$(DISKPROJECT)/shared/testing-shared
DATADIR    := $(TESTINGDIR)/data

include $(SELFDIR)/mk-model.mk

OUTDIR     := $(MODELDIR)/inf_out
LOGDIR     := $(MODELDIR)/inf_logs
SCRDIR     := $(MODELDIR)/inf_scores

TESTCONFIG  := $(MODELDIR)/testing.yaml
MAKESCRIPT  := $(MODELDIR)/inf_make.sh

MET_FLAG   := $(MODELDIR)/metrics.submitted
PLAN_FLAG  := $(MODELDIR)/inference-planning.lock
SACRE_FLAG := $(MODELDIR)/metrics-planning.lock

SACRE_VENV := $(TESTINGDIR)/venvs/sacre
TORCH_VENV := $(TESTINGDIR)/venvs/torch

INF_ACTIVATE    := $(VENV)/bin/activate
SACRE_ACTIVATE  := $(TESTINGDIR)/venvs/sacre/bin/activate
VIEW_ACTIVATE   := $(TESTINGDIR)/venvs/view/bin/activate

SACREPYTHON  := $(TESTINGDIR)/venvs/sacre/bin/python
VIEWPYTHON   := $(TESTINGDIR)/venvs/view/bin/python
TORCHPYTHON  := $(TESTINGDIR)/venvs/torch/bin/python
INFPYTHON    := $(VENV)/bin/python

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

INF_TEMPLATE := $(SELFDIR)/slurm_inf.templ
CNT_TEMPLATE := $(SELFDIR)/slurm_cnt.templ
MET_TEMPLATE := $(SELFDIR)/slurm_met.templ

INF_SCRIPT   := $(OUTDIR)/inf.slurm
CNT_SCRIPT   := $(OUTDIR)/cnt.slurm
MET_SCRIPT   := $(OUTDIR)/met.slurm

INF_SBATCH   := $(OUTDIR)/inf.sbatch
CMT_SBATCH   := $(OUTDIR)/cmt.sbatch
MET_SBATCH   := $(OUTDIR)/met.sbatch

# -----------------------------------------------------------------------------
# Top-level workflow
# -----------------------------------------------------------------------------

.DEFAULT_GOAL := list

# Make alias goals no-op targets so `make docmt4 mk-yamls` works.
.PHONY: all mk-shared mk-yamls mk-infer metrics status $(MODEL_ALIASES) help list 

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
  make docmt4 mk-shared\n\
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

# This is used to inpect the model properties - helps to choose the correct MAMMOTH version
$(MODEL_ALIASES):
> @if [ "$(words $(MAKECMDGOALS))" -eq 1 ]; then \
>   $(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODEL_$@)" inspect-model-summary; \
> fi
> @echo Remember to mount the project directories when using containers

include $(SELFDIR)/mk-shared.mk
include $(SELFDIR)/mk-slurm.mk
include $(SELFDIR)/mk-pairs.mk
include $(SELFDIR)/mk-calls.mk
include $(SELFDIR)/mk-infer.mk
include $(SELFDIR)/mk-score.mk
include $(SELFDIR)/mk-control.mk


all: mk-infer 

.PHONY: status
status:
> @module load cray-python; python3 "$(SELFDIR)/status.py" $(STATUS_MODELS)
