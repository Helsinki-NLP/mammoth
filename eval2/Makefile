# The main targets in this Makefile are:
#   inference-yamls - constructs  "testing.yaml"
#   

# Makefile
#
# Purpose
# -------
# Top-level entry point for the eval2 workflow.
#
# This Makefile defines the shared and model-specific path layout for eval2 and
# includes the fragment Makefiles that implement the actual workflow stages.
# It is intended to be run from the eval2 directory with MODELDIR set to the
# model directory being evaluated.
#
# Current behavior
# ----------------
# The Makefile:
#   1) defines the shared project, testing, Mammoth, virtual-environment, data,
#      and script directories,
#   2) requires MODELDIR to be provided by the caller,
#   3) derives model-specific paths such as:
#        - TRAINCONFIG
#        - TESTCONFIG
#        - OUTDIR
#        - LOGDIR
#        - SCRDIR
#        - supervised and zero-shot pair files
#        - inference and scoring call-list files
#   4) includes the active workflow fragments:
#        - mk-shared.mk
#        - mk-yamls.mk
#   5) exposes the current top-level workflow target:
#        all
#      which currently runs:
#        - mk-shared
#        - mk-yamls
#        - mk-infer
#
# Inputs
# ------
# Required:
#   MODELDIR
#       Path to the model directory. Must be passed on the make command line.
#
# Shared paths
# ------------
#   DISKPROJECT
#   JOBPROJECT
#   TESTINGDIR
#   MAMMOTH
#   VENV
#   DATADIR
#
# Model-specific derived paths
# ----------------------------
#   MODEL
#   OUTDIR
#   LOGDIR
#   SCRDIR
#   TRAINCONFIG
#   TESTCONFIG
#   MAKESCRIPT
#   ZEROSHOTPAIRSINPUT
#   ZEROSHOTPAIRS
#   SUPERVISEDPAIRSINPUT
#   SUPERVISEDPAIRS
#   INFERENCE_CALLS
#   SACRE_CALLS
#   INF_FLAG
#   MET_FLAG
#   PLAN_FLAG
#   SACRE_FLAG
#
# Python / virtual-environment paths
# ----------------------------------
#   SACRE_VENV
#   SACRE_ACTIVATE
#   SACREPYTHON
#   VIEWPYTHON
#   INFPYTHON
#
# Typical use
# -----------
#   make MODELDIR=/path/to/model
#
# Notes
# -----
# - MODELDIR is mandatory; the Makefile stops immediately if it is not given.
# - The active pipeline currently covers the shared-setup and YAML-preparation
#   stages through the included fragments.
# - Runtime and scoring fragments are present in the repository but are not yet
#   included in this top-level Makefile.
# - This Makefile is the main entry point of the new eval2 pipeline.


.DEFAULT_GOAL := help

.RECIPEPREFIX := >
SHELL := /bin/bash
.SHELLFLAGS := -lc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

SELFDIR := /scratch/project_462000964/members/aylijyra/dev/mammoth/eval2
INSPECT_MODEL_FILES := /scratch/$(DISKPROJECT)/members/aylijyra/dev/mammoth/eval2/inspect-model-files.py

DISKPROJECT := project_462000964
JOBPROJECT  := project_462001087

SIF := /appl/local/laifs/containers/lumi-multitorch-u24r64f21m43t29-20260225_144743/lumi-multitorch-full-u24r64f21m43t29-20260225_144743.sif

#PARTITION  := small
#TIME       := 04:00:00
#MEM        := 8G

MAMMOTHDEF := /scratch/$(DISKPROJECT)/shared/mammoth-shared/mammoth-dev/mammoth
MAMMOTH64  := /scratch/$(DISKPROJECT)/shared/mammoth-shared/mammoth-64/mammoth
VENV       := /scratch/$(DISKPROJECT)/shared/mammoth-shared/.venv

TESTINGDIR := /scratch/$(DISKPROJECT)/shared/testing-shared
DATADIR    := $(TESTINGDIR)/data

# -----------------------------------------------------------------------------
# Model aliases
# -----------------------------------------------------------------------------
# Shared base directory for model aliases.
MODELS := /scratch/project_462000964/members/tiedeman/MARMoT/models
SODELS := /scratch/project_462000964/members/tiedeman/MARMoT/sandbox/tiedeman/oellm-lg

# Add model aliases here.
MODEL_docmt4denoisebase   := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-base/mammoth
MODEL_docmt4denoisesmall  := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-small/mammoth
MODEL_docmt4denoisexl     := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-xl/mammoth
MODEL_docmt4              := $(MODELS)/docmt-4pivots/mammoth
MODEL_finnish             := $(MODELS)/finnish/mammoth
#MODEL_mammothdocmt4       := $(SODELS)/mammoth-docmt4
#MODEL_mammothdocmtlarge   := $(SODELS)/mammoth-docmt-large
#MODEL_mammothflanmtlarge  := $(SODELS)/mammoth-flan-mt-large
#MODEL_mammothmtlarge      := $(SODELS)/mammoth-mt-large
#MODEL_mammothmt           := $(SODELS)/mammoth-mt
#MODEL_mammothmtsynth      := $(SODELS)/mammoth-mtsynth
#MODEL_oellmtranslate236   := $(SODELS)/oellm-translate2-36
#MODEL_sentmtdenoise       := $(SODELS)/sentmt-denoise/mammoth
#MODEL_sentmthalfsmall     := $(SODELS)/sentmt-halfsharedenc-small/mammoth
#MODEL_sentmthalfxl        := $(SODELS)/sentmt-halfsharedenc-xl/mammoth
MODEL_xtran               := /scratch/project_462001087/members/wangchao/train/compare/x_transformers/model
MODEL_ALIASES := docmt4denoisebase  docmt4denoisesmall docmt4denoisexl docmt4 finnish xtran
#	mammothdocmt4 mammothdocmtlarge mammothflanmtlarge mammothmtlarge mammothmt mammothmtsynth 
#	oellmtranslate236 sentmtdenoise sentmthalfsmall sentmthalfxl 
# The sandbox models have been deleted due to the lack of space.

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
    $(error Please specify exactly one model alias. Run 'make list-models' to see choices.)
  endif
endif
# Prevent workflow targets from running without a model, but allow help/list-models.
NON_MODEL_GOALS := $(filter-out help list-models $(MODEL_ALIASES),$(MAKECMDGOALS))
ifneq ($(strip $(NON_MODEL_GOALS)),)
  ifndef MODELDIR
    $(error MODELDIR is unset. Use 'make list-models' or 'make <alias> <target>' or 'make MODELDIR=/path/to/model <target>')
  endif
endif
MODELDIR_CLEAN := $(patsubst %/,%,$(MODELDIR))
MODEL := $(MODELDIR_CLEAN)/

OUTDIR     := $(MODELDIR)/inf_out
LOGDIR     := $(MODELDIR)/inf_logs
SCRDIR     := $(MODELDIR)/inf_scores

TESTCONFIG  := $(MODELDIR)/testing.yaml
MAKESCRIPT  := $(MODELDIR)/inf_make.sh

SACRE_CALLS     := $(OUTDIR)/calls.sacre.out

MET_FLAG   := $(MODELDIR)/metrics.submitted
PLAN_FLAG  := $(MODELDIR)/inference-planning.lock
SACRE_FLAG := $(MODELDIR)/metrics-planning.lock

SACRE_VENV := $(TESTINGDIR)/venvs/sacre
TORCH_VENV := $(TESTINGDIR)/venvs/torch

SACREPYTHON := $(TESTINGDIR)/venvs/sacre/bin/python
VIEWPYTHON  := $(TESTINGDIR)/venvs/view/bin/python
TORCHPYTHON := $(TESTINGDIR)/venvs/torch/bin/python
INFPYTHON   := $(VENV)/bin/python

# -----------------------------------------------------------------------------
# Top-level workflow
# -----------------------------------------------------------------------------

.DEFAULT_GOAL := list-models

# Make alias goals no-op targets so `make docmt4 mk-yamls` works.
.PHONY: all mk-shared mk-yamls mk-infer metrics status $(MODEL_ALIASES) help list-models 

help: 
> @printf "\n\
eval2 Makefile\n\
\n\
Usage:\n\
  make list-models\n\
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
  fresh-yamls\n\n\
Model aliases:\n"


list-models: help
> @$(foreach a,$(MODEL_ALIASES),printf "  %-10s -> %s\n" "$(a)" "$(MODEL_$(a))";)

# This is used to inpect the model properties - helps to choose the correct MAMMOTH version
$(MODEL_ALIASES):
> @if [ "$(words $(MAKECMDGOALS))" -eq 1 ]; then \
>   $(MAKE) --no-print-directory -C "$(SELFDIR)" MODELDIR="$(MODEL_$@)" inspect-model-summary; \
> fi

include mk-shared.mk

include mk-yamls.mk

include mk-infer.mk

include mk-score.mk

all: mk-shared mk-yamls mk-infer 
