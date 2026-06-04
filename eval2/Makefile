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
#   PROJECT
#   TESTINGDIR
#   MAMMOTH
#   VENV
#   DATADIR
#   BINDIR
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


.RECIPEPREFIX := >
SHELL := /bin/bash
.SHELLFLAGS := -lc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

PROJECT := project_462000964
#PARTITION  := small
#TIME       := 04:00:00
#MEM        := 8G

TESTINGDIR := /scratch/$(PROJECT)/shared/testing-shared
MAMMOTH    := /scratch/$(PROJECT)/shared/mammoth-shared/mammoth-dev/mammoth
VENV       := /scratch/$(PROJECT)/shared/mammoth-shared/.venv
DATADIR    := $(TESTINGDIR)/data
BINDIR     := $(TESTINGDIR)/bin

# MODELDIR   := 
ifndef MODELDIR
$(info The MOLDELDIR unset. Please call make with MODEL=/path)
$(info Here are the options:)
$(error I need to stop now.)
endif
MODELDIR_CLEAN := $(patsubst %/,%,$(MODELDIR))
MODEL := $(MODELDIR_CLEAN)/

OUTDIR     := $(MODELDIR)/inf_out
LOGDIR     := $(MODELDIR)/inf_logs
SCRDIR     := $(MODELDIR)/inf_scores

TRAINCONFIG := $(MODELDIR)/train.yaml
TESTCONFIG  := $(MODELDIR)/testing.yaml
MAKESCRIPT  := $(MODELDIR)/inf_make.sh

ZEROSHOTPAIRSINPUT    := $(MODELDIR)/inf_zeroshot.txt.input
ZEROSHOTPAIRS         := $(MODELDIR)/inf_zeroshot.txt
SUPERVISEDPAIRSINPUT  := $(MODELDIR)/inf_supervised.txt.input
SUPERVISEDPAIRS       := $(MODELDIR)/inf_supervised.txt


INFERENCE_CALLS := $(OUTDIR)/calls.out
SACRE_CALLS     := $(OUTDIR)/calls.sacre.out

INF_FLAG   := $(MODELDIR)/inference.submitted
MET_FLAG   := $(MODELDIR)/metrics.submitted
PLAN_FLAG  := $(MODELDIR)/inference-planning.lock
SACRE_FLAG := $(MODELDIR)/metrics-planning.lock

SACRE_VENV := $(TESTINGDIR)/venvs/sacre
SACRE_ACTIVATE := $(SACRE_VENV)/bin/activate

SACREPYTHON := $(TESTINGDIR)/venvs/sacre/bin/python
VIEWPYTHON  := $(TESTINGDIR)/venvs/view/bin/python
INFPYTHON   := $(VENV)/bin/python

# -----------------------------------------------------------------------------
# Top-level workflow
# -----------------------------------------------------------------------------

.PHONY: all mk-shared mk-yamls mk-infer metrics status

# mk-shared 
all: mk-shared mk-yamls mk-infer 

include mk-shared.mk
include mk-yamls.mk

# include mk-infer.mk
#metrics: metrics-submit
# include mk-score.mk

