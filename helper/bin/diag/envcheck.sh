#! /usr/bin/bash

# a stupid but useful way to check the environment
env | egrep '^(RCCL|FI_|NCCL_|PLUGIN|LD_LIB|HSA_|WITH_CONDA|CXI_|SLURM_|MIOPEN_)' | sort


