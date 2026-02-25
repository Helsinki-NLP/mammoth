#!/usr/bin/env bash
# Written by Anssi Yli-Jyrä for MAMMOTH framework @ lumi.csc.fi, (c) 2025, MIT License
# see the documentation in the end of the file.
# Usage:
#    sbatch-dev-g   -A <accoount> -J <job-name> <this-file> 
#    sbatch-small-g -A <accoount> -J <job-name> <this-file> 

#SBATCH --time=0-04:00:00                         
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8                 # preferred way to specify GPUs

# DO NOT CHANGE THE END OF THE SCRIPT REGULARLY

#SBATCH --partition=__MUST_USE_WRAPPER__  # keep; use e.g. tools/helper/bin/sbatch-dev-g to auto-set 
#SBATCH --mail-user=__MUST_USE_WRAPPER    # keep; use e.g. tools/helper/bin/sbatch-dev-g to auto-set

#SBATCH --account=__MUST_USE_WRAPPER__    # keep; use the command line option -A
#SBATCH --job-name=__MUST_USE_WRAPPER__   # keep; use the command line option --job-name

#SBATCH --mail-type=ALL                   # keep
#SBATCH --ntasks-per-node=1               # keep; MAMMOTH style
#SBATCH --mem-per-gpu=60G                 # keep; recommended: 60G/GPU (with 4GiB slack for the system)
                                          # => --mem = nodes x gpus-per-node x mem-per-gpu
#SBATCH --cpus-per-gpu=7                  # keep; recommended: 7CPUs/GPU x GPUs
#SBATCH --hint=nomultithread              # keep; prefer 1 hw-thread per core; MAMMOTH is GPU-bound

#SBATCH --output=jobs/%j/out               # keep; all outputs in one dir
#SBATCH  --error=jobs/%j/err               # keep; all outputs in one dir

###################################################
#         BASH SCRIPT STARTS HERE                 #
###################################################

mkdir -p          "jobs/${SLURM_JOB_ID}"  # ensure the job dir used by --output=job/%j/{out,err}
export OUTPUT_DIR="jobs/${SLURM_JOB_ID}"  # use it for other outputs too

echo "Starting the sbatch at `date`" 1>&2
set -Eeuo pipefail
trap 'echo "sbatch script ($BASH_COMMAND) has an error at line $LINENO" >&2' ERR

###################################################
#              ENVIRONMENT VARIABLES              #
###################################################
require_set() {
  local v
  for v; do
    # ${!v-} expands to empty if unset (safe with set -u)
    if [[ -z "${!v-}" ]]; then
      printf '❌ variable %s must be set\n' "$v" >&2
      return 1
    fi
  done
}
require_set  SLURM_JOB_NAME SLURM_JOB_ACCOUNT SLURM_JOB_PARTITION \
             SLURM_JOB_NUM_NODES SLURM_NTASKS_PER_NODE SLURM_GPUS_PER_NODE \
	     SLURM_CPUS_PER_GPU SLURM_MEM_PER_GPU \
	     SLURM_JOB_ID SLURM_JOB_NODELIST \
    || exit 1;
if [ "${DRY_RUN:-0}" = 1 ]; then
    : "${mail_user:=$MAIL_USER}" 
else
    mail_user=$(
	scontrol show job "$SLURM_JOB_ID" -o |
	    sed -n 's/.*MailUser=\([^ ]*\).*/\1/p'
	     )
    : "${mail_user:=$USER}" # fallback: if empty, Slurm’s default is the submitting user
fi
export TIME_LIMIT=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
echo ==================================================================       1>&2
echo " SLURM_JOB_NAME        (--job-name)        : ${SLURM_JOB_NAME:-}"       1>&2
echo " SLURM_JOB_ID             (automatic)      : ${SLURM_JOB_ID:-}"         1>&2
echo " SLURM_SUBMIT_DIR         (automatic)      : ${SLURM_SUBMIT_DIR:-}"     1>&2
echo " JOB_SBAT_SCRIPT_NAME  (CLI argument)      : ${JOB_SBAT_SCRIPT_NAME:-}" 1>&2
echo " SLURM_JOB_ACCOUNT     (--account)         : ${SLURM_JOB_ACCOUNT:-}"    1>&2
echo " MailUser              (--mail-user)       : ${mail_user:-}"            1>&2
echo ==================================================================       1>&2
echo " TIME_LIMIT               (--time)         : ${TIME_LIMIT:-}"           1>&2
echo " SLURM_JOB_PARTITION   (--partition)       : ${SLURM_JOB_PARTITION:-}"  1>&2
echo " SLURM_JOB_NUM_NODES   (--nodes)           : ${SLURM_JOB_NUM_NODES:-}"  1>&2
echo " SLURM_NNODES            (computed)        : ${SLURM_NNODES:-}"         1>&2
echo " SLURM_JOB_NODELIST      (automatic)       : ${SLURM_JOB_NODELIST:-}"   1>&2
echo ==================================================================       1>&2
echo " SLURM_NTASKS_PER_NODE (--ntasks-per-node) : ${SLURM_NTASKS_PER_NODE:-}" 1>&2
echo " SLURM_TASKS_PER_NODE    (computed)        : ${SLURM_TASKS_PER_NODE:-}"  1>&2
echo " SLURM_NTASKS            (computed)        : ${SLURM_NTASKS:-}"          1>&2
echo ==================================================================        1>&2
echo " SLURM_GPUS_PER_NODE   (--gpus-per-node)   : ${SLURM_GPUS_PER_NODE:-}"   1>&2
echo " SLURM_GPUS_ON_NODE      (computed)        : ${SLURM_GPUS_ON_NODE:-}"    1>&2
echo " SLURM_JOB_GPUS          (computed)        : ${SLURM_JOB_GPUS:-}"        1>&2
echo " SLURM_CPUS_PER_GPU    (--cpus-per-gpu)    : ${SLURM_CPUS_PER_GPU:-}"    1>&2
echo " SLURM_CPUS_ON_NODE      (computed)        : ${SLURM_CPUS_ON_NODE:-}"    1>&2
echo " SLURM_MEM_PER_GPU     (--mem-per-gpu)     : ${SLURM_MEM_PER_GPU:-}"     1>&2
echo " SLURM_HINT            (--hint)            : ${SLURM_HINT:-}"            1>&2

MAMMOTH=/scratch/project_462000964/shared/mammoth_helper # set this according to your installation
HELPER=$MAMMOTH/tools/helper/
export COMMS_DEBUG=1
export RUN_CONFIG="$(readlink -f config.yaml)"   # find out the real config.yaml file
export RUN_SCRIPT=$MAMMOTH/train.py  
export RUN_ARGS="--config ${RUN_CONFIG} --save_model ${OUTPUT_DIR}/model" 
export ALLOC_DIR="${SLURM_JOB_PARTITION}-${SLURM_JOB_NUM_NODES}N${SLURM_GPUS_ON_NODE:-0}g${SLURM_CPUS_ON_NODE:-0}c"
export UNBUFFERED=                               # see the documentation why

echo================================================================== 1>&2
echo " DRY_RUN                                   : ${DRY_RUN:-}"       1>&2
echo " COMMS_DEBUG                               : ${COMMS_DEBUG:-}"   1>&2
echo " UNBUFFERED    (--unbuffered ?)            : $UNBUFFERED"        1>&2 
echo ================================================================= 1>&2
echo " MAMMOTH     : $MAMMOTH"      1>&2
echo " HELPER      : $HELPER"       1>&2
echo " RUN_SCRIPT  : $RUN_SCRIPT"   1>&2
echo " RUN_CONFIG  : $RUN_CONFIG"   1>&2
echo " RUN_ARGS    : $RUN_ARGS"     1>&2
echo " OUTPUT_DIR  : $OUTPUT_DIR"   1>&2
echo " ALLOC_DIR   : $ALLOC_DIR"    1>&2

###################################################
#       organize outputs under ALLOC_DIR
###################################################
if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "(We do not organize the dry-run outputs to the directory tree)"  1>&2
else
    cp -p "$RUN_CONFIG"           "${OUTPUT_DIR}/"
    mv    "$JOB_SBAT_SCRIPT_TEMP" "${OUTPUT_DIR}/${JOB_SBAT_SCRIPT_NAME}"  # saving the temporary sbatch
    
    mkdir -p                   $ALLOC_DIR
    ln -s    ../${OUTPUT_DIR}  $ALLOC_DIR              # organize the output dirs according to their parameters
    rm -f       latest
    ln -s       "$ALLOC_DIR/${SLURM_JOB_ID}" latest
fi

###################################################
#      settings for distributed computation
###################################################

export MASTER_NODE="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
if [ "${DRY_RUN:-0}" = 1 ]; then
    export MASTER_ADDR=100.100.100.100  # ad hoc surrogate
    export MASTER_PORT=29500            # ad hoc surrogate
else
    export MASTER_ADDR=$(getent hosts "$MASTER_NODE" | awk '{print $1}' | head -n1)
    export MASTER_PORT="${MASTER_PORT:-$((29500 + SLURM_JOB_ID % 1000))}"
fi
echo ==================================================================  1>&2
echo " MASTER_NODE (first node)                  : $MASTER_NODE"         1>&2
echo " MASTER_ADDR (its ip address)              : $MASTER_ADDR"         1>&2
echo " MASTER_PORT (randomized jobid)            : $MASTER_PORT"         1>&2

CPU_BIND_OPS="--cpu-bind=cores"  # see the documentation why
GPU_BIND_FLAG=""                 # see the documentation why

export DISTR_OPS=$CPU_BIND_OPS $GPU_BIND_FLAG $UNBUFFERED 
echo ==================================================================  1>&2
echo " CPU_BIND_OPS  (--cpu-bind=...)            : $CPU_BIND_OPS"        1>&2
echo " GPU_BIND_FLAG                             : $GPU_BIND_FLAG"       1>&2
echo ==================================================================  1>&2

###################################################
#      call srun to get wet
###################################################

if [ "${DRY_RUN:-0}" = 1 ]; then
    echo "DRY_RUN=1: would run:" 1>&2
    echo "  srun $DISTR_OPS helper/bin/slurm/task-wrapper.sh ${RUN_SCRIPT} ${RUN_ARGS}" 1>&2
    echo "Dry run ends here (no srun executed)." 1>&2
else
    # Detect whether we are running under sbatch/salloc allocation
    if [ -z "${SLURM_JOB_ID-}" ]; then
	echo "Not running under sbatch (no SLURM_JOB_ID). Stopping just before srun" 1>&2
	exit 0
    else
	echo "Starting the srun of task-wrapper.sh at `date`" 1>&2
	echo "Now 'srun task-wrapper.sh' with RUN_SCRIPT=${RUN_SCRIPT} RUN_ARGS=${RUN_ARGS} OUTPUT_DIR=${OUTPUT_DIR}" 1>&2
	srun $HELPER/bin/slurm/task-wrapper.sh
    fi
fi
echo "Finishing the whole sbatch at `date`" 1>&2









##############################
# Documentation of the Details
##############################
#
# CLI arguments:
#   specify the --account,  e.g. with    sbatch-<partition> -A project_462000964 
#   specify the --job-name, e.g. with    sbatch-<partition> -J train-D-1n8g4h-nemo
#
# Design: - project independent   (-A specified via CLI)
#         - job-name independent  (-J specified via CLI)
#         - expects a wrapper 'sbatch-dev-g' (executable) or similar
#         - wrapper such as sbatch-dev-g is used to set the JOB_SBAT_SCRIPT variable
#         - wrapper checks that we are in a directory  ../<job-name>
#         - wrapper checks that there is a file called config.yaml in the current directory
#         - wrapper can "dry-run" the batch file to do sanity checking
#         - <job-name>-directory <=> job-name <=> config.yaml correspondence
#         - logs and data organizes under jobname
#
# File locations:
#         - this batch file can be anywhere; the current directory matters 
#         - config should be available at config.yaml
#         - log  files will be stored to  %j/logs
#         - save files will be created at %j/save (models,vocabs,transforms,embeddings)
#
#         - expects mammoth to be found in the directory 'mammoth-hf-lumi' (*)
#         - expects other scripts to be found in the directory 'bin'

###################################################
# Software architectured parameters - do not change 
###################################################

# #SBATCH --output=job-%j/out   
# #SBATCH  --error=job-%j/err   
#    Each job launch is encapsulated to its own subtree where
#    logs, models, vocabs, embeddings, and snapshot copies of sbatch, confs are stored.
# #SBATCH --partition=__MUST_USE_WRAPPER__  # overridden by sbatch-<partition> launcher
#    This is a special trick to stop 'sbatch trains.sbatch' running this script directly
#    Meant to be overriden with sbatch-<partition> not with --partition CLI argument.
# #SBATCH --ntasks-per-node=1         # constant: MAMMOTH spawns the GPUs and allocate them
#    Slurm should allocate only one task per node for MAMMOTH
#    => natural to use --gpus-per-node, --mem-per-gpu, and --cpus-per-gpu


###################################################
#     USER's FREE PARAMETERS FOR MAMMOTH          #
###################################################

# #SBATCH --time=0-04:00:00                         
# #SBATCH --nodes=1

# #SBATCH --gpus-per-node=8

# Meaning: "Give all GPUs of the node to this node’s launcher"
# This is also less confusing than --gpus-per-task, since
# gpus-per-task (1) * ntasks-per-node <= gpus-per-node.

# --gres=gpu:1 is fine too, but many sites prefer --gpus-per-node=1 now
# because it’s more explicit and aligns with newer GPU-aware options.
# Slurm documents both.

# #SBATCH --partition=__MUST_USE_WRAPPER__  # keep; use e.g. tools/helper/bin/sbatch-dev-g to auto-set 
# #SBATCH --mail-user=__MUST_USE_WRAPPER    # keep; use e.g. tools/helper/bin/sbatch-dev-g to auto-set

# If you are using the #SBATCH --mail-user=<your email address> option
# to override the default email address, ensure that the email set is
# valid.

# #SBATCH --account=__MUST_USE_WRAPPER__    # keep; use the command line option -A
# #SBATCH --job-name=__MUST_USE_WRAPPER__   # keep; use the command line option --job-name

# #SBATCH --mail-type=ALL                   # keep

###################################################
#        LUMI RECOMMENDATIONS - do not change     #
###################################################

# #SBATCH --mem-per-gpu=60G                 # recommended: 60G/GPU (with 4GiB slack for the system)
                                          # => --mem = nodes x gpus-per-node x mem-per-gpu
#    On LUMI-G the official recommendation is: Use at most 7 CPU cores
#    and 60 GB CPU memory per GPU.  Full node: 8 GPUs → 56 cores + 480
#    GB, leaving some slack for the system.
#
#    --mem-per-gpu=60G is huge on many clusters; Over-requesting
#    memory tends to increase queue times / reduce throughput for
#    everyone (many centers explicitly warn about requesting excess
#    memory).  *However*, if you allocate all GPUs anyway, no one else
#    (except the system) can use the CPU memory of the node.  60G is
#    actually often recommeded.
#
#
# #SBATCH --cpus-per-gpu=7                  # recommended: 7CPUs/GPU x GPUs
#
#    `man sbatch` + `man srun` @ LUMI: --cpus-per-gpu advises Slurm that
#    ensuing job steps will require ncpus processors per allocated GPU.
#    Not compatible with the --cpus-per-task option.
#
#    If the code is not multithreaded, CPUs may be wasted, but if it
#    is (e.g., dataloaders), it’s appropriate.
#
# #SBATCH --hint=nomultithread        # prefer 1 hw-thread per core; MAMMOTH is GPU-bound
#
#    We may want additionally control PyTorch / BLAS with:
#        export OMP_NUM_THREADS=1
#        export MKL_NUM_THREADS=1 (if MKL is present)
#        export TORCH_NUM_THREADS=1
#    That makes each training process effectively “single-threaded” on
#    CPU, minimizing thread noise, apart from tiny runtime threads.
#    These additions have not yet been full scrutinized for adoption,
#    but they do not seem to be dangerous with "1 hw-thread" anyway.

###################################################
#              MASTER SELECTION                   #
###################################################
# Use first hostname from nodelist, avoid plain `hostname` (not always in containers)
# Pick a port that collides less: 29500 + (JOBID mod 1000)
#
# DRY_RUN=1 when the script is run without really allocating resources
# 
# Mammoth expects --master_ip MASTER_IP  but PyTorch standard env name is
# MASTER_ADDR / MASTER_ADDRESS.  MASTER_PORT, however, is both PyTorch env
# and corresponds to Mammoth CLI option --master_addr.
# MASTER_NODE is just your own name for “master host”, it is not used.

###################################################
#              CPU CORE BINDINGS                  #
#      If your job shape changes, your mask_cpu   #
#      almost certainly needs to change too!      #
#                                                 #
#      Don’t bother with mask_cpu unless          #
#      you’re doing very specific tuning.         #
#                                                 #
#      --cpu-bind=cores is usually enough.        #
###################################################
# CPU_BIND_OPS="--cpu-bind=cores"
# Rely on HIP/ROCR_VISIBLE_DEVICES=$SLURM_LOCALID *inside* srun.  That
# is, just use --cpu-bind=cores and map tasks→GCDs via
# HIP/ROCR_VISIBLE_DEVICES=$SLURM_LOCALID inside srun.
# You do not get “7 cores per GPU” in a strict sense; you get “56
# cores total, all workers share them”. That’s fine for Mammoth and
# usually good enough.
#
# Optional: if you want the exact topology mapping from the LUMI slides, use
# the provided CPU mask string instead of --cpu-bind=cores. It’s a fine-tuning, not a requirement.
#   LUMI_CPU_MASKS=mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000
#   CPU_BIND_OPS="--cpu-bind=$LUMI_CPU_MASKS" # the slide’s exact LUMI masks
# Refer to slide: https://462000265.lumidata.eu/ai-20250204/files/LUMI-ai-20250204-09-Extreme_scale_AI.pdf
# The masks are per task: have 8 CPU masks per node only if you also have 8 tasks per node.
#
# Note that as the framework (MAMMOTH) spans its own tasks, each node
# has only one task but 56 available cores.  In this case, you should
# not use 8 masks (above) but one mask that ORs them:
#   CPU_BIND_OPS="--cpu-bind=mask_cpu:fe00fe00fe00fe00fe00fe00fe00fe"
# Repeating fe00 across the 128-bit mask gives you:
#   “7 CPUs on, then 9 CPUs off, 7 on, 9 off, …” across the CPU index space.
#
# --cpus-per-gpu=7 asks Slurm to allocate 7 logical CPUs per GPU (when
# nomultithread: 7 logical = 7 physical cores).  The mask_cpu:fe00...
# binding says exactly which CPUs (by index) each task is allowed to run on.
# Since tasks-per-node=1, this task can use any of 56 physical cores not
# reserved for the system.
#
# Mammoth doesn’t do its own core pinning;
# - It starts 1 “launcher” process per node (in your setup, --ntasks-per-node=1),
# - That process spawns per-GPU worker processes,
# - All of those processes inherit whatever cpuset Slurm gave to the srun task.
#
# Compare --cpu-bind=cores vs. --cpu-bind=mask_cpu:fe00fe00fe00fe00fe00fe00fe00fe:
#
# - For Mammoth’s logic itself, there is no behavioral difference between the two.
#   With --cpu-bind=cores, Slurm will choose which 56 it will give you,
#   according to its topology-aware placement (usually good NUMA locality,
#   compact placement),
#
# - Only switch to mask_cpu:... if you have evidence (profiling) that you need
#   tighter control over which cores are used.

###################################################
#       GPU-TASK BINDING AND VISIBILITY           #
#                                                 #
#           No gpu binding on LUMI                #
#           No VISIBILITY before srun             #
#                                                 #
#           You do not need these                 #
###################################################
# GPU_BIND_FLAG=""     
# --gpu-bind is an NVIDIA feature; on LUMI you want CPU binding, not GPU binding.
#
# Never export *VISIBLE_DEVICES in the batch script before srun.
# Outside srun there’s no SLURM_LOCALID, you’ll end up exposing all
# GPUs to every task VISIBLE_DEVICES, if exported, is exported in
# task-wrapper.sh

###################################################
#            srun OPTIONS                         #
#            Include --cpu-bind                   #
#            Use --unbuffered sparingly           #
#            Rely on the sbatch allocation        #
###################################################
# UNBUFFERED=
# For Slurm, srun --unbuffered (you already have this in DISTR_OPS)
# reduces output buffering between tasks and the collector. Handy for
# debugging, but it adds overhead—use sparingly on big jobs.

# Without allocation options, srun command creates a step that uses
# the entire allocation: all nodes, the 1 task per node, all 8 GPUs
# per node, all CPUs.  You only need to give extra srun resource
# options if you want a subset of the allocation (e.g. a diagnostics
# step on 1 node when you allocated 4).
#
# In particular, --gpus-per-task=1 is wrong for MAMMOTH since only 1
# GPU per node is used by the step, the other 7 GPUs on that node
# become “unused” and the MAMMOTH won’t see all 8 GPUs as intended.

###################################################
#       TO SRUN OR NOT TO SRUN (DRY_RUN)?         #
#   In dry-run, sbatch-dry imitates sbatch        #
#   without allocating any physical resources.    #
###################################################
