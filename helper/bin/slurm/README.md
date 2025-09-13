# Slurm scripts for running Mammoth safely

This directory contains `sbatch-entry.sh` (template) and `sbatch-tail.sh` (constant).
Usage:

   1) Ensure you have the *places* for the helper and the experiments
      ```
      export PROJHOME=your-readily-built-base-dir    # base dir
      export PROJDATA=your-readily-built-work-dir    # your place to store data and models
      export GITHOME=your-readily-built-base-dir/git # your git root 
      cd       $PROJHOME
      ln -s    $PROJDATA          data               # link data to projhome
      mkdir    $PROJDATA/base                        # for partial inhertance to avoid cycles
      cd       $PROJDATA/base
      ln -s    ../venv            .
      ln -s    $GITHOME           git                # or ln -s ../git .  if that is the $GITHOME
      ln -s    git/mammoth-hf     mammoth            # tentative branch
      ln -s    git/mammoth-helper .                  # until we join the branches
      ```
   2) give a *name* to your project and create the job directory, with link to $PROJHOME
      ```
      export JOB_NAME=your-job-name
      mkdir -p $PROJHOME/data/$JOB_NAME
      cd       $PROJHOME/data/$JOB_NAME
      ln    -s $PROJHOME/base .                      # inherits all except itself
      ```
   3) Create and edit your own `sbatch-entry.sh`
      cd       $PROJHOME/data/$JOB_NAME
      cp       base/mammoth-helper/helper/bin/slurm/sbatch-entry.sh .
      emacs    sbatch-entry.sh 
      
   4) Add other subdirectories to your job and link files there
      cd       $PROJHOME/data/$JOB_NAME
      mkdir    logs models tensorboard
      
   5) Run the sbatch in the directory called .../$JOB_NAME
      cd       $PROJHOME/data/$JOB_NAME
      sbatch -J "$JOB_NAME" -A "$ACCOUNT" -o logs/%x-%j.out -e logs/%x-%j.err sbatch-entry.sh

The contents of `sbatch-entry.sh`

   #SBATCH directives
   ...
   
   export JOB_PATTERN=slurm|torchrun
   export JOB_SCRIPT=your-python-script-path-in-the-helper-tree
   export JOB_ARGS=your-job-arguments-to-the-script
   export GUARD_MAX_NODES=4
   export GUARD_TIME="${GUARD_TIME:-0-01:00:00}"
   source base/mammoth-helper/helper/bin/slurm/sbatch-tail.sh

The automation provided by `sbatch-tail.sh` (from back to the front):

   - a well-though wrapper for node-specific executions under `srun`
     (`base/mammoth-helper/helper/bin/slurm/6-task-wrapper.sh`)
   
   - setup of important machine specific environment variables for
     multi-processor communications
     (`base/mammoth-helper/helper/bin/slurm/5-comms-setup.sh`)
   
   - launching the virtual environment from `venv/bin/activate`
   
   - loading machine and partition specific modules
     (`base/mammoth-helper/helper/bin/slurm/4-module-loads.sh`)

   - sanity checking of the SLURM parameters
     (`base/mammoth-helper/helper/bin/slurm/3-sanity-checks.sh`)

   - setting up the remaining multiprocessor parameters
     (`base/mammoth-helper/helper/bin/slurm/2-distributed-setup.sh`)
   
   - ingredity checks the scripts
     (`base/mammoth-helper/helper/bin/slurm/1-integrity-checks.sh`)

Authors
   Anssi Yli-Jyrä (c) 2025

Licence 
   CC-NC-BY
