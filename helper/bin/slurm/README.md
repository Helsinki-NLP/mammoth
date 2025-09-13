# Slurm scripts for running Mammoth safely

This directory contains `sbatch-entry.sh` (template) and `sbatch-tail.sh` (constant).

## Usage:

   1) Follow instructions in
      [](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/slurm)
      to install this branch of MAMMOTH and the create the environment.

   2) give a *name* to your project(directory), and make it inhert the `base`
      ```
      export JOB_NAME=your-job-name
      mkdir -p $PROJHOME/data/$JOB_NAME
      ln    -s $PROJHOME/base $PROJHOME/data/$JOB_NAME  # inherits all except itself
      ```
   3) Add `sbatch-entry.sh` and other subdirectories
      ```
      cd       $PROJHOME/data/$JOB_NAME
      mkdir    logs models tensorboard
      cp       base/mammoth-helper/helper/bin/slurm/sbatch-entry.slurm .
      ```
   4) Complete your job directory by preparing config.yaml, sbatch-entry.slurm, and data
      
   5) Run the sbatch in the job directory
      ```
      cd       $PROJHOME/data/$JOB_NAME
      sbatch -J "$JOB_NAME" -A "$ACCOUNT" -o logs/%x-%j.out -e logs/%x-%j.err sbatch-entry.slurm
      ```
## The contents of `sbatch-entry.slum`
   ```
   #SBATCH ...
   #SBATCH ...
   export JOB_PATTERN=slurm|torchrun
   export JOB_SCRIPT=your-python-script-path-in-the-helper-tree
   export JOB_ARGS=your-job-arguments-to-the-script
   export GUARD_MAX_NODES=4
   export GUARD_TIME="${GUARD_TIME:-0-01:00:00}"
   source base/mammoth-helper/helper/bin/slurm/sbatch-tail.sh
   ```
   
## The automation you get from `sbatch-tail.sh` 

   - A well-though wrapper for node-specific executions under `srun`
     (`base/mammoth-helper/helper/bin/slurm/6-task-wrapper.sh`)
   
   - Setup of important machine specific environment variables for
     multi-processor communications
     (`base/mammoth-helper/helper/bin/slurm/5-comms-setup.sh`)
   
   - Launching the virtual environment from `venv/bin/activate`
   
   - Loading machine and partition specific modules
     (`base/mammoth-helper/helper/bin/slurm/4-module-loads.sh`)

   - Sanity checking of the SLURM parameters
     (`base/mammoth-helper/helper/bin/slurm/3-sanity-checks.sh`)

   - Setting up the remaining multiprocessor parameters
     (`base/mammoth-helper/helper/bin/slurm/2-distributed-setup.sh`)
   
   - Ingredity checks the scripts
     (`base/mammoth-helper/helper/bin/slurm/1-integrity-checks.sh`)

## Authors:
   Anssi Yli-Jyrä (c) 2025

##  Licence 
   CC-NC-BY
