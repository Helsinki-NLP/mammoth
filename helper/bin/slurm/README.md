# Slurm scripts for running Mammoth safely

This directory contains `sbatch-entry.sh` (template) and `sbatch-tail.sh` (constant).

## Usage:

   1) Install feat/helper branch of MAMMOTH following the instructions on
      [this page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/conf)

   2) Install common datasets by following the instructions on [this
page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/gets)

   3) Create a job directory under `$PROJHOME/data` using `mkjob.sh`:
      ```
      export JOB_NAME=train-L-1n1g10m-en:es-test
      sh $PROJHOME/base/mammoth-helper/helper/bin/create/mkjob.sh $JOB_NAME
      ```
      For more information follow the instructions on [this
page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/creat)

   4) Create a sbatch file for your job directory using ``mksbatch.sh`
      ```
      sh $PROJHOME/base/mammoth-helper/helper/bin/create/mksbatch.sh
      cd       $PROJHOME/data/$JOB_NAME
      mv       sbatch-entry.slurm.new sbatch-entry.slurm
      ```
      
   5) Complete your job directory by preparing config.yaml, and linking the datasets
      to the job directory.  (We are working on the related helper scripts.)
      
   6) Run the sbatch in the job directory
      ```
      sbatch   sbatch-entry.slurm
      ```
      
## The automation you get for free with `sbatch-tail.sh` 

   | Functionality                                                   | Script Component    |
   |-----------------------------------------------------------------|---------------------|
   | A well-though wrapper for node-specific executions under `srun` | `6-task-wrapper.sh` |
   | Setup of important machine specific environment variables for  multi-processor communications | 
     `5-comms-setup.sh` |
   | Launching the virtual environment from `venv/bin/activate`      |                     |
   | Loading machine and partition specific modules                  | `4-module-loads.sh` |
   | Sanity checking of the SLURM parameters                         | `3-sanity-checks.sh`|
   | Setting up the remaining multiprocessor parameters              | `2-distributed-setup.sh` |
   | Integrity checks of the script combo, covering also your `sbatch-entry.slurm` | `1-integrity-checks.sh` |

## Metadata of the Helper Features

Authors: Anssi Yli-Jyrä (c) 2025
License: CC-NC-BY

