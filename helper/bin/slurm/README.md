# Slurm scripts for running Mammoth safely

This directory contains `sbatch-entry.sh` (template) and `sbatch-tail.sh` (constant).

## Usage:

   1) Install feat/helper branch of MAMMOTH following the instructions on
      [this page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/conf)

   2) Install common datasets by following the instructions on [this
page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/gets)

   3) Create a job directory under `$PROJHOME/data` using
`mkjobdir.sh`.  Follow the instructions on [this
page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/create)

   4) Create a sbatch file for your job directory using `mksbatch.sh`.
      Follow the instructions on [this
      page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/create)
      
   5) Complete your job directory by preparing config.yaml, and linking the datasets
      to the job directory.  (We are working on the related helper scripts.)
      
   6) Run the sbatch in the job directory
      ```
      sbatch   sbatch-entry.slurm
      ```
      
## The Added Value 

   | Functionality                                                   | Script Component    |
   |-----------------------------------------------------------------|---------------------|
   | Quick creation of the local environment for SBATCH jobs         | **`mkjobdir.sh`**       |
   | Quick linking of the input files for SBATCH jobs                | **`mkinputs.sh`** (N/A) |
   | Quick creation of the MAMMOTH config files for SBATCH jobs      | **`mkconfig.sh`** (N/A) |
   | Quick creation of SBATCH scripts                                | **`mksbatch.sh`**       |
   | Variable, **user-editable setings** of SBATCH jobs              | **`sbatch-entry.sh`**   |
   | Generalizing and organizing the constant cross-machine know-how of SBATCH jobs | **`sbatch-tail.sh`**      |
   | - Integrity checks of the script combo, covering also your `sbatch-entry.slurm` | **`1-integrity-checks.sh`** |
   | - Setting up the remaining multiprocessor parameters              | **`2-distributed-setup.sh`** |
   | - Sanity checking of the SLURM parameters                         | **`3-sanity-checks.sh`** |
   | - Loading machine and partition specific modules                  | **`4-module-loads.sh`** |
   | - Launching the virtual environment from `venv/bin/activate`      |                     |
   | - Adjusting important variables for  multi-processor communications | **`5-comms-setup.sh`** |
   | - A well-though wrapper for node-specific executions under `srun` | **`6-task-wrapper.sh`** |
   | Switching between machines, directories, and project accounts     | **.profile** |
   


