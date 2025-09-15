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

   | Functionality                                                   | Script Component    | Group |
   |-----------------------------------------------------------------|---------------------|-------|
   | Switching between machines, directories, and project accounts     | **`.profile`** | **conf** |
   | Creating the virtual environment with rocm-pytorch and packages      | **`build-venv.sh`**  | **conf** |
   | Downloading and managing shared data directories                | **`datasets.sh`**       | **gets** |
   | Quick creation of the local environment for SBATCH jobs         | **`mkjobdir.sh`**       | **create** |
   | Quick linking of the input files for SBATCH jobs                | **`mkinputs.sh`** (N/A) | **create** |
   | Quick creation of the MAMMOTH config files for SBATCH jobs      | **`mkconfig.sh`** (N/A) | **create** |
   | Quick creation of SBATCH scripts                                | **`mksbatch.sh`**       | **create** |
   | Variable, **user-editable setings** of SBATCH jobs              | **`sbatch-entry.sh`**   | **slurm** |
   | Generalizing and organizing the constant cross-machine know-how of SBATCH jobs | **`sbatch-tail.sh`**      | **slurm** |
   | - Integrity checks of the script combo, covering also your `sbatch-entry.slurm` | **`1-integrity-checks.sh`** | **slurm** |
   | - Setting up the remaining multiprocessor parameters              | **`2-distributed-setup.sh`** | **slurm** |
   | - Sanity checking of the SLURM parameters                         | **`3-sanity-checks.sh`** | **slurm** |
   | - Loading machine and partition specific modules                  | **`4-module-loads.sh`** | **slurm** |
   | - Adjusting important variables for  multi-processor communications | **`5-comms-setup.sh`** | **slurm** |
   | - A well-though wrapper for node-specific executions under `srun` | **`6-task-wrapper.sh`** | **slurm** |
   


