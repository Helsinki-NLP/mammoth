# Slurm scripts for running Mammoth safely

This directory contains a componentwise split of an sbatch script for running Mammoth NLP safely.
The main script is `sbatch-tail.sh` that is designed to be sourced right after your #SBATCH directives.
For example:
```
#!/usr/bin/env bash
#SBATCH --job-name=2025-08-21_enfi_1n1g-30m
#SBATCH --account=project_462000964
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7              # ~1/8 of node
##SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:mi250:1
#SBATCH --time=00:30:00
#SBATCH --mem-per-gpu=60G              # responsible use, room for OS
##SBATCH --mem=60G                     # never combine --mem with --mem-per-*
##SBATCH --mem-per-cpu=8G              # 7 CPUs → 56G for the task
##SBATCH --mem=0                       # (or omit): take all host RAM on the node
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
echo ======================================
echo "Reserved  ┌──────────┐"
echo "          │ Task 0   │ GPU 0 │ CPU 0–6"
echo "          └──────────┘"
echo ======================================
source "$PROJHOME/bin/slurm/sbatch-tail.sh"
```

The `sbatch-tail.sh` does the following things:

1. It checks that all the necessary environment variables are set.
2. It reports these variables as it goes to enable debugging.
3. It checks that all the scripts are present.
4. It has a sequence of script includes:
   1. distributed-setup.sh -- calculate or infer some job variables used in srun and tensorboard
   2. exp/$SLURM_JOB_NAME/job-setup.sh -- set up variables that describe the experiment
   3. sanity-checks.sh -- verify that allocations are responsibe and within the experiment guards
   4. module-loads.sh -- load pytorch and related modules
   5. activate -- activate the virtual environment
   6. comms-setup.sh -- setop environment variables for multiprocessor communications
   7. task-wrapper.sh -- a wrapper for local node setup, monitoring and execution of train script
5. `task-wrapper` does the following steps:
   1. it calls local-setup.sh that takes care of the pattern (python/torchrun), CPU/GPU bindings and monitoring
   2. takes care of python and torchrun arguments
   3. summarizes the variables related to command setup
   4. executes the pattern and the train script with arguments
6. These scripts have been developed so that they should run correctly on LUMI, Puhti and Mahti.
   However, the scripts have not yet been tested on Puhti and Mahti.

There is EXPERIMENTS.md in the parent directory telling how do you set up your experiment.
In relation to this directory, you will need

- a copy of `job-setup_template.sh` as `job-setup.sh`.

- a modified and correct copy of a train-sbatch-<machine>....slurm bash script (in `moreslurms` directory; not yet in GitHub) - no 100% correct templates are available yet.

- a modified copy of a working config.yml.envsubst file (no temptlates are available yet)


