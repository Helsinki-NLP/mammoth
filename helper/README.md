# Helper functionalities for shared installation

Helper Functionalities are a set of scripts and files located under tools/helper:

### Readily Build Software Stack for MAMMOTH

The following resources are needed to build the tailored software stack:
```
tools/helper/install/build-venv.sh         - a tool for building the software stack 
tools/helper/install/requirements_lumi.txt - copy of csc_env/lumi/requirements_lumi.txt
tools/helper/modules                       - contains a component of the software stack
tools/helper/wrappers                      - contains components of the software stack
tools/helper/venv                          - location for the virtual environment
```
The tailred software stack can be activated with the command:
```
source tools/helper/install/module-loads.sh     - activator for the software stack
```
This command loads/activates/sets
- the required standard CSC modules
- the most recent CSC-built singularity image with pytorch/ROCm support
- the virtual environment covering `requirements_lumi`
- python/pip commands that automatically start the singularity image
- library paths and LUMI environment variables supporting efficient comms
- AI bindings and RCCL related system lib bindings

The software stack activator command is included in the following.

### Readily Build Task Wrapper Infra for Distributed MAMMOTH Running

The following scripts are designed to help running a task in a Slurm node:
```
tools/helper/bin/slurm/{task-wrapper,smi-monitor}.sh
```
These scripts take care of:
- activation of the appropriate software stack in GPU nodes
- passing node_rank, master_address, master_port to MAMMOTH
- setting MIOPEN caches for ROCm libraries that Mammoth uses via PyTorch
- setting comms debugging variables (when COMMS_DEBUG=1)
- starting and stopping SMI monitoring logs
- exceuting a Python script (train.py etc)

The infrastructure is activated under Slurm nodes by command assuming
the Python script (RUN_SCRIPT) and the job arguments (RUN_ARGS) in the
environment.
```
srun tools/helper/bin/slurm/task-wrapper.sh
```

The infrastructure activation is included in the following.

### Readily Build SBATCH File Example

There is an example SBATCH file for running a MAMMOTH training job on LUMI.
This assumes an existing MAMMOTH installation with the tools/helper tree.
```
tools/helper/examples/train-4h1n8g.sbatch  - An example sbatch file
```
This batch file is easy to configure:
- RUN_SCRIPT points to mammoth/train.py inside the installed MAMMOTH
- RUN_CONFIG points to a config.yaml file in the current working directory .
- Mammoth assumes tasks-per-node=1, spawing the GPUS via one task per node.
- User parameterizes **only**:  --nodes, --gpus-per-task, and --time.
- The script sets the cpu-binding appropriately (core)
- The outputs of this batch go to the subdirectory ./id-$SLURM_JOB_ID
- The script activates the task wrapper infra automatically
- Account (-A), job-name (-J) and partition (-p) must be specified via CLI options

The following is a handy way to call these SBATCH files.

### SBATCH Command Wrappers

The following tools are handy for validating and running the provided
SBATCH file examples.
```
tools/helper/bin/sbatch-dry-run
tools/helper/bin/sbatch-dev-g
tools/helper/bin/sbatch-small-g
```

The first allows test running the example-resembing SBATCH files
without queueing for resources.  It mimics the behaviour of `sbatch`
and runs the SBATCH script immediately.  Srun is not entered.  This
allows to quicly validate the script for errors that do not require
running a GPU.

The other tools (sbatch-dev-g and sbatch-small-g) ultimately execute
sbatch on the SBATCH script, but before doing it, it runs two quick
tests on the script.

1. sbatch --test-only to test the syntax of sbatch directives 
2. sbatch-dry-run to dry-run the SBATCH file without Slurm allocations.

### EXAMPLE CONFIGURATION FILES FOR MAMMOTH

There is also an example yaml file that could allow you to test
`tools/helper/examples/train-4h1n8g.sbatch`.  To be fully functional
without modifications, you need to access to the specified
vocabularies and translation examples.  The yaml file path is:
```
tools/helper/examples/config.yaml
```

### Usage in Short

1. You need to belong to a project (-A project_XXXXX) that has the
   above readily available.  Assume that the mammoth repository
   containing tool/helper is installed to directory $MAMMOTH
   
2. Set up a directory whose file name will be your job bame (-J
   job_name). Change to this directory.

3. Copy the files $MAMMOTH/tools/helper/examples/config.yaml
   and tools/helper/examples/train-4h1n8g.sbatch there.

4. You need access rights to the files specified in the config file
   config.yaml
   
5. Sbatch the job with the wrappers (note the script name first):

   ```
   $MAMMOTH/tools/helper/bin/sbatch-small-g
   $MAMMOTH/tools/helper/examples/train-4h1n8g.sbatch
     -A project_XXXXX \
     -J your_job_name   
   ```

6. Expect logs and outputs appear in the directory `./latest/`.
   Wait for 4 hours = the time limit.

