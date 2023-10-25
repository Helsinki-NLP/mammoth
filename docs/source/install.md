
# Installation

Installing from source usually works. Here, we provide two examples that install MAMMOTH in Nvidia and AMD platforms. 

## Puhti / Mahti 

This first example install MAMMOTH on clusters ([Puhti/Mahti](https://docs.csc.fi/computing/)) with Nvidia GPUs.

### Install
In the login node, create a directory in the `projappl` linked to our project to host the shared python dependencies, and install the code base & dependencies there:

```
# where to install the necessary python packages
PROJECT=<your_project_name_is_your_account>
ENV_DIR="/projappl/${PROJECT}/test"
# where the codebase was copied to
CODE_DIR="/scratch/${PROJECT}/path/to/mammoth"

# set up variables & modules
module load pytorch
mkdir -p $ENV_DIR
export PYTHONUSERBASE=$ENV_DIR

#install dependencies
cd $CODE_DIR
pip3 install -e . --user

# optionally, to make sure that other people can access this install:
chmod -R 777 $ENV_DIR
chmod -R 777 $CODE_DIR
```

### Run
In slurm job scripts, update environment variables to get python to run your code properly:

```
ENV_DIR="/projappl/${PROJECT}/test"
CODE_DIR="/scratch/${PROJECT}/path/to/mammoth"

module load pytorch
export PYTHONUSERBASE=$ENV_DIR
# note: this overwrites the path, you can also try appending this subdirectory instead
export PYTHONPATH=$ENV_DIR/lib/python3.9/site-packages/

srun python3 -u $CODE_DIR/train.py ...
```


## LUMI

The GPU partition of [LUMI supercomputer](https://lumi-supercomputer.github.io) is AMD-based. 
This example uses python virtual environment without container. But as recommended by the LUST team of LUMI, better to use container. An instruction to install MAMMOTH under Singularity container will release in the future.
<!-- TODO -->

### Install 
1. Start an interactive session `srun --account="$PROJECT" --partition=dev-g --ntasks=1 --gres=gpu:mi250:1 --time=2:00:00 --mem=25G --pty bash`
2. Load modules: 
    
    ```bash
    module load cray-python
    module load LUMI/22.08 partition/G rocm/5.2.3
    
    module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
    module load aws-ofi-rccl/rocm-5.2.3
    ```
3. Create virtual environment `python -m venv your_vevn_name` and activate it `source your_venv_name/bin/activate`. 

4. Install pytorch: `python -m pip install --upgrade torch==1.13.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2`

5. Install mammoth 
    
    ```bash
    cd /pfs/lustrep1/projappl/${PROJECT}/${USER}/mammoth
    pip3 install -e .
    pip3 install sentencepiece==0.1.97 sacrebleu==2.3.1
    ```

### Run
You can train in slurm job scripts, such as
```bash
srun python -u $CODE_DIR/train.py ...
```
For more details, we refer to the tutorial section. 