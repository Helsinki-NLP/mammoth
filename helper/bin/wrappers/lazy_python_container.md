## Turning the Container into a Module

When any sif-based module is loaded (CSC made as well as your self-made one), the singularity is not executed during the load command.  Instead, the entering to the singularity container is lazy and happens after `srun` command, separately in each task.  To implement this, the `module load pytorch` in `/appl/local/csc/modulefiles/pytorch/2.7.lua` changes the environment in such a way that a special python wrapper `python` and other similar wrappers for related commands -- to be found in the directory `/appl/local/csc/soft/ai/bin` -- are included to the path.  

### The Trick of Delayed Container

The wrapper for `python` (and `pip` etc.) is in `/appl/local/csc/soft/ai/bin` and occurs first on the path.  The wrapper will ensure that the container is running, and then executes the `python` command with the passed arguments and virtual environment.  This tecnique works as follows:
1. The `module load pytorch` prepares the trick of *delayed container* by doing two things:
   - Sets the value of `SING_IMAGE` to point to a `.sif` file, but does not launch the container.
   - Prepends `/appl/local/csc/soft/ai/bin` to the path; this location contains the python wrapper.
2. When python is called, this wrapper runs do the trick:
   - Verifies that the wrapper is used only in context that module load has prepared. This is teste by checking tha `$SING_IMAGE` is defined.
   - Assumes that the container `$SING_IMAGE` will have the container's own Python command family (`python`, `pip` etc.) in the path. 
   - Bind `/appl/local/csc/soft/ai/bin` and some other locations (`/appl/lumi` and optionally `/var/lib/project_info`) to container.
   - Converts the directories in `VIRTUAL_ENV` to items in the `PYTHONPATH`.  
3. Starts the container and runs the wrapper command inside the singularity, using its own path settings to find the real python.

This wrapper for `python` is actually directly re-usable also in user-defined `pytorch` modules.  It is efficient if you only run the python program a few times, but if your task launches python dozens of times, then the overhead of starting Singularity separately becomes inefficient.

### The User-Made Module for Container

I have now dubbed this for `lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif` as follows.

1. I added symlink `cd $PROJHOME; ln -s sif images` to use a similar directory name as the `pytorch` module.

3. I have made a project copy of `/appl/local/csc/soft/ai/bin/python` to `$PROJHOME/bin` since it turned out that the container of choice does not offer python on the path before command `$WITH_CONDA` is given.  So, I had to change the following:
   - the wrapper will apply the `$WITH_CONDA` command before trying to find `python` in the container.

I also experimented with automatising the virtual environment, but it turned out that this is both unnecessary and possibly incoherent approach.  For example, I failed to launch mammoth-training when this changes was in place.  The mammoth command will surely need its own virtual environment launched, but it seems to a wrong approach to mingle with vens in our own ways.  Rather, it seems that the wrapper handles the virtual environment by itself.
Perhaps we can do `source $PROJHOME/venv/mammoth3.10-with-site/bin/activate` after the `module load` and the the delayed container trick will silently pass the environment inside the container.

2. Project's modules are stored in `$PROJHOME/modules` directory that I created.  There, I have made my own module called `pytorch-rocm-mammoth/1.0` by adding the file `$PROJHOME/modules/pytorch-rocm-mammoth/1.0.lua`.  This file is a modified copy of the file `/appl/local/csc/modulefiles/pytorch/2.7.lua`.

```bash
> local singName = 'lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif'
> local pytorchVersion = '2.3.0'
> local loadTxt = capture('cat <SOMETHING>/txt/load-pytorch-rocm-mammoth.txt')
> ROCm-enabled PyTorch version %s for Python and MAMMOTH venv
> local signRoot = os.getenv('PROJHOME')
> prepend_path('PATH', '/project/<SOMETHING>/bin')
> -- ROCm/PyTorch environment hook (so users can `eval $WITH_CONDA`)
> setenv("WITH_CONDA", "source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch")
```

4. I copied `/appl/local/csc/soft/ai/txt/load-pytorch.txt` to a new directory `$PROJHOME/txt` as file `load-pytorch-rocm-mammoth.txt` and (should have) changed it.

5. One can now activate the module with commands:
```bash
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings                # AI bindings will be needed
module use $PROJHOME/modules
module load pytorch-rocm-mammoth
source $PROJHOME/venv/mammoth3.10-with-site/bin/activate
```
4. Run python ...  or srun python ... (these are using the wrapper on the path to carry out on-the-fly container launches).

Note that this procedure references:
- the location of the sif file in `$PROJHOME/sif`
- the locstion of the module in `$PROJHOME/modules`
- the location of the wrapper in `$PROJHOME/bin`
- the location of the message in `$PROJHOME/txt`
- variable `SING_IMAGE` to tell which container has been delayed
- container internal directory `/.singularity.d/` indicating it was launched
- `$WITH_COMMAND` a variable containing commands to add container-internal python to the path.

### Similar Thing: Tykky
[Tykky](https://github.com/CSCfi/hpc-container-wrapper) is a set of tools which wrap installations inside an Apptainer/Singularity container to improve startup times, reduce IO load, and lessen the number of files on large parallel filesystems.   This is a tool to create installations using existing containers. The basic idea is to install software through a container, convert this into a filesystem image and mount this filesystem image when running the container.
