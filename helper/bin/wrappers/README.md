# Why This Directory?

## When this is used

This directory contains wrappers for using site module called
`pytorch-rocm-mammoth`.  It is a module to use singularity container
lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif without tears.

- You should load module (`../modules/python-rocm-mammoth`) first.
  The module adds this directory to the path.

- The current directory contains singularity wrappers for python, pip,
  python3, pip3, bash (`sing-bash`).

## Added value

The wrappers ensure the container is running, bind a number of
locations, and set RCCL-related environment variables.  This is is to
guarantee the best possible experience.  You can simulate this with
`singularity exec python ..` but then you need to set up several
environment variables, paths and container bindings.

## Correct Use

When building the Python virtual environment for this container, you
must first load `pytorch-rocm-mammoth`.  As you now call the wrapper
without having the readily built virtual environment in `../../venv`,
you will get a warning.  This is normal if you are building the venv.

It is WRONG to build virtual environment (venv) before loading the
module, since in that case the venv will not link these wrappers, and
the wrappers will not start the virtual environment.  If you rebuild,
the correct order is: delete the venv + load the module + build the
venv.

## How it works?

If the venv has been created after loading the module
`pytorch-rocm-mammoth`, everything works no matter which one (venv or
module) is activated first.  The python-wrappers of the current
directory lift an active python venv or the default environment
`../../venv` to PYTHONPATH.  In the second case, the environment is
temporarily activated.



