# Choosing the right Python version and pytorch container for MAMMOTH
When we start using LUMI, we first want to know does it run MAMMOTH and what modules, containers or pip installs are needed.  I have first experimented with pytorch and other modules, but due to inter-process communication and efficiency, I was adviced by the LUMI super-users to switch to the use of certain python-pytorch containers that provide the support for LUMI's Slingshot 11 architecture.  

## What Was Available:
- pytorch module (involving a .sif container) - old versions up to ...
- new .sif-images in `/appl/local/containers/sif-images`, specifically the images:
  - `lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif`
    
    (With command `singularity inspect /appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif` we find out that this is build from a Docker source image: `org.label-schema.usage.singularity.deffile.from: localhost:5000/lumi/lumi-pytorch:rocm-6.2.0-python-3.10-pytorch-v2.3.0`)
    
  - `lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.0.sif`

## Key steps:
- I created a personal view of the project at `$PROJHOME` in the path:
  `/project/$ACCOUNT/members/$USER/`
- I created a directory for my local backup copy of the relevant sifs at:
  `/project/$ACCOUNT/members/$USER/images`
- I tested to activate the container with the commands:
```bash
# export SIF=/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif
export SIF=$PROJHOME/images/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif  # local copy
singularity exec $SIF bash # start a container
```

*later addition*: Since the images are large I did not want to copy them.  Instead I used symlinks.
This is against the advice given but I am not afraid of losing these images.  A similar one will probably
be offered by CSC anyway if the image is removed for some deprecation reasons.

- Use a LUMI trict to set the path to include Python binaries inside the container
```bash
$WITH_CONDA    # without this, we do not have access to python, pip, pytorch library
```
- After dozens of attempts and failed improvements with the setup, I found out that
  the package `pyonmttok` required by MAMMOTH is neither readily available or feasibly
  compilable and installable for Python 3.12 yet.
  There exists a recipe to compile the package from scratch but my attempts failed due to
  the discrepancy and incompatibily between the compiled package and the C/C++ compilers
  used to compile Python 3.12.  Thus, although compilation was possible, installation failed.
  There were also other deprecations that affected this failure.  This led to the conclusion
  that we need to stick to the older container `lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif`
  as its Python version is 3.10 for which `pyonmttok` is readily available.
- To test furter compatibility of MAMMOTH and this container, I did some successful attempts
  and determined that I would use the path `$PROJHOME/venv/mammoth3.10-with-conda/` to store
  the Python virtual environment built on the top of the container and its $WITH_CONDA paths.
- In this virtual environment, I installed packages that were as close to the current requirements
  in setup.py in mammoth.  After this, command `deactive` closes the environment.
- Finally, we exit the container with `exit`.
