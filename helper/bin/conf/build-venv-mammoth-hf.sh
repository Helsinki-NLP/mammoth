#! /usr/bin/bash
# (c) 2025 Anssi Yli-Jyrä, CC-BY

# The script pulls updates in $GITHOME/mammoth
# The script checkouts feat/hf_integration as $GITHOME/mammoth-hf
# The script checkouts feat/helper         as $GITHOME/mammoth-helper
: "${GITHOME:?❌ GITHOME is not set}"           # $HOME/git
cd $GITHOME/mammoth
git pull   # get the latest updates
git worktree add ../mammoth-hf     feat/hf_integration
git worktree add ../mammoth-helper feat/helper

# This script assumes $PROJHOME
# The script creates symlink $PROJHOME/mammoth-hf
# The script deletes/rebuilds $PROJHOME/venv/mammoth-hf
: "${PROJHOME:?❌ PROJHOME is not set, suggestion: export PROJHOME=/project/$ACCOUNT/members/$USER}"
cd $PROJHOME
ln -s $GITHOME/mammot-hf            mammoth
ln -s $GITHOME/mammot-helper/helper .
ln -s helper/*                      .

# Load the partition/L software stack since we are running on login node
# If this script is run inside srun, partion/C or partition/G will be loaded
cd $PROJHOME
export NODE_KIND=gpu  # The partition/G software stack is forced even on login node
source helper/bin/slurm/module-loads.sh   
# After this, python is a wrapper that launches a pytorch/rocm container and runs python.

# Temporary hack: I do not want to use setup.py provided by feat/hf_integration
# but rather setup-mammoth-hf.py provided in helper/bin/conf.  Rename and copy:
cd $PROJHOME/mammoth
mv setup.py setup.py.old
cp $PROJHOME/helper/bin/conf/setup-mammoth-hf.py setup.py

# Build virtual environment for this particular branch (feat/hf_integration as mammoth-hf)
# We use `pip -e .` to install mammoth to the virtual environment via symlinks,
# i.e., in editable form.  This uses the branch-specific `setup.py` file 
cd $PROJHOME
rm -Rf venv/mammoth-hf
# Python/pip will run now in a modulerized-container with GPU-enabled pytorch and rocm libraries
# There is no need to do installation inside `srun` nor singularity, but partition/G is useful.
# We create virtual environment so that it inherits pytorch etc in the container
python -m venv --system-site-packages venv/mammoth-hf
source venv/mammoth-hf/bin/activate
python -m pip install --upgrade pip
cd $PROJHOME/mammoth
pip install -e . 
deactivate

# Expect something like this on LUMI:

# ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
#        This behaviour is the source of the following dependency conflicts.
# datasets 4.0.0 requires fsspec[http]<=2025.3.0,>=2023.1.0, but you have fsspec 2025.7.0 which is incompatible.
# lightning 2.5.1 requires packaging<25.0,>=20.0, but you have packaging 25.0 which is incompatible.
# vllm 0.10.1+rocm624 requires setuptools<80,>=77.0.3; python_version > "3.11", but you have setuptools 80.9.0 which is incompatible.
# vllm 0.10.1+rocm624 requires setuptools<80.0.0,>=77.0.3, but you have setuptools 80.9.0 which is incompatible.

# Successfully installed ConfigArgParse-1.7.1 MarkupSafe-3.0.2 certifi-2025.8.3 charset-normalizer-3.4.3 einx-0.3.0 frozendict-2.4.6 fsspec-2025.7.0 idna-3.10 loguru-0.7.3 networkx-3.5 packaging-25.0 protobuf-6.32.0 setuptools-80.9.0 sympy-1.14.0 tensorboard-2.20.0 transformers-4.55.4 typing_extensions-4.14.1 urllib3-2.5.0


