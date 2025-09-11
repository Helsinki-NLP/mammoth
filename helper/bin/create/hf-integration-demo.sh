# hf-integration-demo.sh
#   create an experiment directory and run mammoth-hf there
#   used to demonstrate HF integration
#   based on 

# requirements
: "${SYSTEM:?❌ SYSTEM is not set}"       # e.g., lumi
: "${ACCOUNT:?❌ ACCOUNT is not set}"     # e.g., project_462000964
: "${PROJDATA:?❌ PROJDATA is not set}"   # e.g., /scratch/project_462000964/members/aylijyra
: "${PROJHOME:?❌ PROJHOME is not set}"   # e.g., /project/project_462000964/members/aylijyra
# places
d="$PROJDATA/exp";  [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
d="$PROJHOME/bin";  [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
f="$PROJHOME/bin/create/hf-integration-demo.sh"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
d="$PROJHOME/mammoth-hf";  [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }
# If you do not have $PROJHOME/mammoth-hf, you can do it as follows
# 1. Go to your git directory ($GITHOME)
# 2. Clone mammoth
# 3. cd mammoth
# 4. git worktree add ../mammoth-hf feat/hf_integration
# 5. Go to $PROJHOME
# 6. ln -s $GITHOME/mammoth-hf
# 7. pip install -r $PROJHOME/bin/conf/requirements-lumi.txt
# but this path is unoptimal as it installs too much and to your home directory
# and it does not create venv, required in the following
#
# Instead:
# 1. sh $PROJHOME/bin/conf/build-venv-mammoth-hf.sh
# This installs mammoth, feat/hf_integration, and builds a venv with the requirements
# This is recommended approach.
#
d="$PROJHOME/venv/mammoth-hf/bin";  [[ -d "$d" && -r "$d" && -x "$d" ]] || { echo "❌ Bad dir: $d" >&2; exit 1; }

# set variables
TODAY=$(date +"%Y-%m-%d")
EXP_ID=${TODAY}_enes_ft_1g30m
THIS=$PROJDATA/exp/$EXP_ID
mkdir -p $THIS
cd $THIS
mkdir -p save/converted_model  # add output directory
mkdir logs

# link some stuff to this directory
ln -s $PROJDATA/exp
ln -s $PROJHOME/bin
ln -s $PROJHOME/mammoth-hf mammoth
ln -s $PROJHOME/venv/mammoth-hf venv
ln -s bin/create/hf-integration-demo.sh create-this.sh
ln -s bin/templates

# add a local slurm file whose tail is automated via the existing $PROJHOME/bin/sbatch-tail.sh
echo '#!/usr/bin/env bash
# train-sbatch-lumi-1n1g-10m.slurm
#SBATCH --job-name=$EXP_ID
#SBATCH --account=$ACCOUNT
#SBATCH --partition=dev-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7              # ~1/8 of node
#SBATCH --gpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mem=60G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# ┌──────────┐
# │ Task 0   │ GPU 0 │ CPU 0–6
# └──────────┘
source "$PROJHOME/bin/sbatch-tail.sh"
' | envsubst > enter.slurm

# Add a local job-setup.sh file (called from $PROJHOME/bin/sbatch-tail.sh)
# This file provides the following:
#   RUN_SCRIPT
#   RUN_ARGS[@]
#   GUARD_MAX_NODES
#   GUARD_TIME
#   PATTERN
#   MASTER_ARGS
# See $PROJHOME/templates/job-setup
echo


MAMMOTH=$PROJDATA/mammoth  # local
f="$MAMMOTH/bin/activate"; [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; exit 1; }
source $MAMMOTH/bin/activate                       # enter the environment

# Basic conversion from Hugging Face model hub (language pair needs to be specified)
#
# You can do this on the login node (not recommended) as follows:
#  1. bin/slurm/module-loads.sh         # set up the software stack
#  2. bin/source venv/bin/activate      # enter the virtual environment
#  3. python mammoth/hf2mammoth.py vgaraujov/bart-base-translation-en-es ./save/converted_model --src-lang en --tgt-lang es
#
# Or, you can run the slurm script in this directory:
#  1. sbatch hf2mammoth.slurm


