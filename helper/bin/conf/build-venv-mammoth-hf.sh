#! /usr/bin/bash
# (c) 2025 Anssi Yli-Jyrä, CC-BY

set -euo pipefail

log() { printf '%s %s\n' "[build-venv-mammoth-hf]" "$*"; }

# --- Required env ----------------------------------------------------------------
# This script assumes $GITHOME
# This script assumes $PROJHOME
: "${PROJHOME:?❌ PROJHOME is not set (e.g., export PROJHOME=/project/$ACCOUNT/members/$USER)}" || exit 1
: "${GITHOME:?❌ GITHOME is not set (e.g., export GITHOME=$HOME/git, or GITHOME=$PROJHOME/git)}" || exit 1

check_under() {
  local base="$1"; shift
  local name f missing=0
  for name in "$@"; do
    f="$base/$name"
    [[ -s "$f" ]] || { echo "❌ Missing/empty: $f" >&2; missing=1; }
  done
  (( missing == 0 )) || exit 1
}

MAMMOTH_REPO="$GITHOME/mammoth"
HF_WT_DIR="$GITHOME/mammoth-hf"
HELPER_WT_DIR="$GITHOME/mammoth-helper"
HF_BRANCH="feat/hf_integration"
HELPER_BRANCH="feat/helper"


# --- Helpers ---------------------------------------------------------------------
ensure_branch_exists () {
  local repo_dir="$1" branch="$2"
  git -C "$repo_dir" fetch --all --prune
  if ! git -C "$repo_dir" show-ref --verify --quiet "refs/heads/$branch" &&
     ! git -C "$repo_dir" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    echo "❌ Branch '$branch' not found in $repo_dir (nor origin/$branch)"; exit 1
  fi
}

# Ensure a worktree at target_dir for the given branch.
ensure_worktree () {
  local repo_dir="$1" branch="$2" target_dir="$3"

  # Normalize target path
  mkdir -p "$(dirname "$target_dir")"

  # Prune stale records so add doesn't complain
  git -C "$repo_dir" worktree prune

  if [ -d "$target_dir/.git" ] || [ -f "$target_dir/.git" ]; then
    # Already a git worktree/checkout; check what branch it has
    current_branch="$(git -C "$target_dir" rev-parse --abbrev-ref HEAD || echo 'DETACHED')"
    if [ "$current_branch" = "$branch" ]; then
      log "Worktree already on $branch at $target_dir"
      return
    fi
    # If it's detached but points to the same branch tip often we still prefer to reset to branch
    log "Existing git dir at $target_dir is on '$current_branch', resetting to '$branch'"
    git -C "$target_dir" fetch --all --prune
    git -C "$target_dir" checkout -B "$branch" "origin/$branch"
    return
  fi

  # If directory exists but isn't a git dir (empty or junk), remove it
  if [ -d "$target_dir" ] && [ -z "$(ls -A "$target_dir" 2>/dev/null)" ]; then
    rmdir "$target_dir"
  fi
  if [ -e "$target_dir" ]; then
    echo "❌ Target path '$target_dir' exists and is not a git dir/empty folder. Remove or move it."; exit 1
  fi

  git -C "$repo_dir" worktree add "$target_dir" "$branch"
  log "Added worktree $target_dir for $branch"
}

# --- Update main repo and ensure worktrees ---------------------------------------
# The script pulls updates in $GITHOME/mammoth
# The script checkouts feat/hf_integration as $GITHOME/mammoth-hf
# The script checkouts feat/helper         as $GITHOME/mammoth-helper

log "Updating $MAMMOTH_REPO"
git -C "$MAMMOTH_REPO" pull --ff-only

ensure_branch_exists "$MAMMOTH_REPO" "$HF_BRANCH"
ensure_branch_exists "$MAMMOTH_REPO" "$HELPER_BRANCH"

# uh, this is hard to read... but the purpose is to have it idempotent 
ensure_worktree "$MAMMOTH_REPO" "$HF_BRANCH"     "$HF_WT_DIR"
ensure_worktree "$MAMMOTH_REPO" "$HELPER_BRANCH" "$HELPER_WT_DIR"

log "git worktree setup complete."

# --- Symlinks into $PROJHOME -----------------------------------------------------
# The script creates symlink $PROJHOME/mammoth-hf
# The script deletes/rebuilds $PROJHOME/venv/mammoth-hf

mkdir -p "$PROJHOME"
mkdir -p "$PROJHOME/base"

cd "$PROJHOME/base"
ln -sfn $PROJHOME/venv     venv
ln -sfn $GITHOME           git
ln -sfn git/mammoth-hf     mammoth-hf
ln -sfn git/mammoth-helper mammoth-helper

check_under "$PROJHOME/base/mammoth-helper/helper/bin/slurm/" 4-module-loads.sh
check_under "$PROJHOME/base/mammoth-helper/helper/bin/conf/" setup-mammoth-hf.py

log "symlink things done at $PROJHOME"

# --- Load module stack ------------------------------------------------------------
cd "$PROJHOME"
# By default, load the partition/L software stack since we are running on login node but if
# this script is run inside srun, partion/C or partition/G will be loaded
export JOB_NODE_KIND=gpu  # Force partition/G stack even on login node
# shellcheck source=../slurm/4-module-loads.sh
source base/mammoth-helper/helper/bin/slurm/4-module-loads.sh
# After this, python is a wrapper that launches a pytorch/rocm container and runs python.
log "module loads done."

# --- Use our own setup.py for the HF branch --------------------------------------

# Build virtual environment for this particular branch (feat/hf_integration as mammoth-hf)
# We use `pip -e .` to install mammoth to the virtual environment via symlinks,
# i.e., in editable form.  This uses the branch-specific `setup.py` file 
# Temporary hack: I do not want to use setup.py provided by feat/hf_integration
# but rather setup-mammoth-hf.py provided in helper/bin/conf.  Rename and copy:

# Backup the original setup.py only once
if [ -f base/mammoth-hf/setup.py ] && [ ! -f base/mammoth-hf/setup.py.orig ]; then
  cp -p base/mammoth-hf/setup.py base/mammoth-hf/setup.py.orig
fi

# Replace with our custom setup if different
CUSTOM_SETUP="base/mammoth-helper/helper/bin/conf/setup-mammoth-hf.py"
if [ ! -f "$CUSTOM_SETUP" ]; then
  echo "❌ Custom setup file not found: $CUSTOM_SETUP"; exit 1
fi
# Copy only if contents differ
if ! cmp -s "$CUSTOM_SETUP" base/mammoth-hf/setup.py; then
  cp -p "$CUSTOM_SETUP" base/mammoth-hf/setup.py
  log "using our own setup.py"
else
  log "custom base/mammoth-hf/setup.py already in place"
fi

# --- Build (clean) virtual environment -------------------------------------------
# Python/pip will run now in a modulerized-container with GPU-enabled pytorch and rocm libraries
# There is no need to do installation inside `srun` nor singularity, but partition/G is useful.
# We create virtual environment so that it inherits pytorch etc in the container
rm -rf venv/mammoth-hf
python-silent -m venv --system-site-packages venv/mammoth-hf
# shellcheck disable=SC1091
source venv/mammoth-hf/bin/activate
log "created and activated venv."

venv/mammoth-hf/bin/python -m pip install --upgrade pip
export PIP_USER=no
cd base/mammoth-hf
$PROJHOME/venv/mammoth-hf/bin/python -m pip install -e .
deactivate

log "installed mammoth and the requirements"

