#!/usr/bin/env bash
set -euo pipefail

# (c) 2025 Anssi Yli-Jyrä, CC-BY

# This script is for temporary need: it allows me to work on two-three
# branches of MAMMOTH at the same time:
#  
#  main              - main branch; for reference/comparison only 
#  feat/integration  - the newest MAMMOTH code; HF integration 
#  feat/helper       - contains the helper-tree of auxiliary script 
#
# These will be stored in the following directories, respectively:
# <your-workspace>/
# ├─ mammoth/                 # the "hosting" repo (has the full .git object DB)
# │  └─ .git/                 # shared metadata + worktree registry
# ├─ mammoth-hf/              # worktree checked out at feat/integration (dependent)
# ├─ mammoth-helper/          # worktree checked out at feat/helper (dependent)
# ├─ some-other-repo/
# └─ yet-another-repo/
#
# This script is idenpotent, but it has not been checked against
# accidental overwriting of uncommitted files.

# --- config ---------------------------------------------------------------
REPO_SSH="git@github.com:Helsinki-NLP/mammoth.git"
REPO_HTTPS="https://github.com/Helsinki-NLP/mammoth.git"
HELPER_BRANCH="feat/helper"
INTEG_BRANCH="feat/integration"
MAIN_DIR="mammoth"
HELPER_DIR="mammoth-helper"
INTEG_DIR="mammoth-hf"

# Switch to https if requested
REPO_URL="$REPO_SSH"
if [[ "${1:-}" == "--https" ]]; then
  REPO_URL="$REPO_HTTPS"
fi

# ---- helpers ----------------------------------------------------------------
die(){ echo "❌ $*" >&2; exit 1; }

git_clean() {  # 0 = clean; 1 = dirty (mods, staged, or untracked)
  git -C "$1" update-index -q --refresh
  [[ -z "$(git -C "$1" status --porcelain=v1 --untracked-files=normal)" ]]
}

have_remote_branch(){ git -C "$1" ls-remote --exit-code --heads origin "$2" >/dev/null 2>&1; }

ensure_anchor(){
  local anchor="$1"
  if [[ -d "$anchor/.git" ]]; then
    git -C "$anchor" remote set-url origin "$REPO_URL"
    git -C "$anchor" fetch --prune --tags
  else
    [[ -e "$anchor" ]] && die "$anchor exists but is not a git repo"
    git clone --no-tags "$REPO_URL" "$anchor"
    git -C "$anchor" fetch --prune --tags
  fi
}

ensure_worktree(){
  # args: anchor_dir worktree_dir target_branch startpoint
  local anchor="$1" wdir="$2" br="$3" start="$4"
  if [[ -d "$wdir/.git" ]]; then
    # already a worktree; ensure it's on the right branch
    git -C "$wdir" checkout -q "$br" || true
    git -C "$wdir" fetch --prune || true
    # only fast-forward if clean
    if git_clean "$wdir"; then
      git -C "$wdir" pull --ff-only || true
    else
      echo "↪  Skipping pull in $wdir (dirty tree)."
    fi
  else
    if [[ -e "$wdir" ]]; then
      die "$wdir exists but is not a git worktree"
    fi
    if have_remote_branch "$anchor" "$br"; then
      git -C "$anchor" worktree add -B "$br" "$wdir" "origin/$br"
    else
      git -C "$anchor" worktree add -B "$br" "$wdir" "$start"
    fi
  fi
  have_remote_branch "$anchor" "$br" && \
    git -C "$wdir" branch --set-upstream-to="origin/$br" "$br" >/dev/null 2>&1 || true
}

rebase_integration_into_helper(){
  local integ="$1" helper="$2"
  # fetch latest integration into helper's view
  git -C "$helper" fetch origin "$INTEG_BRANCH" || true

  if ! git_clean "$helper"; then
    echo "↪  Skipping merge/rebase in $helper (dirty tree). Commit/stash first."
    return 0
  fi

  # Try fast-forward first
  if git -C "$helper" merge --ff-only "origin/$INTEG_BRANCH"; then
    echo "✔ $HELPER_BRANCH fast-forwarded to origin/$INTEG_BRANCH"
    return 0
  fi

  # Otherwise rebase helper onto integration
  echo "ℹ  Rebasing $HELPER_BRANCH onto origin/$INTEG_BRANCH…"
  if git -C "$helper" rebase "origin/$INTEG_BRANCH"; then
    echo "✔ $HELPER_BRANCH rebased on origin/$INTEG_BRANCH"
  else
    echo "❗ Rebase hit conflicts; aborting. Resolve manually in: $helper"
    git -C "$helper" rebase --abort || true
    return 1
  fi
}

# ---- figure out workspace layout -------------------------------------------
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  # inside any worktree: derive anchor and workspace paths
  common_git=$(git rev-parse --git-common-dir)                 # …/mammoth/.git
  anchor_dir=$(cd "$common_git/.." && pwd -P)                  # …/mammoth
  workspace_dir=$(cd "$anchor_dir/.." && pwd -P)
else
  # not in a repo: use current dir as workspace
  workspace_dir=$(pwd -P)
  anchor_dir="$workspace_dir/$ANCHOR_NAME"
fi

integ_dir="$workspace_dir/$INTEG_DIR_NAME"
helper_dir="$workspace_dir/$HELPER_DIR_NAME"

echo "Workspace   : $workspace_dir"
echo "Anchor repo : $anchor_dir  (origin: $REPO_URL)"
echo "Integration : $integ_dir  ← $INTEG_BRANCH"
echo "Helper      : $helper_dir ← $HELPER_BRANCH"
echo

# ---- do the work ------------------------------------------------------------
ensure_anchor "$anchor_dir"

ensure_worktree "$anchor_dir" "$integ_dir" "$INTEG_BRANCH" "origin/$INTEG_BRANCH"
ensure_worktree "$anchor_dir" "$helper_dir" "$HELPER_BRANCH" "origin/$INTEG_BRANCH"

# refresh integration (if clean) to latest origin
if git_clean "$integ_dir"; then
  git -C "$integ_dir" pull --ff-only || true
else
  echo "↪  Skipping pull in $integ_dir (dirty tree)."
fi

# merge/rebase integration → helper (only if helper clean)
rebase_integration_into_helper "$integ_dir" "$helper_dir" || true

echo
echo "Done."
echo "Tip: commit/stash any local changes in '$helper_dir' or '$integ_dir' to allow auto fast-forward/rebase."
