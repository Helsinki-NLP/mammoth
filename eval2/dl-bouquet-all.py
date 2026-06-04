#!/usr/bin/env python3
# dl-bouquet-all.py
#
# Purpose
# -------
# Prepare local sentence-level BOUQuET benchmark files for the eval2 workflow.
#
# This script loads the public BOUQuET dataset from Hugging Face, reorganizes
# it into language-centered plain-text files, and writes one file per language
# and split under:
#
#     data/bouquet/<split>/<language>.txt
#
# Current behavior
# ----------------
# The script:
#   1) selects a BOUQuET configuration (`default`, `sentence_level`,
#      or `paragraph_level`),
#   2) loads the requested public splits,
#   3) checks that each split is present and non-empty,
#   4) verifies that the required dataset columns exist,
#   5) collects all texts by language and `uniq_id`,
#   6) checks that repeated occurrences of the same language/id pair have
#      consistent text,
#   7) writes one text file per language, ordered by `uniq_id`.
#
# Inputs
# ------
# Internal settings near the top of the script:
#   CONFIG
#       Dataset configuration to load:
#         - "default"
#         - "sentence_level"
#         - "paragraph_level"
#   SPLITS
#       Public dataset splits to prepare, currently `dev` and `test`.
#   OUT_ROOT
#       Root directory where the language-centered files are written.
#   CHECK_EXISTING
#       If True, compare existing files against newly extracted content and
#       fail on mismatch instead of silently overwriting.
#
# Reads
# -----
#   - Hugging Face dataset: `facebook/bouquet`
#
# Writes
# ------
# For each selected split and language:
#   data/bouquet/<split>/<language>.txt
#
# Output
# ------
# The script prints:
#   - file-writing progress,
#   - file-match confirmations when CHECK_EXISTING is enabled,
#   - a final completion message.
#
# Notes
# -----
# - Files are language-centered rather than pair-centered.
# - Text lines are ordered by `uniq_id`.
# - The script aborts if:
#     - a requested split is missing or empty,
#     - required dataset columns are missing,
#     - inconsistent texts are found for the same language and `uniq_id`,
#     - an existing output file differs from the newly extracted content.

from collections import defaultdict
from datasets import load_dataset
import os
# ----------------------------
# USER SETTINGS
# ----------------------------
# Config options:
#   "default"            -> both levels together
#   "sentence_level"     -> sentence level only
#   "paragraph_level"    -> paragraph level only
CONFIG = "sentence_level"
# BOUQuET public splits
SPLITS = ["dev", "test"]
# Write language-centered files here
OUT_ROOT = "data/bouquet"
# If True, compare existing files and fail on mismatch
CHECK_EXISTING = True
# ----------------------------
# HELPERS
# ----------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
def read_text_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
def write_text_file(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")
def compare_or_write(path, new_lines):
    if os.path.exists(path):
        old_lines = read_text_file(path)
        if old_lines == new_lines:
            print(f"OK: existing file matches newly extracted data: {path}")
            return
        # Find first mismatch to make debugging easier
        min_len = min(len(old_lines), len(new_lines))
        for i in range(min_len):
            if old_lines[i] != new_lines[i]:
                raise ValueError(
                    f"Mismatch in existing file {path} at line {i+1}\n"
                    f"OLD: {old_lines[i]!r}\n"
                    f"NEW: {new_lines[i]!r}"
                )
        if len(old_lines) != len(new_lines):
            raise ValueError(
                f"Mismatch in existing file {path}: "
                f"{len(old_lines)} old lines vs {len(new_lines)} new lines"
            )
    else:
        write_text_file(path, new_lines)
        print(f"Wrote: {path}")
# ----------------------------
# LOAD DATA
# ----------------------------
if CONFIG == "default":
    ds_all = load_dataset("facebook/bouquet")
else:
    ds_all = load_dataset("facebook/bouquet", CONFIG)
# ----------------------------
# EXTRACT LANGUAGE-CENTERED FILES
# ----------------------------
for split in SPLITS:
    if split not in ds_all:
        raise KeyError(f"Split {split!r} not found in loaded dataset")
    ds = ds_all[split]
    if len(ds) == 0:
        raise ValueError(f"Split {split!r} is empty")
    required = {
        "uniq_id",
        "src_lang",
        "tgt_lang",
        "src_text",
        "tgt_text",
    }
    missing = required - set(ds.features.keys())
    if missing:
        raise KeyError(f"Missing required columns in split {split}: {sorted(missing)}")
    # language -> uniq_id -> text
    texts_by_lang = defaultdict(dict)
    for ex in ds:
        uid = ex["uniq_id"]
        src_lang = ex["src_lang"]
        tgt_lang = ex["tgt_lang"]
        src_text = ex["src_text"].strip()
        tgt_text = ex["tgt_text"].strip()
        # Insert source-side text
        if uid in texts_by_lang[src_lang]:
            if texts_by_lang[src_lang][uid] != src_text:
                raise ValueError(
                    f"Inconsistent text for language {src_lang}, uniq_id {uid}\n"
                    f"OLD: {texts_by_lang[src_lang][uid]!r}\n"
                    f"NEW: {src_text!r}")
        else:
            texts_by_lang[src_lang][uid] = src_text
        # Insert target-side text
        if uid in texts_by_lang[tgt_lang]:
            if texts_by_lang[tgt_lang][uid] != tgt_text:
                raise ValueError(
                    f"Inconsistent text for language {tgt_lang}, uniq_id {uid}\n"
                    f"OLD: {texts_by_lang[tgt_lang][uid]!r}\n"
                    f"NEW: {tgt_text!r}")
        else:
            texts_by_lang[tgt_lang][uid] = tgt_text
    outdir = os.path.join(OUT_ROOT, split)
    ensure_dir(outdir)
    # Write one file per language, ordered by uniq_id
    for lang, uid_to_text in sorted(texts_by_lang.items()):
        ordered_ids = sorted(uid_to_text.keys())
        ordered_lines = [uid_to_text[uid] for uid in ordered_ids]
        out_path = os.path.join(outdir, f"{lang}.txt")
        if CHECK_EXISTING:
            compare_or_write(out_path, ordered_lines)
        else:
            write_text_file(out_path, ordered_lines)
            print(f"Wrote: {out_path}")
print("Done.")
