#!/usr/bin/env python3
# dl-wmt-all.py
#
# Purpose
# -------
# Prepare local WMT24++ benchmark files for the eval2 workflow.
#
# This script downloads selected English-to-target WMT24++ configurations from
# Hugging Face, filters out bad-source examples if requested, normalizes text
# fields into one-line format, and writes the resulting benchmark files into a
# local directory structure under:
#
#     data/wmt24pp/<split>/<config>/
#
# Current behavior
# ----------------
# The script:
#   1) queries all available `google/wmt24pp` dataset configs,
#   2) keeps only English-to-target configs (`en-*`),
#   3) maps a selected set of FLORES-style language identifiers to WMT language
#      codes,
#   4) determines which task languages are supported by WMT24++ and which are
#      missing,
#   5) builds the final list of relevant WMT24++ configs,
#   6) downloads each selected split,
#   7) optionally filters out rows where `is_bad_source` is true,
#   8) writes:
#        - source.en.txt
#        - reference.target.txt
#        - reference_original.target.txt
#        - metadata.tsv
#   9) validates that the written files exist and have the expected line counts.
#
# Inputs
# ------
# Internal settings near the top of the script:
#   split
#       Dataset split to prepare, currently defaulting to "train".
#   filter_bad_source
#       Whether to remove rows with `is_bad_source == True`.
#   languages
#       FLORES-style language identifiers considered relevant to the evaluation
#       workflow.
#   flores_to_wmt
#       Mapping from FLORES-style language identifiers to WMT language codes.
#
# Reads
# -----
#   - Hugging Face dataset: `google/wmt24pp`
#
# Writes
# ------
# For each selected config:
#   data/wmt24pp/<split>/<config>/source.en.txt
#   data/wmt24pp/<split>/<config>/reference.target.txt
#   data/wmt24pp/<split>/<config>/reference_original.target.txt
#   data/wmt24pp/<split>/<config>/metadata.tsv
#
# Output
# ------
# The script prints:
#   - supported task languages,
#   - missing task languages,
#   - selected WMT24++ configs,
#   - per-config row counts and output-file status,
#   - counts of empty source / target / original-target lines.
#
# Notes
# -----
# - Text fields are normalized to one-line format before writing.
# - `metadata.tsv` contains:
#       lp, domain, document_id, segment_id, is_bad_source
# - Missing or malformed outputs cause the script to raise an error.


from datasets import load_dataset
from datasets import get_dataset_config_names
import os

# ----------------------------
# USER SETTINGS
# ----------------------------

split = "train"
filter_bad_source = True
flores_to_wmt = {
    "bul_Cyrl": "bg",
    "cat_Latn": "ca",
    "ces_Latn": "cs",
    "dan_Latn": "da",
    "deu_Latn": "de",
    "ell_Grek": "el",
    "ekk_Latn": "et",
    "fin_Latn": "fi",
    "fra_Latn": "fr",
    "hrv_Latn": "hr",
    "hun_Latn": "hu",
    "isl_Latn": "is",
    "ita_Latn": "it",
    "lvs_Latn": "lv",
    "lit_Latn": "lt",
    "nld_Latn": "nl",
    "nob_Latn": "no",
    "pol_Latn": "pl",
    "por_Latn": "pt",
    "ron_Latn": "ro",
    "slk_Latn": "sk",
    "slv_Latn": "sl",
    "spa_Latn": "es",
    "srp_Cyrl": "sr",
    "swe_Latn": "sv",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
}
languages = [
    "eng_Latn",    "bos_Latn",    "bul_Cyrl",    "cat_Latn",
    "ces_Latn",    "dan_Latn",    "deu_Latn",    "ell_Grek",
    "ekk_Latn",    "eus_Latn",    "fin_Latn",    "fra_Latn",
    "gle_Latn",    "glg_Latn",    "hrv_Latn",    "hun_Latn",
    "isl_Latn",    "ita_Latn",    "kat_Geor",    "lvs_Latn",
    "lit_Latn",    "mkd_Cyrl",    "mlt_Latn",    "nld_Latn",
    "nno_Latn",    "nob_Latn",    "pol_Latn",    "por_Latn",
    "ron_Latn",    "slk_Latn",    "slv_Latn",    "spa_Latn",
    "srp_Cyrl",    "swe_Latn",    "tur_Latn",    "ukr_Cyrl",
    "als_Latn"]

# All WMT24++ English-to-target configs you want to prepare.
# Extend or reduce this list as needed.
all_configs = get_dataset_config_names("google/wmt24pp")
en_configs = [c for c in all_configs if c.startswith("en-")]

from datasets import get_dataset_config_names

languages = [
    "eng_Latn", "bos_Latn", "bul_Cyrl", "cat_Latn",
    "ces_Latn", "dan_Latn", "deu_Latn", "ell_Grek",
    "ekk_Latn", "eus_Latn", "fin_Latn", "fra_Latn",
    "gle_Latn", "glg_Latn", "hrv_Latn", "hun_Latn",
    "isl_Latn", "ita_Latn", "kat_Geor", "lvs_Latn",
    "lit_Latn", "mkd_Cyrl", "mlt_Latn", "nld_Latn",
    "nno_Latn", "nob_Latn", "pol_Latn", "por_Latn",
    "ron_Latn", "slk_Latn", "slv_Latn", "spa_Latn",
    "srp_Cyrl", "swe_Latn", "tur_Latn", "ukr_Cyrl",
    "als_Latn"
]

flores_to_wmt = {
    "bos_Latn": "bs",
    "bul_Cyrl": "bg",
    "cat_Latn": "ca",
    "ces_Latn": "cs",
    "dan_Latn": "da",
    "deu_Latn": "de",
    "ell_Grek": "el",
    "ekk_Latn": "et",
    "eus_Latn": "eu",
    "fin_Latn": "fi",
    "fra_Latn": "fr",
    "gle_Latn": "ga",
    "glg_Latn": "gl",
    "hrv_Latn": "hr",
    "hun_Latn": "hu",
    "isl_Latn": "is",
    "ita_Latn": "it",
    "kat_Geor": "ka",
    "lvs_Latn": "lv",
    "lit_Latn": "lt",
    "mkd_Cyrl": "mk",
    "mlt_Latn": "mt",
    "nld_Latn": "nl",
    "nno_Latn": "nn",
    "nob_Latn": "no",
    "pol_Latn": "pl",
    "por_Latn": "pt",
    "ron_Latn": "ro",
    "slk_Latn": "sk",
    "slv_Latn": "sl",
    "spa_Latn": "es",
    "srp_Cyrl": "sr",
    "swe_Latn": "sv",
    "tur_Latn": "tr",
    "ukr_Cyrl": "uk",
    "als_Latn": "sq",
}

all_configs = get_dataset_config_names("google/wmt24pp")
en_configs = [c for c in all_configs if c.startswith("en-")]

# One pass over WMT configs:
available_wmt_langs = set()
configs_by_wmt_lang = {}
for cfg in en_configs:
    tgt_lang = cfg.split("-")[1].split("_")[0]
    available_wmt_langs.add(tgt_lang)
    configs_by_wmt_lang.setdefault(tgt_lang, []).append(cfg)
# Determine supported/missing task languages
supported_languages = []
missing_languages = []
for lang in languages:
    if lang == "eng_Latn":
        continue
    wmt_lang = flores_to_wmt.get(lang)
    if wmt_lang is None or wmt_lang not in available_wmt_langs:
        missing_languages.append(lang)
    else:
        supported_languages.append(lang)
# Build relevant configs from supported languages
relevant_configs = sorted({
    cfg
    for lang in supported_languages
    for cfg in configs_by_wmt_lang[flores_to_wmt[lang]]
})
print("Supported task languages in WMT24++:")
for lang in supported_languages:
    wmt_lang = flores_to_wmt[lang]
    print(f"  {lang} -> {wmt_lang} -> {', '.join(configs_by_wmt_lang[wmt_lang])}")
print("\nMissing task languages in WMT24++:")
for lang in missing_languages:
    print(f"  {lang} -> {flores_to_wmt.get(lang, 'NO_MAPPING')}")
print("\nRelevant WMT24++ configs:")
for cfg in relevant_configs:
    print(" ", cfg)

required_keys = {
    "lp",
    "domain",
    "document_id",
    "segment_id",
    "is_bad_source",
    "source",
    "target",
    "original_target",
}
# ----------------------------
# HELPERS
# ----------------------------
def one_line(text):
    if text is None:
        return ""
    return " ".join(" ".join(str(text).splitlines()).split()).strip()
def count_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)
def ensure_line_count(path, expected, label):
    actual = count_lines(path)
    if actual != expected:
        raise ValueError(f"{label}: {actual} lines vs expected {expected}")
def ensure_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing output file: {path}")

# ----------------------------
# MAIN LOOP
# ----------------------------
for config in relevant_configs:
    print(f"\n=== Preparing {config} ({split}) ===")
    ds = load_dataset("google/wmt24pp", config, split=split)
    if len(ds) == 0:
        raise ValueError(f"{config}: dataset is empty")
    keys = set(ds[0].keys())
    missing = required_keys - keys
    if missing:
        raise KeyError(f"{config}: missing required keys {sorted(missing)}")
    print(f"Loaded {len(ds)} rows")
    if filter_bad_source:
        ds = ds.filter(lambda ex: not ex["is_bad_source"])
        if len(ds) == 0:
            raise ValueError(f"{config}: removed by bad-source filtering")
        print(f"Rows after bad-source filtering: {len(ds)}")
        
    outdir = f"data/wmt24pp/{split}/{config}"
    os.makedirs(outdir, exist_ok=True)
    src_path = os.path.join(outdir, "source.en.txt")
    ref_path = os.path.join(outdir, "reference.target.txt")
    orig_path = os.path.join(outdir, "reference_original.target.txt")
    meta_path = os.path.join(outdir, "metadata.tsv")
    n_rows = 0
    n_empty_source = 0
    n_empty_target = 0
    n_empty_original = 0
    with open(src_path, "w", encoding="utf-8") as fsrc, \
         open(ref_path, "w", encoding="utf-8") as fref, \
         open(orig_path, "w", encoding="utf-8") as forig, \
         open(meta_path, "w", encoding="utf-8") as fmeta:
        fmeta.write("lp\tdomain\tdocument_id\tsegment_id\tis_bad_source\n")
        for ex in ds:
            src = one_line(ex["source"])
            tgt = one_line(ex["target"])
            orig = one_line(ex["original_target"])
            if not src:
                n_empty_source += 1
            if not tgt:
                n_empty_target += 1
            if not orig:
                n_empty_original += 1
            fsrc.write(src + "\n")
            fref.write(tgt + "\n")
            forig.write(orig + "\n")
            fmeta.write(
                f"{ex['lp']}\t{ex['domain']}\t{ex['document_id']}\t"
                f"{ex['segment_id']}\t{ex['is_bad_source']}\n"
            )
            n_rows += 1
    ensure_exists(src_path)
    ensure_exists(ref_path)
    ensure_exists(orig_path)
    ensure_exists(meta_path)
    ensure_line_count(src_path, n_rows, "Src")
    ensure_line_count(ref_path, n_rows, "Ref")
    ensure_line_count(orig_path, n_rows, "Org ref")
    ensure_line_count(meta_path, n_rows + 1, "Metadata")
    print(f"Wrote {src_path}")
    print(f"Wrote {ref_path}")
    print(f"Wrote {orig_path}")
    print(f"Wrote {meta_path}")
    print(f"Final rows: {n_rows}")
    print(f"Empty source lines: {n_empty_source}")
    print(f"Empty target lines: {n_empty_target}")
    print(f"Empty original target lines: {n_empty_original}")
print("\nDone. All requested WMT24++ configs prepared.")
