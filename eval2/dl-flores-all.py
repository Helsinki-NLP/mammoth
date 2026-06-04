#!/usr/bin/env python3
# dl-flores-all.py
#
# Purpose
# -------
# Prepare local FLORES+ benchmark files for the eval2 workflow.
#
# This script downloads the selected FLORES+ languages from Hugging Face,
# verifies that they are available, checks that each language split matches
# the English reference split, and writes one plain-text file per language
# and split under:
#
#     data/flores_plus/<split>/<language>.txt
#
# Current behavior
# ----------------
# The script:
#   1) defines the set of FLORES+ languages relevant to the eval2 workflow,
#   2) checks that each requested language is available in the dataset,
#   3) loads the English reference language (`eng_Latn`) for both `dev`
#      and `devtest`,
#   4) writes the English text files first,
#   5) loads each non-English language,
#   6) verifies for each split that:
#        - the feature keys match the English split,
#        - the number of rows matches the English split,
#        - the example IDs match the English split row by row,
#   7) writes one text file per language and split.
#
# Inputs
# ------
# Internal settings near the top of the script:
#   split
#       Default split label used by the workflow. The script currently writes
#       both `dev` and `devtest`.
#   languages
#       List of FLORES+ language identifiers to prepare.
#   av
#       List of dataset configurations considered available.
#
# Reads
# -----
#   - Hugging Face dataset: `openlanguagedata/flores_plus`
#
# Writes
# ------
# For each selected language and split:
#   data/flores_plus/dev/<language>.txt
#   data/flores_plus/devtest/<language>.txt
#
# Output
# ------
# The script prints:
#   - language-availability checks,
#   - dataset-loading progress,
#   - created output directories,
#   - written output files.
#
# Notes
# -----
# - English (`eng_Latn`) is treated as the reference language and is written
#   first for both splits.
# - Each output file contains one stripped text segment per line.
# - The script aborts if any requested language is unavailable or if any
#   language split fails the structural checks against English.

import os
split = "devtest"
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
av = ['default', 'ace_Arab', 'ace_Latn', 'acm_Arab', 'acq_Arab',
      'aeb_Arab', 'afr_Latn', 'als_Latn', 'amh_Ethi', 'lav_Latn',
      'apc_Arab_nort3139', 'apc_Arab_sout3123', 'arb_Arab',
      'arb_Latn', 'arg_Latn', 'ars_Arab', 'ary_Arab', 'arz_Arab',
      'asm_Beng', 'ast_Latn', 'awa_Deva', 'ayr_Latn', 'azb_Arab',
      'azj_Latn', 'bak_Cyrl', 'bam_Latn', 'ban_Latn', 'bel_Cyrl',
      'bem_Latn', 'ben_Beng', 'bho_Deva', 'bjn_Arab', 'bjn_Latn',
      'bod_Tibt', 'bos_Latn', 'brx_Deva', 'bug_Latn', 'bul_Cyrl',
      'cat_Latn', 'cat_Latn_vale1252', 'ceb_Latn', 'ces_Latn',
      'chv_Cyrl', 'cjk_Latn', 'ckb_Arab', 'cmn_Hans', 'cmn_Hant',
      'crh_Latn', 'cym_Latn', 'dan_Latn', 'dar_Cyrl', 'deu_Latn',
      'dgo_Deva', 'dik_Latn', 'dyu_Latn', 'dzo_Tibt', 'ekk_Latn',
      'ell_Grek', 'eng_Latn', 'epo_Latn', 'eus_Latn', 'ewe_Latn',
      'fao_Latn', 'fij_Latn', 'fil_Latn', 'fin_Latn', 'fon_Latn',
      'fra_Latn', 'fur_Latn', 'fuv_Latn', 'gaz_Latn', 'gla_Latn',
      'gle_Latn', 'glg_Latn', 'gom_Deva', 'gug_Latn', 'guj_Gujr',
      'hat_Latn', 'hau_Latn', 'heb_Hebr', 'hin_Deva', 'hne_Deva',
      'hrv_Latn', 'hun_Latn', 'hye_Armn', 'ibo_Latn', 'ilo_Latn',
      'ind_Latn', 'isl_Latn', 'ita_Latn', 'jav_Latn', 'jpn_Jpan',
      'kaa_Latn', 'kab_Latn', 'kac_Latn', 'kam_Latn', 'kan_Knda',
      'kas_Arab', 'kas_Deva', 'kat_Geor', 'kaz_Cyrl', 'kbp_Latn',
      'kea_Latn', 'khk_Cyrl', 'khk_Mong', 'khm_Khmr', 'kik_Latn',
      'kin_Latn', 'kir_Cyrl', 'kjh_Cyrl', 'kmb_Latn', 'kmr_Latn',
      'knc_Arab', 'knc_Latn', 'kor_Hang', 'ktu_Latn', 'lao_Laoo',
      'lij_Latn', 'lim_Latn', 'lin_Latn', 'lit_Latn', 'lld_Latn',
      'lld_Latn_gard1241', 'lmo_Latn', 'ltg_Latn', 'ltz_Latn',
      'lua_Latn', 'lug_Latn', 'luo_Latn', 'lus_Latn', 'lvs_Latn',
      'mag_Deva', 'mai_Deva', 'mal_Mlym', 'mar_Deva', 'mfe_Latn',
      'mhr_Cyrl', 'min_Arab', 'min_Latn', 'mkd_Cyrl', 'mlt_Latn',
      'mni_Beng', 'mni_Mtei', 'mos_Latn', 'mri_Latn', 'mya_Mymr',
      'myv_Cyrl', 'nld_Latn', 'nno_Latn', 'nob_Latn',
      'nob_Latn_radical', 'npi_Deva', 'nqo_Nkoo', 'nso_Latn',
      'nus_Latn', 'nya_Latn', 'oci_Latn', 'oci_Latn_aran1260',
      'ory_Orya', 'pag_Latn', 'pan_Guru', 'pap_Latn', 'pbt_Arab',
      'pes_Arab', 'plt_Latn', 'pol_Latn', 'por_Latn', 'prs_Arab',
      'quy_Latn', 'ron_Latn', 'run_Latn', 'rus_Cyrl', 'sag_Latn',
      'san_Deva', 'sat_Olck', 'scn_Latn', 'shn_Mymr', 'sin_Sinh',
      'slk_Latn', 'slv_Latn', 'smo_Latn', 'sna_Latn', 'snd_Arab',
      'snd_Deva', 'som_Latn', 'sot_Latn', 'spa_Latn', 'srd_Latn',
      'srp_Cyrl', 'ssw_Latn', 'sun_Latn', 'swe_Latn', 'swh_Latn',
      'szl_Latn', 'tam_Taml', 'taq_Latn', 'taq_Tfng', 'tat_Cyrl',
      'tel_Telu', 'tgk_Cyrl', 'tha_Thai', 'tir_Ethi', 'tpi_Latn',
      'tsn_Latn', 'tso_Latn', 'tuk_Latn', 'tum_Latn', 'tur_Latn',
      'twi_Latn_akua1239', 'twi_Latn_asan1239', 'tyv_Cyrl',
      'uig_Arab', 'ukr_Cyrl', 'umb_Latn', 'urd_Arab', 'uzn_Latn',
      'uzs_Arab', 'vec_Latn', 'vie_Latn', 'vmw_Latn', 'war_Latn',
      'wol_Latn', 'wuu_Hans', 'xho_Latn', 'ydd_Hebr', 'yor_Latn',
      'yue_Hant', 'zgh_Tfng', 'zsm_Latn', 'zul_Latn']
for lang in languages:
    if lang not in av:
        print(f"Language {lang} not available.")
        exit(1)
print("Loading reference language: eng_Latn (all splits)")
from datasets import load_dataset
ds_eng_all = load_dataset("openlanguagedata/flores_plus", "eng_Latn")
for split in ["dev", "devtest"]:
    outdir = f"data/flores_plus/{split}"
    print(f"creating the output directory {outdir}...")
    os.makedirs(outdir, exist_ok=True)
    ds_eng = ds_eng_all[split]
    eng_keys = set(ds_eng.features.keys())
    eng_ids = [ex["id"] for ex in ds_eng]
    eng_out = os.path.join(outdir, "eng_Latn.txt")
    with open(eng_out, "w", encoding="utf-8") as f:
        for ex in ds_eng:
            f.write(ex["text"].strip() + "\n")
    print(f"Wrote: {eng_out}")
    print(f"Rows: {len(ds_eng)}")
for lang in languages:
    if lang == "eng_Latn":
        continue        
    print(f"\nLoading {lang} (all splits)")
    ds_all = load_dataset("openlanguagedata/flores_plus", lang)
    for split in ["dev", "devtest"]:
        print(f"  Processing {lang} ({split})")
        ds = ds_all[split]
        outdir = f"data/flores_plus/{split}"
        os.makedirs(outdir, exist_ok=True)
        lang_keys = set(ds.features.keys())
        lang_ids = [ex["id"] for ex in ds]
        # Compare against the corresponding English split
        ds_eng = ds_eng_all[split]
        eng_keys = set(ds_eng.features.keys())
        eng_ids = [ex["id"] for ex in ds_eng]
        if lang_keys != eng_keys:
            raise ValueError(
                f"Key mismatch for {lang} ({split}).\n"
                f"Missing: {sorted(eng_keys - lang_keys)}\n"
                f"Extra: {sorted(lang_keys - eng_keys)}")
        if len(ds) != len(ds_eng):
            raise ValueError(
                f"Row count mismatch for {lang} ({split}): "
                f"{len(ds)} vs {len(ds_eng)}")
        if lang_ids != eng_ids:
            for i, (eid, lid) in enumerate(zip(eng_ids, lang_ids)):
                if eid != lid:
                    raise ValueError(
                        f"ID mismatch for {lang} ({split}) at row {i}: "
                        f"{lid} vs {eid}")
        out_path = os.path.join(outdir, f"{lang}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in ds:
                f.write(ex["text"].strip() + "\n")
        print(f"  Wrote: {out_path}")
print("\nDone. All language files processed.")
