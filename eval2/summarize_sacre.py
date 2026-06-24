#!/usr/bin/env python3
# summarize_sacre.py
#
# Purpose
# -------
# Read SacreBLEU result files and summarize them as human-readable ASCII tables.
#
# This script scans a directory of *.sacre files, parses the embedded JSON score
# payloads, groups the files by dataset and language pair, and prints summary
# matrices or tables to stdout.
#
# Supported datasets:
#   - WMT24++
#   - Flores+
#   - Bouquet
#
# Supported metrics:
#   - BLEU
#   - chrF2
#   - TER
#
# The script currently expects score files whose names encode:
#   - task kind: mt / sentmt / docmt
#   - source language side
#   - target language side
#   - dataset tag
#
# Typical use
# -----------
#   python summarize_sacre.py INF_SCORES --kind mt
#
# Example:
#   python summarize_sacre.py inf_scores --kind docmt
#
# Single metric only:
#   python summarize_sacre.py inf_scores --kind mt --metric BLEU
#
# Wide table output:
#   python summarize_sacre.py inf_scores --kind mt --wide
#
# Inputs
# ------
# Positional:
#   indir
#       Directory containing *.sacre files
#
# Required option:
#   --kind
#       One of: mt, sentmt, docmt
#
# Optional:
#   --metric
#       Restrict output to one metric
#   --codes
#       Use language codes instead of names in labels
#   --wide
#       Also print one wide table per dataset with all metrics
#
# Reads
# -----
#   - *.sacre files in the input directory
#
# Writes
# ------
#   - no files directly
#
# Output
# ------
#   - ASCII matrices or tables printed to stdout
#   - warnings about malformed or skipped files printed to stderr
#
# Notes
# -----
# The script expects each .sacre file to contain a JSON array with metric items.
# It extracts only recognized metrics and ignores unknown entries.

import argparse
import json
import os
import re
import sys
from collections import defaultdict


DATASET_NAMES = {
    "wmt": "WMT24++",
    "flo": "Flores+",
    "bqt": "Bouquet",
    "bqtpar": "Bouquet-par",
}

# METRICS = ["BLEU", "chrF2", "TER"]
METRICS = ["BLEU", "chrF2"]

# Language-name map for the codes that appear in your files.
# Fallback is the code itself if something is missing.
LANG_NAMES = {
    "eng": "English",
    "por": "Portuguese",
    "fra": "French",
    "bos": "Bosnian",
    "bul": "Bulgarian",
    "cat": "Catalan",
    "ces": "Czech",
    "dan": "Danish",
    "deu": "German",
    "ell": "Greek",
    "est": "Estonian",
    "eus": "Basque",
    "fin": "Finnish",
    "gle": "Irish",
    "glg": "Galician",
    "hrv": "Croatian",
    "hun": "Hungarian",
    "isl": "Icelandic",
    "ita": "Italian",
    "kat": "Georgian",
    "lav": "Latvian",
    "lit": "Lithuanian",
    "mkd": "Macedonian",
    "mlt": "Maltese",
    "nld": "Dutch",
    "nno": "Norwegian Nynorsk",
    "nob": "Norwegian Bokmal",
    "pol": "Polish",
    "ron": "Romanian",
    "slk": "Slovak",
    "slv": "Slovenian",
    "spa": "Spanish",
    "sqi": "Albanian",
    "srp": "Serbian",
    "swe": "Swedish",
    "tur": "Turkish",
    "ukr": "Ukrainian",
}


def parse_lang_token(token: str):
    """
    Parse one side of a pair like:
      XX.eng
      BR.por
      XX.srp_Cyrl
    Returns:
      (lang_code, country_code_or_None)
    Rules:
      - drop country XX
      - drop script suffixes like _Cyrl
    """
    m = re.fullmatch(r"([A-Z]{2})\.([A-Za-z_]+)", token)
    if not m:
        raise ValueError(f"Cannot parse language token: {token}")

    country, lang = m.groups()
    lang = lang.split("_", 1)[0]  # ignore script, e.g. srp_Cyrl -> srp
    country = None if country == "XX" else country
    return lang, country


def lang_display(lang: str, country: str | None, use_names: bool = True):
    base = LANG_NAMES.get(lang, lang) if use_names else lang
    if country:
        return f"{base} ({country})"
    return base


def pair_display(src_lang, src_country, tgt_lang, tgt_country, use_names=True):
    src = lang_display(src_lang, src_country, use_names=use_names)
    tgt = lang_display(tgt_lang, tgt_country, use_names=use_names)
    return f"{src} -> {tgt}"


def sort_key_for_pair(src_lang, src_country, tgt_lang, tgt_country, use_names=True):
    src = lang_display(src_lang, src_country, use_names=use_names)
    tgt = lang_display(tgt_lang, tgt_country, use_names=use_names)
    return (src.lower(), tgt.lower())


def parse_filename(filename: str, kind: str):
    base = os.path.basename(filename)
    m = re.fullmatch(
        rf"{re.escape(kind)}_([A-Z]{{2}}\.[A-Za-z_]+)-([A-Z]{{2}}\.[A-Za-z_]+)\.(wmt|flo|bqt|bqtpar)\.(0s)?sacre",
        base,
    )
    if not m:
        return None

    src_tok, tgt_tok, dataset, zs = m.groups()
    src_lang, src_country = parse_lang_token(src_tok)
    tgt_lang, tgt_country = parse_lang_token(tgt_tok)

    return {
        "dataset": dataset,
        "src_lang": src_lang,
        "src_country": src_country,
        "tgt_lang": tgt_lang,
        "tgt_country": tgt_country,
    }

def parse_sacre_file(path: str):
    """
    SacreBLEU file has a header, then a JSON array.
    Return a dict of scores, or raise ValueError if the file is malformed.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0 or end < start:
        raise ValueError("No JSON array found")

    payload = json.loads(text[start:end + 1])
    scores = {}
    for item in payload:
        name = item.get("name")
        score = item.get("score")
        if name in METRICS and isinstance(score, (int, float)):
            scores[name] = float(score)
    return scores




def lang_code_display(lang: str, country: str | None, dataset: str):
    # Keep country variants only for WMT
    if dataset == "wmt" and country:
        return f"{lang}+{country}"
    return lang


def build_matrix(dataset_rows, metric, dataset):
    """
    Build a matrix:
      rows    = source language labels
      columns = target language labels
      cells   = metric scores
    """
    row_labels = set()
    col_labels = set()
    values = {}

    for item in dataset_rows:
        val = item["scores"].get(metric)
        if val is None:
            continue

        src = lang_code_display(item["src_lang"], item["src_country"], dataset)
        tgt = lang_code_display(item["tgt_lang"], item["tgt_country"], dataset)

        row_labels.add(src)
        col_labels.add(tgt)
        key = (src, tgt)
#        if key in values:
#            print(f"Warning: duplicate cell for {dataset} {metric}: {src} -> {tgt}", file=sys.stderr)
        values[key] = val

    row_labels = sorted(row_labels, key=str.lower)
    col_labels = sorted(col_labels, key=str.lower)
    return row_labels, col_labels, values


def ascii_matrix(row_labels, col_labels, values, title=None, cell_fmt="{:.1f}"):
    """
    Render an ASCII matrix with source langs as rows and target langs as columns.
    """
    headers = ["src \\ tgt"] + col_labels

    def cell_value(r, c):
        v = values.get((r, c))
        return "" if v is None else cell_fmt.format(v)

    rows = []
    for r in row_labels:
        rows.append([r] + [cell_value(r, c) for c in col_labels])

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        return "|" + "|".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + "|"

    # sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    sep = "+" + "+".join("-" * w for w in widths) + "+"
    
    lines = []
    if title:
        lines.append(title)
    lines.append(sep)
    lines.append(fmt_row(headers))
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def print_metric_matrices(dataset_to_rows, only_metric=None):
    metrics = [only_metric] if only_metric else METRICS

    for dataset in ["wmt", "flo", "bqt", "bqtpar"]:
        dataset_name = DATASET_NAMES.get(dataset, dataset)
        rows_for_dataset = dataset_to_rows.get(dataset, [])

        for metric in metrics:
            row_labels, col_labels, values = build_matrix(rows_for_dataset, metric, dataset)
            
            title = f"{dataset_name} — {metric}"
            print(ascii_matrix(row_labels, col_labels, values, title=title))
            print()

           
def ascii_table(headers, rows, title=None):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row):
        return "| " + " | ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"

    lines = []
    if title:
        lines.append(title)
    lines.append(sep)
    lines.append(fmt_row(headers))
    lines.append(sep)
    for row in rows:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def collect_scores(indir: str, kind: str, use_names=True, verbose=True):
    dataset_to_rows = defaultdict(list)
    skipped_bad_name = []
    skipped_bad_content = []

    prefix = f"{kind}_"

    for fn in sorted(os.listdir(indir)):
        if not fn.endswith(".sacre") and not fn.endswith(".0ssacre"):
            continue
        if not fn.startswith(prefix):
            continue

        meta = parse_filename(fn, kind)
        if meta is None:
            skipped_bad_name.append(fn)
            continue

        fullpath = os.path.join(indir, fn)

        try:
            scores = parse_sacre_file(fullpath)
        except Exception as e:
            skipped_bad_content.append((fn, str(e)))
            continue

        if not scores:
            skipped_bad_content.append((fn, "No recognized metrics"))
            continue

        pair_label = pair_display(
            meta["src_lang"], meta["src_country"],
            meta["tgt_lang"], meta["tgt_country"],
            use_names=use_names,
        )
        skey = sort_key_for_pair(
            meta["src_lang"], meta["src_country"],
            meta["tgt_lang"], meta["tgt_country"],
            use_names=use_names,
        )

        dataset_to_rows[meta["dataset"]].append({
            "pair_label": pair_label,
            "sort_key": skey,
            "scores": scores,
            "filename": fn,
            "src_lang": meta["src_lang"],
            "src_country": meta["src_country"],
            "tgt_lang": meta["tgt_lang"],
            "tgt_country": meta["tgt_country"],
        })

    for dataset in dataset_to_rows:
        dataset_to_rows[dataset].sort(key=lambda x: x["sort_key"])

    if verbose:
        if skipped_bad_name:
            print(f"Skipped {len(skipped_bad_name)} {kind} files with unrecognized filenames:", file=sys.stderr)
            for fn in skipped_bad_name:
                print(f"  {fn}", file=sys.stderr)

        if skipped_bad_content:
            print(f"Skipped {len(skipped_bad_content)} malformed or incomplete {kind} score files:", file=sys.stderr)
            for fn, err in skipped_bad_content:
                print(f"  {fn}: {err}", file=sys.stderr)

    return dataset_to_rows


def old_collect_scores(indir: str, use_names=True, verbose=True):
    dataset_to_rows = defaultdict(list)
    skipped_bad_name = []
    skipped_bad_content = []

    for fn in sorted(os.listdir(indir)):
        if not fn.endswith(".sacre"):
            continue

        meta = parse_filename(fn)
        if meta is None:
            skipped_bad_name.append(fn)
            continue

        fullpath = os.path.join(indir, fn)

        try:
            scores = parse_sacre_file(fullpath)
        except Exception as e:
            skipped_bad_content.append((fn, str(e)))
            continue

        # Optional: skip files that parsed but contained no known metrics
        if not scores:
            skipped_bad_content.append((fn, "No recognized metrics"))
            continue

        pair_label = pair_display(
            meta["src_lang"], meta["src_country"],
            meta["tgt_lang"], meta["tgt_country"],
            use_names=use_names,
        )
        skey = sort_key_for_pair(
            meta["src_lang"], meta["src_country"],
            meta["tgt_lang"], meta["tgt_country"],
            use_names=use_names,
        )

        dataset_to_rows[meta["dataset"]].append({
            "pair_label": pair_label,
            "sort_key": skey,
            "scores": scores,
            "filename": fn,
            "src_lang": meta["src_lang"],
            "src_country": meta["src_country"],
            "tgt_lang": meta["tgt_lang"],
            "tgt_country": meta["tgt_country"],
        })

    for dataset in dataset_to_rows:
        dataset_to_rows[dataset].sort(key=lambda x: x["sort_key"])

    if verbose:
        if skipped_bad_name:
            print(f"Skipped {len(skipped_bad_name)} files with unrecognized filenames:", file=sys.stderr)
            for fn in skipped_bad_name:
                print(f"  {fn}", file=sys.stderr)

        if skipped_bad_content:
            print(f"Skipped {len(skipped_bad_content)} malformed or incomplete score files:", file=sys.stderr)
            for fn, err in skipped_bad_content:
                print(f"  {fn}: {err}", file=sys.stderr)

    return dataset_to_rows


def format_score(x):
    return "" if x is None else f"{x:.1f}"


def print_metric_tables(dataset_to_rows):
    for dataset in ["wmt", "flo", "bqt", "bqtpar"]:
        dataset_name = DATASET_NAMES.get(dataset, dataset)
        rows_for_dataset = dataset_to_rows.get(dataset, [])

        for metric in METRICS:
            headers = ["Pair", metric]
            rows = []
            for item in rows_for_dataset:
                val = item["scores"].get(metric)
                if val is not None:
                    rows.append([item["pair_label"], format_score(val)])

            title = f"{dataset_name} — {metric}"
            print(ascii_table(headers, rows, title=title))
            print()


def print_wide_tables(dataset_to_rows):
    """
    Optional: one table per dataset with all metrics side-by-side.
    """
    for dataset in ["wmt", "flo", "bqt", "bqtpar"]:
        dataset_name = DATASET_NAMES.get(dataset, dataset)
        rows_for_dataset = dataset_to_rows.get(dataset, [])

        headers = ["Pair"] + METRICS
        rows = []
        for item in rows_for_dataset:
            rows.append([
                item["pair_label"],
                format_score(item["scores"].get("BLEU")),
                format_score(item["scores"].get("chrF2")),
                format_score(item["scores"].get("TER")),
            ])

        print(ascii_table(headers, rows, title=f"{dataset_name} — all metrics"))
        print()


def main():
    ap = argparse.ArgumentParser(
        description="Summarize sacreBLEU score files into ASCII matrices."
    )
    ap.add_argument("indir", help="Directory containing *.sacre files")
    ap.add_argument(
        "--metric",
        choices=METRICS,
        help="Only print one metric"
    )
    ap.add_argument(
        "--kind",
        required=True,
        choices=["docmt", "sentmt", "mt"],
        help="Which score-file family to read"
    )
    ap.add_argument(
        "--codes",
        action="store_true",
        help="Use language codes instead of language names in pair labels"
    )
    ap.add_argument(
        "--wide",
        action="store_true",
        help="Also print one wide table per dataset with all metrics"
    )
    args = ap.parse_args()

    dataset_to_rows = collect_scores(
        args.indir,
        kind=args.kind,
        use_names=False,
        verbose=True,
    )
    print_metric_matrices(dataset_to_rows, only_metric=args.metric)
    # print_metric_tables(dataset_to_rows)

    if args.wide:
        print_wide_tables(dataset_to_rows)


if __name__ == "__main__":
    main()
    
