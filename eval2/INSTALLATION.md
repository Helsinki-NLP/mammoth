
## 1. Dataset preparation scripts

The archive includes standalone download/preparation helpers:

- `dl-wmt-all.py` — WMT24++;
- `dl-flores-all.py` — FLORES+;
- `dl-bouquet-all.py` — sentence-level BOUQuET;
- `dl-bouquet-all-par.py` — paragraph-level BOUQuET.

They transform public benchmark datasets into the plain-text layout expected by the planner. They are conceptually independent of Slurm and should be documented/configured as data-preparation tools rather than LUMI execution components.

Typical use:
```text
python dl-wmt-all.py
python dl-flores-all.py
python dl-bouquet-all.py
python dl-bouquet-all-par.py
```

### `dl-wmt-all.py`

Prepares WMT24++ benchmark data for the evaluation workflow.

It selects relevant English-to-target configurations from `google/wmt24pp`,
downloads the required splits, filters bad-source examples when applicable,
normalizes text, and writes:

- `source.en.txt`
- `reference.target.txt`
- `reference_original.target.txt`
- `metadata.tsv`

under:

```text
data/wmt24pp/<split>/<config>/
```

### `dl-flores-all.py`

Prepares local FLORES+ benchmark files.

For each selected language it downloads the `dev` and `devtest` splits,
checks their alignment against English using row counts and example IDs, and
writes:

```text
data/flores_plus/dev/<language>.txt
data/flores_plus/devtest/<language>.txt
```

### `dl-bouquet-all.py` and `dl-bouquet-all-par.py`

Prepare sentence-level and paragraph-level BOUQuET benchmark files,
respectively.

The scripts reorganize the public BOUQuET dataset into language-centered
plain-text files. They verify required columns, collect texts by language and
`uniq_id`, check consistency of repeated entries, and write the examples in
`uniq_id` order.

Outputs are written under:

```text
data/bouquet/dev/<language>.txt
data/bouquet/test/<language>.txt
data/bouquet_par/dev/<language>.txt
data/bouquet_par/test/<language>.txt
```

