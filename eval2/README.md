# Mammoth eval2 tools

This directory contains a newer set of evaluation helpers under development. Compared with the earlier `eval` workflow, the tools here are more modular and increasingly Makefile-driven. At the moment, some scripts are already specific enough to document in detail, while others are still evolving and are best described only at a high level.

## Current scope

The tools in this directory appear to cover four main areas:

1. **Benchmark data preparation**
   - download and normalize benchmark datasets
   - store them in a local directory structure expected by the evaluation workflow

2. **Inference planning**
   - choose language pairs and evaluation tasks
   - prepare inference YAML files and call lists

3. **Inference execution**
   - distribute inference calls across Slurm tasks and GPUs
   - run inference inside the configured runtime environment

4. **Scoring and reporting**
   - compute metric outputs
   - aggregate or summarize results

Because this workflow is still under active development, the exact interfaces of some scripts may still change.

## Directory contents

Current files include:

- dataset download / preparation scripts:
  - `dl-bouquet-all.py`
  - `dl-bouquet-all-par.py`
  - `dl-flores-all.py`
  - `dl-wmt-all.py`

- inference planning / task selection:
  - `inf_pairs.py`
  - `inf_plan.py`
  - `inf_top_zeroshots.py`
  - `inf_top_zeroshots.sh`

- inference runtime helpers:
  - `inf_distr.sh`
  - `inf_wrapper.sh`

- Makefile fragments:
  - `mk-shared.mk`
  - `mk-yamls.mk`
  - `mk-calls.mk`
  - `mk-infer.mk`
  - `mk-score.mk`
  - `Makefile`

## Script descriptions

### `dl-wmt-all.py`

`dl-wmt-all.py` prepares the WMT24++ benchmark data needed by the evaluation workflow. It downloads selected English-to-target WMT24++ configurations from Hugging Face, filters out bad-source examples, normalizes text fields into one-line format, and writes benchmark files into a local directory structure under `data/wmt24pp/<split>/<config>/`. 

The script currently:

- queries all available `google/wmt24pp` configs,
- keeps only `en-*` configs,
- maps a selected set of FLORES-style language identifiers to WMT language codes,
- determines which task languages are supported and which are missing,
- builds the list of relevant WMT24++ configs,
- downloads each selected dataset split,
- optionally filters out rows where `is_bad_source` is true,
- writes:
  - `source.en.txt`
  - `reference.target.txt`
  - `reference_original.target.txt`
  - `metadata.tsv`
- validates that the written files exist and have the expected line counts.

For each selected config, the output files are written to:

```text
data/wmt24pp/<split>/<config>/
```

### `dl-flores-all.py`

`dl-flores-all.py` prepares local FLORES+ benchmark files for the eval2 workflow.

It downloads the selected FLORES+ languages from Hugging Face, verifies that they are available, checks that each language split is aligned with the English reference split, and writes one plain-text file per language and split under `data/flores_plus/`.

For each selected language, the script:

- checks that the language is available in FLORES+,
- loads both `dev` and `devtest`,
- compares the language split against the corresponding English split,
- verifies matching feature keys,
- verifies matching row counts,
- verifies matching example IDs,
- writes one output text file per split.

The resulting files are written as:

- `data/flores_plus/dev/<language>.txt`
- `data/flores_plus/devtest/<language>.txt`

Typical use:

```bash
python dl-flores-all.py
```



### `dl-bouquet-all.py` and `dl-bouquet-all-par.py`

`dl-bouquet-all.py` and `dl-bouquet-all-par.py` prepare local BOUQuET benchmark files for the eval2 workflow.

Both scripts load the public BOUQuET dataset from Hugging Face, reorganize it into language-centered plain-text files, and write one file per language and split. The scripts use the same extraction logic, but target different output roots and configurations:

- `dl-bouquet-all.py`
  - prepares sentence-level BOUQuET files under `data/bouquet/`
- `dl-bouquet-all-par.py`
  - prepares paragraph-level BOUQuET files under `data/bouquet_par/`

For each selected split, the scripts:

- check that the split exists and is non-empty,
- verify that the required dataset columns are present,
- collect all source-side and target-side texts by language and `uniq_id`,
- check that repeated occurrences of the same language and `uniq_id` are textually consistent,
- write one output file per language, ordered by `uniq_id`.

The resulting files are written as:

- `data/bouquet/dev/<language>.txt`
- `data/bouquet/test/<language>.txt`
- `data/bouquet_par/dev/<language>.txt`
- `data/bouquet_par/test/<language>.txt`

If `CHECK_EXISTING` is enabled, each script compares existing files against newly extracted content and fails on mismatch instead of overwriting them.

Typical use:

```bash
python dl-bouquet-all.py
python dl-bouquet-all-par.py
```



### `inf_top_zeroshots.py`

`inf_top_zeroshots.py` ranks candidate zero-shot evaluation pairs from a Mammoth `train.yaml`.

It reads the supervised translation tasks already defined in the training configuration, interprets them as a directed language graph, and proposes source-target pairs that are not directly present as training tasks but are reachable through the graph.

The ranking combines several signals:

- graph reachability and shortest-path structure,
- pivot availability,
- pivot hub strength based on centrality,
- availability of alternative shortest paths,
- linguistic compatibility based on URIEL/lang2vec distances.

The script is intended as a planning aid for selecting promising zero-shot evaluation pairs.

Typical use:

```bash
python inf_top_zeroshots.py train.yaml
```


### `inf_pairs.py`

`inf_pairs.py` extracts and ranks evaluation language pairs from a Mammoth `train.yaml`.

It is part of the eval2 planning workflow and supports two main uses:

- writing the supervised `src-tgt` pairs found in the training configuration,
- ranking candidate zero-shot pairs that are not directly present as training tasks.

To do this, the script reads the supervised translation tasks from `train.yaml`, interprets them as a directed language graph, and uses graph structure together with URIEL/lang2vec-based linguistic compatibility scores to rank candidate zero-shot pairs.

Typical use:

```bash
python inf_pairs.py train.yaml --zs-out zeroshot_pairs.input

#Write all supervised pairs and quit:
python inf_pairs.py train.yaml --supervised-pairs-and-quit supervised_pairs.input

#Print only the grouped zero-shot rank list:
python inf_pairs.py train.yaml --rank-list-only
```



### `inf_plan.py`

`inf_plan.py` is the main planning script of the eval2 workflow.

It reads the training configuration together with the selected supervised and zero-shot pair lists, expands the evaluation tasks into localized variants, generates per-task inference YAML files, and writes the call lists used later for translation and scoring.

The script performs the following main steps:

1. validates the required environment variables and directories,
2. builds an internal language inventory from the hard-coded `TRIPLES` table,
3. extracts and expands supervised tasks from `TRAINCONFIG`,
4. filters them against the selected supervised pairs,
5. checks that the selected tasks are covered by the available benchmark data,
6. adds localized zero-shot tasks from the selected zero-shot pairs,
7. resolves sharing groups, vocab codes, transforms, and template tasks,
8. writes one inference YAML per original training task,
9. writes:
   - `calls.out`
   - `calls.sacre.out`
   - `calls.comet.out`

Main outputs:

- `OUTDIR/*.yaml`
  - per-task inference YAML files
- `OUTDIR/calls.out`
  - inference commands for `translate.py`
- `OUTDIR/calls.sacre.out`
  - SacreBLEU / chrF scoring commands
- `OUTDIR/calls.comet.out`
  - COMET scoring commands

Typical use is indirect, via the surrounding Makefile workflow.


