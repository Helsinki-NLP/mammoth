
# Interfaces for Mammoth eval2 tools

This document records the current interfaces of the eval2 scripts. Because the eval2 workflow is still evolving, some sections are already precise while others are intentionally provisional.



## `dl-wmt-all.py`

### Purpose

Prepare local WMT24++ benchmark files for the eval2 workflow.

### Current behavior

The script:

1. queries all available `google/wmt24pp` dataset configs
2. keeps only English-to-target configs (`en-*`)
3. maps a selected set of FLORES-style language identifiers to WMT language codes
4. determines which task languages are supported by WMT24++ and which are missing
5. builds the final list of relevant configs
6. downloads each selected split
7. optionally filters out rows where `is_bad_source` is true
8. writes benchmark files under:
   - `data/wmt24pp/<split>/<config>/`

### Internal settings

Configured near the top of the script:

- `split`
  - dataset split to prepare
- `filter_bad_source`
  - whether to remove rows with `is_bad_source == True`
- `languages`
  - FLORES-style language identifiers considered relevant
- `flores_to_wmt`
  - mapping from FLORES-style language identifiers to WMT language codes

### Reads

- Hugging Face dataset:
  - `google/wmt24pp`

### Writes

For each selected config:

- `data/wmt24pp/<split>/<config>/source.en.txt`
- `data/wmt24pp/<split>/<config>/reference.target.txt`
- `data/wmt24pp/<split>/<config>/reference_original.target.txt`
- `data/wmt24pp/<split>/<config>/metadata.tsv`

### Output

The script prints:

- supported task languages
- missing task languages
- selected WMT24++ configs
- per-config row counts and output-file status
- counts of empty source / target / original-target lines

### File roles

- `source.en.txt`
  - English source sentences, one per line
- `reference.target.txt`
  - cleaned target-language references, one per line
- `reference_original.target.txt`
  - original target references, one per line
- `metadata.tsv`
  - tab-separated metadata with columns:
    - `lp`
    - `domain`
    - `document_id`
    - `segment_id`
    - `is_bad_source`




## `dl-flores-all.py`

### Purpose

Prepare local FLORES+ benchmark files for the eval2 workflow.

### Current behavior

The script:

1. defines the FLORES+ language set relevant to eval2
2. checks that each requested language is available in the dataset
3. loads the English reference language (`eng_Latn`) for both `dev` and `devtest`
4. writes the English text files first
5. loads each non-English language
6. compares each language split against the corresponding English split
7. writes one plain-text file per language and split

### Internal settings

Configured near the top of the script:

- `split`
  - default split label used by the workflow
  - note: the script currently processes both `dev` and `devtest`
- `languages`
  - FLORES+ language identifiers to prepare
- `av`
  - language identifiers considered available

### Reads

- Hugging Face dataset:
  - `openlanguagedata/flores_plus`

### Writes

For each selected language and split:

- `data/flores_plus/dev/<language>.txt`
- `data/flores_plus/devtest/<language>.txt`

### Output format

Each output file contains one stripped text segment per line.

### Validation performed

For each non-English language and split, the script checks that:

- the feature keys match the corresponding English split
- the number of rows matches the corresponding English split
- the example IDs match the corresponding English split row by row

If any of these checks fail, the script raises an error and stops.

### Output messages

The script prints:

- language availability checks
- progress while loading datasets
- created output directories
- names of written files

### Notes

- `eng_Latn` is treated as the reference language
- both `dev` and `devtest` are prepared
- the script aborts immediately if any requested language is unavailable






## `dl-bouquet-all.py` and `dl-bouquet-all-par.py`

### Purpose

Prepare local BOUQuET benchmark files for the eval2 workflow.

The two scripts share the same extraction logic but target different BOUQuET views:

- `dl-bouquet-all.py`
  - sentence-level BOUQuET files under `data/bouquet/`
- `dl-bouquet-all-par.py`
  - paragraph-level BOUQuET files under `data/bouquet_par/`

### Current behavior

Each script:

1. selects a BOUQuET configuration
2. loads the requested public splits
3. checks that each split is present and non-empty
4. verifies that the required dataset columns exist
5. collects all texts by language and `uniq_id`
6. verifies that repeated occurrences of the same language/id pair are consistent
7. writes one plain-text file per language and split

### Internal settings

Configured near the top of each script:

- `CONFIG`
  - dataset configuration to load
  - supported values:
    - `default`
    - `sentence_level`
    - `paragraph_level`

- `SPLITS`
  - dataset splits to prepare
  - currently:
    - `dev`
    - `test`

- `OUT_ROOT`
  - root directory for output files
  - current defaults:
    - `data/bouquet` for `dl-bouquet-all.py`
    - `data/bouquet_par` for `dl-bouquet-all-par.py`

- `CHECK_EXISTING`
  - if `True`, compare existing files with newly extracted content and fail on mismatch

### Reads

- Hugging Face dataset:
  - `facebook/bouquet`

### Required dataset columns

For each loaded split, the scripts expect the following columns:

- `uniq_id`
- `src_lang`
- `tgt_lang`
- `src_text`
- `tgt_text`

If any required column is missing, the script raises an error.

### Writes

For each selected split and language:

- `data/bouquet/dev/<language>.txt`
- `data/bouquet/test/<language>.txt`
- `data/bouquet_par/dev/<language>.txt`
- `data/bouquet_par/test/<language>.txt`

depending on which script is used.

### Output format

Each output file contains one stripped text segment per line.

The lines are ordered by `uniq_id`.

### Consistency checks

For each example, the scripts insert both:

- the source-side text under `src_lang`,
- the target-side text under `tgt_lang`.

If the same language and `uniq_id` pair occurs again with a different text, the script raises an error.

### Existing-file handling

If `CHECK_EXISTING` is `True`:

- existing files are read back,
- their contents are compared line by line against the newly extracted content,
- the script prints a confirmation if the contents match,
- the script raises an error if a mismatch is found.

If `CHECK_EXISTING` is `False`, files are written unconditionally.

### Output messages

The scripts print:

- written file paths,
- existing-file match confirmations,
- a final `Done.` message.

### Notes

- files are language-centered rather than translation-pair-centered
- both source-side and target-side texts contribute to the same per-language files
- `dl-bouquet-all.py` is intended for sentence-level BOUQuET preparation
- `dl-bouquet-all-par.py` is intended for paragraph-level BOUQuET preparation






## INTERFACES.md section

```md id="8eflxq"
## `inf_top_zeroshots.py`

### Purpose

Rank candidate zero-shot evaluation pairs from a Mammoth training configuration.

### Intended use

This script belongs to the planning stage of the eval2 workflow. It is intended to help select which zero-shot language pairs should be evaluated, based on the supervised task graph already present in `train.yaml`.

### Calling convention

```bash
python inf_top_zeroshots.py TRAIN_YAML [options]
```








## `inf_pairs.py`

### Purpose

Extract supervised evaluation pairs and rank candidate zero-shot evaluation pairs from a Mammoth training configuration.

### Intended use

This script belongs to the planning stage of the eval2 workflow. It is used to derive pair-selection inputs from `train.yaml` before the main inference planner is run.

In the current workflow, it is used to produce:

- a suggested zero-shot pair list
- a supervised pair list

for later consumption by `inf_plan.py`.

### Calling convention

```bash
python inf_pairs.py TRAIN_YAML [options]
```







```md
### INTERFACES.md section

```md
## `inf_plan.py`

### Purpose

Plan evaluation tasks, generate per-task inference YAML files, and prepare inference and scoring call lists for the eval2 workflow.

### Intended use

This script belongs to the planning stage of eval2. It is intended to be called from the Makefile workflow after the supervised and zero-shot pair lists have already been prepared.

### Execution model

The script does not use command-line arguments. Instead, it reads its configuration from environment variables.

### Required environment variables

- `DATADIR`
  - root directory of benchmark data
- `OUTDIR`
  - output directory for generated YAML files and call lists
- `MODEL`
  - model path passed later to `translate.py`
- `TRAINCONFIG`
  - path to the Mammoth training YAML
- `SUPERVISEDPAIRS`
  - file listing supervised evaluation pairs
- `ZEROSHOTPAIRS`
  - file listing zero-shot evaluation pairs
- `MAMMOTH`
  - Mammoth source directory containing `translate.py`
- `LOGDIR`
  - directory for inference stderr logs
- `SCRDIR`
  - directory for scoring outputs

The script validates that these paths exist before proceeding.

### Internal inputs

#### `TRIPLES`

A hard-coded mapping that defines:

- the internal language code
- the benchmark reference-file stem
- the available locale variants

This table is used to build the internal `LanguageInventory`.

#### `zeroshot_base`

A hard-coded list of default zero-shot pairs that are merged into the selected zero-shot pair set.

### Reads

- the training YAML from `TRAINCONFIG`
- the supervised pair list from `SUPERVISEDPAIRS`
- the zero-shot pair list from `ZEROSHOTPAIRS`
- benchmark files under `DATADIR`

### Writes

Under `OUTDIR`:

- one per-task inference YAML file named:
  - `<orig_task>.yaml`
- `calls.out`
- `calls.sacre.out`
- `calls.comet.out`

### Produces

- expanded supervised evaluation tasks
- localized zero-shot evaluation tasks
- per-task inference YAML files
- translation command list
- scoring command lists

### Main stages

#### 1. Environment validation

The script reads and validates all required environment variables and filesystem paths.

#### 2. Language inventory construction

The script builds a `LanguageInventory` from the hard-coded `TRIPLES` table.

This inventory provides:

- benchmark file stems
- localized xcodes
- reverse xcode-to-language mapping
- WMT locale mappings

#### 3. Training-task extraction

The script reads the top-level `tasks` mapping from `TRAINCONFIG`, expands language specifications into localized xcodes, and records:

- original task names
- expanded localized tasks
- task types
- vocab codes
- sharing groups
- transforms

#### 4. Pair-list loading

The script reads:

- the selected supervised pair list
- the selected zero-shot pair list

It also merges in the hard-coded `zeroshot_base` pairs and removes overlaps with the supervised set.

#### 5. Supervised-task filtering

The expanded supervised tasks are filtered so that only those whose normalized `src-tgt` pair appears in the selected supervised set are kept.

#### 6. Data-coverage checking

The script verifies that the selected task xcodes are covered by the available benchmark data inventory.

#### 7. Zero-shot task expansion

For each selected zero-shot pair, the script:

- finds compatible task families
- expands localized source and target xcodes
- resolves candidate template tasks
- derives:
  - encoder sharing group
  - decoder sharing group
  - source vocab code
  - target vocab code
  - transforms

These derived values are stored in `TaskSupport`.

#### 8. Inference YAML generation

For each expanded task, the script writes a per-task inference YAML.

The generated YAML is built by:

- deep-copying a template training configuration
- dropping training-only top-level keys
- dropping dataset-path task keys
- narrowing `tasks` to one task
- narrowing `src_vocab` and `tgt_vocab`
- setting inference-time parameters such as:
  - `beam_size`
  - `batch_size`
  - `batch_type`
  - `gpu`
  - `world_size`
  - `gpu_ranks`

The resulting YAML is intended to resemble the older `inf_extract_yaml.py` output while still using the task-resolution logic from `inf_plan.py`.

#### 9. Call-list generation

For each expanded task, the script plans translation and scoring for:

- FLORES+
- BOUQuET
- BOUQuETpar
- WMT24++

It writes:

- translation commands to `calls.out`
- SacreBLEU / chrF commands to `calls.sacre.out`
- COMET commands to `calls.comet.out`

### Output file semantics

#### `calls.out`

Contains one `python ... translate.py ...` command per required translation.

#### `calls.sacre.out`

Contains shell commands for SacreBLEU / chrF scoring.

#### `calls.comet.out`

Contains shell commands for COMET scoring.

### Per-task YAML semantics

Each generated YAML is keyed by the original training task name, because later translation calls use:

```text
--task_id "<orig_task>"






## `inf_wrapper.sh`

### Purpose

Run a distributed subset of inference commands inside an existing Slurm job.

### Intended caller

Normally launched indirectly from a Slurm batch script through `srun singularity exec`.

### Required environment

- `BINDIR`
  - directory containing `inf_wrapper.sh` and `inf_distr.sh`

### Required execution context

- must run inside a Slurm job
- expects GPU node resources to already be allocated
- expects `calls.out` to have been prepared already

### Reads

- `$BINDIR/inf_wrapper.sh`
- `$BINDIR/inf_distr.sh`
- shared venv activation script

### Writes / effects

- writes progress messages to stdout/stderr
- executes a rank-specific subset of commands from `calls.out`

### Output

The script itself does not produce files directly.
The executed commands typically create hypothesis files and logs.




## `inf_distr.sh`

### Purpose

Distribute and execute inference commands from `calls.out`, or propose a Slurm allocation when run outside Slurm.

### Calling modes

#### Outside Slurm

```bash
bash inf_distr.sh
```




