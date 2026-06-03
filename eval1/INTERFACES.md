# Interfaces for Mammoth evaluation tools

This document specifies the current runtime interfaces of the evaluation scripts.

---

## `inf_wrapper.sh`

### Purpose

Run a distributed subset of inference commands inside an existing Slurm allocation.

### Intended caller

Normally launched indirectly from a Slurm batch script through `srun singularity exec`.

### Required environment variables

- `BINDIR`
  - directory containing `inf_wrapper.sh` and `inf_distr.sh`

### Required execution context

- must run inside a Slurm job
- expects GPU node resources to already be allocated
- expects container runtime to provide Python and GPU libraries

### Reads

- `$BINDIR/inf_wrapper.sh`
- `$BINDIR/inf_distr.sh`
- shared venv activation script:
  - `/scratch/project_462000964/shared/mammoth-shared/.venv/bin/activate`

### Writes

- no files directly
- writes status and timing messages to stdout/stderr

### Produces

- indirect effects only: executes part of the planned commands from `calls.out`

---

## `inf_distr.sh`

### Purpose

Distribute and execute commands from `calls.out`, or propose a Slurm allocation when run outside Slurm.

### Calling modes

#### Outside Slurm

Prints a recommended resource allocation and an example `sbatch` command.

Typical call:

```bash
bash inf_distr.sh
```

## `inf_extract_yaml.py`

### Purpose

Create one single-task inference YAML per Mammoth task from a multi-task training YAML.

### Intended use

This script is a preparation tool. It is typically run before inference planning or job submission when the workflow expects a directory of per-task YAML files.

### Calling convention

```bash
python inf_extract_yaml.py TRAIN_YAML OUT_DIR [options]
```

---

## `inf_plan.sh`

### Purpose

Plan the inference and metric-computation command files for benchmark evaluation.

### Intended use

This script is a planning-stage tool. It is normally called by a higher-level workflow such as `inf_make.sh` or a Make target, rather than manually.

### Required environment variables

- `DATADIR`
  - root directory of benchmark data

- `OUTDIR`
  - output directory for generated YAML files and command lists

- `MAMMOTH`
  - directory containing Mammoth runtime scripts such as `translate.py`

- `MODEL`
  - model checkpoint or model directory used for inference

- `TRAINCONFIG`
  - Mammoth multi-task training YAML

- `BINDIR`
  - directory containing helper scripts, especially `inf_extract_yaml.py`

- `LOGDIR`
  - directory for translation stderr logs

- `SCRDIR`
  - directory for metric output files

### Reads

- `TRAINCONFIG`
- benchmark files under `DATADIR`
- `BINDIR/inf_extract_yaml.py`

### Writes

- `OUTDIR/calls.out`
- `OUTDIR/calls.sacre.out`
- `OUTDIR/calls.comet.out`
- `OUTDIR/<task>.yaml`
- `OUTDIR/extraction_complete`

### Produces

- one translation command per required inference output
- one or more metric commands for outputs that are ready to score
- per-task inference YAML files derived from the training config

### Internal stages

#### 1. Environment validation

The script aborts if required variables are unset or if required files/directories do not exist.

#### 2. Language inventory construction

The script builds locale-aware lookup tables from the internal `triples` array.

This includes mappings such as:

- base language code -> benchmark file stem
- base language code -> expanded locale-aware sides
- expanded side -> base language code
- expanded side -> WMT locale code

#### 3. Task extraction and expansion

The script reads `tasks.*` from `TRAINCONFIG`, keeps tasks of type:

- `mt_*`
- `sentmt_*`
- `docmt_*`

and expands base language task names to locale-aware task variants when needed.

Examples:

- `mt_fra-eng` may expand to:
  - `mt_CA.fra-XX.eng`
  - `mt_FR.fra-XX.eng`

- `mt_bul-eng` may expand to:
  - `mt_XX.bul-XX.eng`

#### 4. Data coverage check

The script verifies that the expanded task sides are covered by the known inventory.
If uncovered tasks remain, it aborts.

#### 5. YAML extraction

If `OUTDIR/extraction_complete` does not exist, the script runs:

```bash
python "$BINDIR/inf_extract_yaml.py" "$TRAINCONFIG" "$OUTDIR"
```


---

## INTERFACES.md section

```md id="l8ohvf"
---

## `inf_make_template.sh`

### Purpose

Provide a top-level Slurm batch script template for running the shell-based Mammoth evaluation workflow.

### Intended use

This script is usually edited per model or experiment. It is the user-facing entry script for the legacy shell-based evaluation flow.

### Main responsibilities

- define model-specific and environment-specific paths
- create output directories
- invoke `inf_plan.sh`
- invoke `inf_distr.sh` in planning mode when outside Slurm
- invoke `inf_wrapper.sh` through `srun singularity exec` when inside Slurm

### Calling modes

#### Outside Slurm

```bash
bash inf_make_template.sh
```



---

## INTERFACES.md section for `met_make.sh`

```md
---

## `met_make.sh`

### Purpose

Run the SacreBLEU / ChrF measurement phase for already generated hypothesis files.

### Intended use

This script is the top-level shell entry point for the
metric-computation stage of the legacy shell-based evaluation
workflow.

### Calling modes

#### Outside Slurm

```bash
bash met_make.sh
```

---


## INTERFACES.md section for `summarize_sacre.py`

```md id="jlwm73"
---

## `summarize_sacre.py`

### Purpose

Summarize `.sacre` score files into human-readable ASCII matrices and tables.

### Intended use

This script is a reporting-stage tool. It is used after metric computation has already produced `.sacre` files.

### Calling convention

```bash
python summarize_sacre.py INDIR --kind {mt|sentmt|docmt} [options]

---

