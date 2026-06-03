# Mammoth eval tools

This directory contains helper scripts for planning, running, measuring, and summarizing machine translation evaluation jobs with Mammoth on LUMI. The tools cover the shell-based evaluation workflow from inference planning to score summarization.
:contentReference[oaicite:0]{index=0}
:contentReference[oaicite:1]{index=1}
:contentReference[oaicite:2]{index=2}
:contentReference[oaicite:3]{index=3}

## Scope

The tools in this directory cover four main stages:

1. **Inference planning**
   - prepare task-specific inference YAML files
   - prepare command lists such as `calls.out`, `calls.sacre.out`, and `calls.comet.out`

2. **Inference execution**
   - distribute planned inference commands over multiple Slurm ranks and GPUs
   - execute translation jobs inside a containerized runtime

3. **Metric computation**
   - run SacreBLEU / ChrF scoring
   - run COMET scoring
   - store metric outputs in the configured score directory

4. **Result summarization**
   - read `.sacre` score files
   - print dataset-specific ASCII tables or matrices for inspection

Planning scripts are still under active development. The execution-side tools are intended to remain relatively stable.

## Short evaluation cycle

A typical evaluation cycle is:

1. prepare inference YAML files and command lists
2. generate `calls.out`, `calls.sacre.out`, and `calls.comet.out`
3. inspect the proposed Slurm allocation if needed
4. submit a Slurm batch job
5. run `inf_wrapper.sh` inside the allocation
6. let `inf_distr.sh` distribute inference calls across ranks
7. produce hypothesis files
8. run metric commands from `calls.sacre.out` and `calls.comet.out`
9. summarize the resulting `.sacre` files with `summarize_sacre.py`

## Main scripts by role

### `inf_make_template.sh`

`inf_make_template.sh` is the top-level shell/Slurm entry script for
the shell-based inference workflow. It defines model-specific paths,
prepares output directories, runs the inference planner, and either
stops after planning or proceeds to distributed inference under
Slurm. It is intended to be copied or adapted per
model. :contentReference[oaicite:4]{index=4}

Typical use:

```bash
bash inf_make_template.sh
bash inf_make_template.sh --fresh
```



Fresh planning:

```bash
bash inf_make_template.sh --fresh
```

Batch execution:

```bash
sbatch [resource options] inf_make_template.sh
```

### `inf_wrapper.sh`

`inf_wrapper.sh` is a Slurm-only task wrapper for distributed inference execution.

It is intended to run inside an existing Slurm allocation on a GPU node. The script checks that the required helper scripts exist, activates the shared Python virtual environment, and then delegates command distribution and execution to `inf_distr.sh`.

Typical use is indirect, via `srun singularity exec ... inf_wrapper.sh`.

### `inf_distr.sh`

`inf_distr.sh` is a dual-mode command distributor.

- Outside Slurm, it reads `calls.out`, counts the planned inference calls, and prints a recommended LUMI allocation together with an example `sbatch` command.
- Inside Slurm, it assigns commands to ranks and executes them so that each rank runs every `world_size`-th command from the shared calls file.

Typical use:

```bash
bash inf_distr.sh [CALLS_FILE]
```

If no calls file is given, the script defaults to `$OUTDIR/calls.out`.

### `inf_extract_yaml.py`

`inf_extract_yaml.py` converts a multi-task Mammoth training configuration into a directory of single-task inference YAML files.

For each task in the training config, it:

- removes training-only top-level settings,
- removes training and validation path fields from the task block,
- keeps only one task in the `tasks` section,
- keeps only the relevant `src_vocab` and `tgt_vocab` entries,
- writes the result as `<task_name>.yaml` in the chosen output directory.

Typical use:

```bash
python inf_extract_yaml.py train.yaml out_dir
```

### `inf_plan.sh`

`inf_plan.sh` is a preparation script that builds the command files used later in the evaluation cycle.

It performs the following steps:

1. validates the required environment variables and directories,
2. builds a locale-aware language inventory from an internal triples table,
3. extracts Mammoth task names from `TRAINCONFIG`,
4. expands base-language tasks into locale-aware task variants where needed,
5. checks that the resulting tasks are covered by known benchmark data,
6. generates one inference YAML per original training task using `inf_extract_yaml.py`,
7. writes:
   - `calls.out`,
   - `calls.sacre.out`,
   - `calls.comet.out`.

Typical use is indirect, via the surrounding Makefile or shell workflow.

Main outputs:

- `OUTDIR/calls.out`
  - inference commands for Mammoth `translate.py`,
- `OUTDIR/calls.sacre.out`
  - SacreBLEU / chrF scoring commands,
- `OUTDIR/calls.comet.out`
  - COMET scoring commands,
- `OUTDIR/*.yaml`
  - per-task inference YAML files used by the planned translation calls.

### `met_make.sh`

`met_make.sh` is the top-level shell entry script for the SacreBLEU / chrF metrics phase.

It is used after hypotheses have already been generated. The script:

1. ensures that a Python virtual environment for `sacrebleu` exists,
2. ensures that `calls.sacre.out` exists, regenerating it through the inference-planning workflow if needed,
3. prevents duplicate submission of the same metrics job,
4. submits itself as a Slurm job when run outside Slurm,
5. executes the metric commands from `calls.sacre.out` when run inside Slurm.

Typical use:

```bash
bash met_make.sh
```

This script belongs to the post-inference measurement phase.

### `summarize_sacre.py`

`summarize_sacre.py` is a reporting tool for `.sacre` score files.

It scans a directory of score files, parses the metric payload from each file, groups results by dataset and language pair, and prints human-readable ASCII summaries.

Supported datasets:

- WMT24++,
- Flores+,
- Bouquet.

Supported metrics:

- BLEU,
- chrF2,
- TER.

Typical use:

```bash
python summarize_sacre.py inf_scores --kind mt
```

Single metric only:

```bash
python summarize_sacre.py inf_scores --kind docmt --metric BLEU
```

Wide table output:

```bash
python summarize_sacre.py inf_scores --kind sentmt --wide
```


## Documentation layout

- `README.md` gives the high-level workflow and intended usage.
- `INTERFACES.md` documents script interfaces in detail:
  - environment variables
  - inputs
  - outputs
  - side effects
  - expected calling conventions

