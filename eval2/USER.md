# Mammoth `eval2` user guide

`eval2` is a Make-driven workflow for evaluating trained Mammoth translation models. It prepares evaluation language pairs, generates inference configurations and command lists, submits inference and scoring work, tracks progress, and summarizes scores.

The workflow has two layers:

- **generic evaluation planning**: inspect a model, choose supervised and zero-shot pairs, generate inference YAML files, plan translation/scoring calls, and summarize results;
- **cluster execution**: turn those calls into batch jobs and run them. The current implementation of this layer is written for **LUMI and Slurm**.

This guide describes the generic workflow first. LUMI-specific details and porting notes are collected near the end.

## 1. Basic model layout

A model directory is expected to contain at least:

```text
<model>/
    train.yaml
    *.pt
```

`eval2` creates or uses these files and directories inside the model directory:

```text
<model>/
    model-summary.yaml           # model parameters gathered from the *.pt files
    mammoth.selected             # the pertinent mammoth version
    inf_supervised.txt.input     # trained language pairs as proposals for evaluation
    inf_supervised.txt           # language pairs selected for evaluation
    inf_zeroshot.txt.input       # most promising zeroshot language pairs as proposals for evaluation
    inf_zeroshot.txt             # zeroshot language pairs selected for evaluation
    testing.yaml.out             # the planner's stderr from a successful planning run
    inf_out/                     # the planner's output files
    inf_logs/                    # log files
    inf_scores/                  # score files
    *.done                       # workflow-level marker saying the stage has completed
    *.submitted                  # workflow-level marker saying the stage has been submitted
```

The generated `.input` pair files are **proposals**. The files without `.input` are the selected pairs that the evaluation actually uses.

## 2. Selecting a model

The top-level Makefile supports two ways to select the model.

### Model alias

Aliases are defined in `mk-model.mk`. List them with:

```bash
make list
```

An alias must be the **first** Make goal:

```bash
make docmt4denhalfbase mk-basic
make docmt4denhalfbase mk-calls
```

Running only the alias prints a model summary:

```bash
make docmt4denhalfbase
```

### Explicit model directory

For a model not listed in `mk-model.mk`:

```bash
make MODELDIR=/path/to/model mk-basic
make MODELDIR=/path/to/model mk-calls
```

For the current LUMI workflow, aliases are the best-tested path because job names and continuation logic also use the first goal as the model name.

## 3. Main targets

| Target | Purpose | Important effect |
|---|---|---|
| `list`                | List configured model aliases | No model changes |
| `status`              | Show status of configured models | Reports YAML/call/output counts and Slurm runtime when available |
| `<alias>`             | Inspect one configured model | Builds/prints `model-summary.yaml` if needed |
| `mt-bleu`             | Summarize MT BLEU scores | Reads `inf_scores/*.sacre` |
| `mt-chrf`             | Summarize MT chrF2 scores | Reads `inf_scores/*.sacre` |
| `mt-bleu-chrf`        | Summarize both MT metrics | Reads `inf_scores/*.sacre` |
| `mk-basic`            | Check model/resources and create evaluation directories | Selects compatible Mammoth checkout and prepares scoring environment |
| `mk-pairs`            | Prepare/validate evaluation pair selections | Stops if proposed pairs have not yet been accepted/edited |
| `mk-pairs-force`      | Accept all automatically proposed pairs | Copies `*.input` pair proposals to selected pair files |
| `mk-calls`            | Generate inference YAMLs and planned commands | Creates `inf_out/*.yaml` and `calls*.out` |
| `mk-calls-force`      | Generate calls while automatically accepting pair proposals | Convenience target for non-interactive setup |
| `mk-infer`            | Submit inference | Uses generated Slurm script and records job id |
| `mk-infer-score`      | Submit inference and a continuation job | Continuation replans and advances the pipeline |
| `mk-score`            | Submit SacreBLEU/chrF scoring | Uses `calls.sacre.out` |
| `continue-eval`       | Replan one model and run the next required stage | Inference → scoring → later stages |
| `continue-eval-force` | Same, but automatically accepts proposed pairs | Useful for unattended runs |
| `continue-all`        | Continue all model aliases not currently running | Iterates through `mk-model.mk` aliases |

There are also developer/diagnostic targets such as `status-infer`, `status-score`, `mk-slurm`, `clean-*`, `test-sif`, and `find-sif`. They are covered in the developer guide.

## 4. Recommended workflow

### Step 1: inspect the configured models

```bash
make list
make <alias>
```

The model inspection stage reads the checkpoint files and writes `model-summary.yaml`. It also chooses between the configured Mammoth implementations and writes that path to `mammoth.selected`.

### Step 2: prepare the basics

```bash
make <alias> mk-basic
```

This checks the model, `train.yaml`, benchmark-data root, required software locations, and local output directories.

### Step 3: select evaluation pairs

First generate proposals:

```bash
make <alias> mk-pairs
```

On the first run, this intentionally stops and points to:

```text
inf_supervised.txt.input
inf_zeroshot.txt.input
```

Review these proposals and create/edit:

```text
inf_supervised.txt
inf_zeroshot.txt
```

Each file contains one `src-tgt` pair per line.

If the automatically proposed pairs are exactly what you want, use:

```bash
make <alias> mk-pairs-force
```

### Step 4: generate inference configurations and calls

```bash
make <alias> mk-calls
```

This runs the planning logic and produces:

- `inf_out/*.yaml` — task-specific Mammoth inference configurations;
- `inf_out/calls.out` — translations that are still needed;
- `inf_out/calls.sacre.out` — SacreBLEU/chrF scoring commands that are still needed;
- `inf_out/calls.comet.out` — planned COMET commands;
- `testing.yaml.out` — successful planning log.

The call lists are state-aware: when expected output files already exist, planning can omit work that is already complete.

### Step 5: inspect before submission

Useful commands are:

```bash
make <alias> status-infer
make <alias> status-score
make status
```

You can also inspect the generated files directly:

```bash
less <model>/inf_out/calls.out
less <model>/inf_out/calls.sacre.out
ls <model>/inf_out/*.yaml
```

### Step 6: run inference and scoring

On the current LUMI setup:

```bash
make <alias> mk-infer-score
```

This submits inference and a dependent continuation job. When inference finishes, the continuation stage replans the workflow and can submit scoring.

For manual stage-by-stage operation:

```bash
make <alias> mk-infer
make <alias> mk-score
```

To re-evaluate what remains and advance one stage:

```bash
make <alias> continue-eval
```

For all configured aliases:

```bash
make continue-all
```

## 5. Results and status files

The important output directories are:

- `inf_out/` — generated configs, call lists, inference hypotheses and batch scripts;
- `inf_logs/` — stderr/log output from inference calls;
- `inf_scores/` — score files, especially `*.sacre`.

Common state markers include:

- `preps.done` — basic preparation completed;
- `pairs.done` — pair selection exists;
- `inference.done` — inference stage completed;
- `metrics.done` — metric stage completed;
- `*.submitted` — contains the Slurm job id for an active/submitted stage.

Do not treat a `.submitted` file as proof that a job is still alive. The workflow checks `squeue` and removes stale locks where appropriate.

## 6. Score summaries and comparisons

For a single model:

```bash
make <alias> mt-bleu
make <alias> mt-chrf
make <alias> mt-bleu-chrf
```

`compare.py` supports comparisons across several model aliases and can produce:

- medal tables;
- pairwise dominance tables;
- Condorcet summaries;
- winner heatmaps;
- delta heatmaps;
- Graphviz dominance graphs.

The current Makefile contains example comparison targets `plot-docmt` and `plot-sentmt`.

## 7. What is generic and what is LUMI-specific?

| Area | Generic idea / reusable code | Current LUMI-specific assumption | Porting action |
|---|---|---|---|
| Model selection | `MODELDIR`, `train.yaml`, checkpoint inspection | Alias paths in `mk-model.mk` point into LUMI `/scratch` | Replace alias roots or use `MODELDIR` |
| Evaluation planning | `inf_pairs.py`, `inf_plan.py`, generated YAML/call-list design | Some helper paths, especially lang2vec, are hard-coded under a LUMI project | Parameterize/install dependencies elsewhere |
| Benchmark data | `DATADIR` abstraction and benchmark file conventions | Default `DATADIR` is under `/scratch/<project>/shared/testing-shared/data` | Point `DATADIR` at equivalent prepared data |
| Mammoth code | Planner receives Mammoth source through `MAMMOTH` | `MAMMOTHDEF`/`MAMMOTH64` are LUMI `/scratch` paths | Set paths to local Mammoth checkouts |
| Model compatibility selection | Inspect model and choose a compatible Mammoth implementation | Inspection runs in a LUMI Singularity image | Replace container invocation if needed |
| Python environments | Separate inference/scoring/plot/view environments | Uses `module load cray-python` and shared `/scratch` venvs | Use local Python/venv/module scheme |
| Batch scheduling | Split independent calls across ranks; submit dependent stages | Slurm commands and variables are assumed | Port scheduler wrapper or run calls directly |
| GPU execution | Parallel independent inference calls | LUMI `small-g`/`dev-g`, 8-GPU/node assumptions | Replace partition/resource policy |
| CPU scoring | Independent metric calls can run in parallel | LUMI `small` partition sizing table | Replace partition/resource policy |
| Containers | Run inference inside a reproducible image | `/appl/local/laifs/...` SIF path and `/usr/bin/singularity` | Use local Apptainer/Singularity/container runtime |
| Filesystem binds | Make model/Mammoth project paths visible in container | Explicit `/scratch/<project>` binds | Bind/mount local filesystem paths |
| Job accounting | Separate disk project and Slurm account | `DISKPROJECT=project_...`, `JOBPROJECT=project_...` | Change/remove accounting variables |
| Status | Count generated files and inspect active jobs | `squeue` is used for runtime/job state | Replace scheduler query or disable runtime column |
| Scoring summaries | `summarize_sacre.py` is filesystem-based | Top-level targets load `cray-python` | Invoke with any suitable Python |
| Model comparison | `compare.py` and output formats are generic | Plot Python path is a shared LUMI venv | Install plotting dependencies locally |
| COMET | Call planning exists | Submission target is currently a placeholder | Implement execution independently of porting |

