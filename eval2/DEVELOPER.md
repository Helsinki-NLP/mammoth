# Mammoth `eval2` developer guide

This document describes the current `eval2` implementation as supplied in the archive: its components, Make targets, generated state, interfaces, and portability boundaries.

## 1. Architecture

At a high level:

```text
                  train.yaml + model checkpoints
                             |
                             v
                        mk-basic.mk
                 model inspection / setup
                             |
                             v
                        mk-pairs.mk
                  inf_pairs.py proposals
                             |
                   selected pair files
                             |
                             v
                        mk-calls.mk
                         inf_plan.py
                             |
           +-----------------+------------------+
           |                 |                  |
           v                 v                  v
       *.yaml            calls.out       calls.sacre.out
           |                 |                  |
           |                 v                  v
           |             mk-infer.mk        mk-score.mk
           |                 |                  |
           +------------ mk-slurm.mk -----------+
                             |
                  slurm_*.templ + m4
                             |
                  *.slurm + *.sbatch
                             |
                    sbatch / srun
                             |
                  slurm_wrapper.sh
                             |
                   slurm_distr.sh
                             |
                 independent commands
                             |
                  hypotheses / scores
                             |
                             v
                       mk-contr.mk
                    replan next stage
```

The key design is **replanning from filesystem state**. Rather than storing one monolithic workflow database, `inf_plan.py` regenerates command lists based on selected tasks and which expected outputs already exist. The continuation stage uses the resulting non-empty call lists to decide what should run next.

## 2. Top-level Makefile

The top-level `Makefile` is both configuration and composition root.

It defines:

- site/project settings (`DISKPROJECT`, `JOBPROJECT`, `SIF`);
- code/environment paths (`SELFDIR`, Mammoth checkouts, venvs);
- benchmark-data roots;
- model-local output paths;
- state marker paths;
- generated script/call-list paths;
- model-alias parsing rules;
- user-facing status/reporting targets;
- includes for all workflow fragments.

It includes:

```text
mk-model.mk
mk-basic.mk
mk-pairs.mk
mk-calls.mk
mk-slurm.mk
mk-infer.mk
mk-score.mk
mk-contr.mk
```

### Model-goal convention

A configured model alias is intended to be the first Make goal:

```bash
make <alias> <target>
```

The Makefile rejects aliases appearing later in the goal list and rejects multiple aliases. For non-alias use, `MODELDIR=/path/...` is supported.

This convention leaks into job naming and recursive Make calls through `FIRST_GOAL`, so developers should treat it as part of the current interface.

## 3. `mk-model.mk`

`mk-model.mk` is the current model registry.

It defines:

```make
MODEL_<alias> := /absolute/path/to/model
MODEL_ALIASES := ...
```

Responsibilities:

- provide convenient user aliases;
- give `status` and `continue-all` a finite set of models to iterate;
- provide model paths to alias goals.

Portability: **site/user configuration**, not core evaluation logic. The current file is LUMI-specific because the paths are absolute `/scratch/...` paths.

## 4. `mk-basic.mk`

### Purpose

Validate prerequisites, inspect checkpoints, choose the compatible Mammoth implementation, create model-local directories, and prepare shared Python environments.

### Main targets

| Target / rule | Role |
|---|---|
| `inspect-model-summary` | Build and display `model-summary.yaml` |
| `<MODELDIR>/model-summary.yaml` | Run `inspect-model-files.py` against non-optimizer `.pt` files |
| `<MODELDIR>/mammoth.selected` | Select `MAMMOTH64` or `MAMMOTHDEF` from inspection result |
| `mk-basic` | Ensure preparation marker and verify invocation is outside Slurm |
| `venvs` | Ensure plotting and SacreBLEU environments |
| `clean-basic` | Remove model summary and selected-Mammoth marker |

### Contract

Inputs include:

- `MODELDIR` and `TRAINCONFIG`;
- `TESTINGDIR`, `DATADIR`;
- `OUTDIR`, `LOGDIR`, `SCRDIR`;
- `INSPECT_MODEL_FILES`;
- `MAMMOTHDEF`, `MAMMOTH64`;
- `SIF`, `DISKPROJECT`.

Outputs/effects:

- `model-summary.yaml`;
- `mammoth.selected`;
- `preps.done`;
- `inf_out/`, `inf_logs/`, `inf_scores/`;
- shared score/plot venvs.

### Site coupling

This fragment contains strong LUMI coupling:

- `module load cray-python`;
- `/usr/bin/singularity`;
- `/scratch/<project>` bind mounts;
- the configured LUMI SIF.

The checkpoint-inspection *idea* is generic; its launcher is not.

## 5. `inspect-model-files.py`

This is the model/checkpoint introspection component.

Capabilities include:

- loading checkpoint/state-dict-like objects;
- inspecting tensor/module structure;
- deriving logical encoder/decoder layer counts;
- extracting head-dimension/head-count hints;
- classifying checkpoint files;
- merging per-file model configuration evidence;
- emitting compressed model summaries;
- warning about head-dimension compatibility.

The Make workflow currently consumes a textual marker in `model-summary.yaml` to choose between two Mammoth implementations. That makes the wording of the inspector output part of an implicit Make/Python interface. A more robust future interface would expose an explicit machine-readable compatibility field.

## 6. `mk-pairs.mk` and `inf_pairs.py`

### Pair-file lifecycle

The planner distinguishes proposals from selections:

```text
inf_supervised.txt.input  ->  inf_supervised.txt
inf_zeroshot.txt.input    ->  inf_zeroshot.txt
```

`inf_pairs.py` reads `train.yaml`, extracts supervised translation edges, and ranks candidate zero-shot edges using graph structure plus linguistic compatibility data.
The zero-shot ranking treats the supervised training tasks as a directed language graph. Candidate pairs are pairs that are not directly present as training tasks but are reachable through that graph.
Ranking combines graph-based and linguistic signals, including:
- reachability and shortest-path structure;
- availability of pivot languages;
- pivot hub strength;
- alternative shortest paths;
- URIEL/lang2vec-based linguistic compatibility.
Thus the ranking is intended as a planning heuristic rather than a statement that the highest-ranked zero-shot pair will necessarily obtain the best translation score.
For debugging or standalone use, `inf_pairs.py` can also be invoked directly:

```bash
python inf_pairs.py train.yaml --zs-out zeroshot_pairs.input
python inf_pairs.py train.yaml --supervised-pairs-and-quit supervised_pairs.input
python inf_pairs.py train.yaml --rank-list-only
```

`mk-pairs` requires explicit selected files. `mk-pairs-force` sets `FORCE_PAIRS=1`, causing proposals to be copied to the selected files.


### Historical implementation: `inf_top_zeroshots.py`

`inf_top_zeroshots.py` ranks candidate zero-shot evaluation pairs from a
Mammoth `train.yaml`.

It reads the supervised translation tasks defined in the training
configuration, interprets them as a directed language graph, and proposes
source-target pairs that are not directly present as training tasks but are
reachable through the graph.

The ranking combines several signals:

- graph reachability and shortest-path structure;
- pivot availability;
- pivot hub strength based on centrality;
- availability of alternative shortest paths;
- linguistic compatibility based on URIEL/lang2vec distances.

The script is intended as a planning aid for selecting promising zero-shot
evaluation pairs. The ranking is a heuristic for evaluation planning; it does
not predict which pairs will necessarily obtain the highest translation
scores.

Typical standalone use:

```bash
python inf_top_zeroshots.py train.yaml
``


### State

`pairs_input.done` marks proposal generation, while `pairs.done` records that selected pair files exist and are current.

## 7. `mk-calls.mk` and `inf_plan.py`

### `mk-calls.mk`

`mk-calls.mk` exports the runtime contract expected by `inf_plan.py`:

```text
MAMMOTH
LOGDIR
SCRDIR
MODEL
TRAINCONFIG
OUTDIR
DATADIR
ZEROSHOTPAIRS
SUPERVISEDPAIRS
```

It captures planner stderr into `testing.yaml.err`, recognizes successful completion by the marker `All stages of planning completed`, and renames the log to `testing.yaml.out` only on success.

That success string is therefore another implicit Make/Python API.

### `inf_plan.py`

`inf_plan.py` is the core evaluation planner. It:

1. validates environment and paths;
2. loads the training config;
3. builds the internal language/locale inventory;
4. derives supported supervised tasks;
5. reads the selected supervised and zero-shot pair sets;
6. verifies benchmark coverage;
7. constructs localized tasks;
8. resolves template tasks, vocabularies, sharing groups, transforms and inference settings;
9. emits per-task inference YAMLs;
10. emits current outstanding inference/scoring commands.

### Outputs

Under `OUTDIR` -- here `inf_out/`:

```text
<task>.yaml              # pair-specific yaml files
plan.out                 # records the selected lg pairs, model conf, available data, and existing results.
calls.out                # inference call list
calls.sacre.out          # sacre scoring call list
calls.comet.out          # comet scoring call list
```

The three call lists represent different stages of the same planned
evaluation:

- `calls.out` — Mammoth translation/inference commands; `calls.out` is validated so executable lines must start with `python`.
- `calls.sacre.out` — SacreBLEU/chrF scoring commands;
- `calls.comet.out` — COMET scoring commands.

The planner regenerates these lists from the selected tasks and current
filesystem state, so they should be treated as derived workflow state rather
than manually maintained job lists.


### Generated YAMLs

Inference YAMLs are derived from training tasks, while training-only top-level keys are dropped and inference-time values are inserted. Zero-shot tasks inherit an appropriate real training task as a template and include internal provenance fields such as pair type and template/original task identifiers.

### Portability

The planner is mostly filesystem/configuration based and is one of the most reusable components. Remaining scheduler coupling includes log filenames containing `${SLURM_JOB_ID}`. That should become a generic run/job-id abstraction if non-Slurm execution is a goal.

## 8. `mk-slurm.mk`

This fragment converts planned call lists into executable Slurm artifacts.

### Main outputs

```text
inf_out/inf.slurm
inf_out/cnt.slurm
inf_out/met.slurm
inf_out/inf.sbatch
inf_out/met.sbatch
```

The `.slurm` scripts are produced from `slurm_*.templ` with `m4`. Running the generated script outside Slurm invokes the distributor in planning mode, which computes and writes a concrete `sbatch` command to the corresponding `.sbatch` file.

### Lock targets

`clean-inf-lock`, `clean-cnt-lock`, and `clean-met-lock` inspect job-id marker files and use `squeue` to distinguish active jobs from stale markers.

These are Slurm-specific by design.

## 9. Slurm templates

### `slurm_inf.templ`

Inference template. It embeds account/project paths, job metadata and model paths. Inside a Slurm allocation it launches the wrapper through:

```text
srun /usr/bin/singularity exec ... slurm_wrapper.sh
```

It explicitly binds project `/scratch` trees into the container.

### `slurm_met.templ`

Scoring template. It uses the shared scoring activation and runs the same wrapper/distributor pattern. If invoked outside Slurm, it delegates to the distributor to plan the submission command.

### `slurm_cnt.templ`

Continuation/control template. It is responsible for returning to the Make workflow after a previous stage and advancing the pipeline.

### Portability

The template mechanism itself is generic; the current template contents are LUMI/Slurm-specific.

## 10. `slurm_distr.sh`

`slurm_distr.sh` has two operating modes.

### Outside Slurm: resource planning

It:

- reads and shuffles commands from the call file;
- counts them;
- chooses an allocation from an internal table;
- computes waves and estimated runtime;
- writes a concrete `sbatch` command.

There are separate hard-coded planning tables for LUMI partitions/topologies, including:

- `dev-g`;
- `small-g`;
- `small`.

The tables encode LUMI-specific GPU/node counts and walltime policy and must not be treated as portable performance models.

### Inside Slurm: command distribution

It uses:

```text
SLURM_PROCID
SLURM_NTASKS
SLURM_LOCALID
```

to assign call `i` to ranks by striding through the shuffled call list. Each rank executes its assigned subset sequentially.

The **strided independent-command distribution algorithm** is generic. The environment-variable interface and launcher are Slurm-specific.

## 11. `slurm_wrapper.sh`

The wrapper is intentionally thin. It:

1. verifies it is inside Slurm;
2. validates helper/call/activation paths;
3. activates the supplied environment;
4. invokes `slurm_distr.sh`;
5. records start/end timestamps.

A future scheduler-neutral design could retain this wrapper contract while injecting a generic rank/world interface.

## 12. `mk-infer.mk`

### Targets

| Target | Purpose |
|---|---|
| `status-infer` | Validate generated inference calls/scripts and show an active job if present |
| `infer` | Execute the generated `inf.sbatch`, extract the submitted job id, write `inference.submitted` |
| `infer-score` | Submit inference then a dependent continuation job |
| `mk-infer` | User alias for `infer` |
| `mk-infer-force` | Recursive force path with pair proposals accepted |
| `mk-infer-score` | User alias for `infer-score` |
| `mk-infer-score-force` | Force version of the combined target |

The job-id parsing assumes the submission command prints a job id as its final whitespace-separated field.

## 13. `mk-score.mk`

Scoring mirrors inference.

| Target | Purpose |
|---|---|
| `status-score` | Validate SacreBLEU calls and generated metric/continuation scripts |
| `score` | Submit scoring, record `metrics.submitted`, submit continuation dependency |
| `mk-score` | User alias for `score` |
| `mk-score-force` | Force path with pair proposals accepted |
| `mk-comet` | Placeholder; COMET execution is not implemented |

`calls.sacre.out` is treated as the implemented metric backend. `calls.comet.out` is generated but has no corresponding execution implementation in this archive.

## 14. `mk-contr.mk`

Continuation is a state machine implemented by **replanning**.

`continue-eval`:

1. cleans stale stage locks;
2. removes the previous planner success/error log;
3. reruns `mk-calls`;
4. counts outstanding inference, SacreBLEU and COMET calls;
5. chooses the first outstanding stage;
6. otherwise attempts visualization/completion.

The priority is currently:

```text
inference -> Sacre scoring -> COMET -> visualization -> complete
```

`continue-eval-force` uses `mk-calls-force`.

`continue-all` and `continue-all-force` iterate over `MODEL_ALIASES`, skip models with active inference/metric jobs, and continue the rest.

### Incomplete edge

The continuation code refers to `mk-viz` and `viz.done`, but no `mk-viz` implementation is included in the supplied archive. Treat visualization as an unfinished extension point.

## 15. Status and reporting

### `status.py`

The top-level `status` target invokes `status.py` for configured model aliases. It reports counts/state including generated YAMLs, calls, hypotheses, zero-shot hypotheses, Sacre outputs, planned resource information, and current Slurm runtime.

Filesystem counting is generic; current-job runtime uses `squeue`.

### `summarize_sacre.py`

Parses `*.sacre` files and renders score tables/matrices for BLEU and chrF2. This is broadly portable Python code, assuming the score filename conventions remain stable.

### `compare.py`

Loads model paths from `mk-model.mk`, collects scores, and writes multi-model comparison artifacts such as:

- medal tables;
- dominance matrices/text;
- Condorcet information;
- Graphviz graphs;
- winner/delta heatmaps.

The comparison logic is generic; model registry and plotting Python path are site configuration.

### `common_files.py`

Utility for comparing file sets across directories, with recursive and filtering options. It is independent of LUMI.

## 17. State and dependency model

The workflow uses Make timestamps plus explicit marker files.

### Slurn scipts and job records:

```text
inf.slurm                # Slurm script for inferences 
cnt.slurm                # Slurm script for continuation
met.slurm                # Slurm script for scoring
inf.sbatch               # record of the submitted Slurm job
met.sbatch               # record of the submitted Slurm job
```

### Preparation/selection markers

```text
preps.done
pairs_input.done
pairs.done
```

### Stage completion markers

```text
inference.done
cnt.done
metrics.done
comet.done
viz.done
```

### Active/submitted markers

```text
inference.submitted
cnt.submitted
metrics.submitted
comet.submitted
viz.submitted
```

The submitted markers contain scheduler job ids. Lock-cleaning targets compare them against `squeue`.

### Planner locks

The top-level Makefile also defines planning-lock paths (`inference-planning.lock`, `metrics-planning.lock`). Developers should verify actual usage before relying on them; they are configuration/state vocabulary but are not prominent in the supplied fragment rules.

## 21. Current inconsistencies / maintenance notes

These are worth resolving before treating the interface as stable:

- The top-level help text mentions older names such as `mk-shared` and `mk-yamls`, while the supplied workflow uses `mk-basic`, `mk-pairs`, and `mk-calls`.
- `continue-eval` references `mk-viz` although no visualization Make fragment/target is supplied.
- `mk-comet` is a placeholder even though COMET calls are planned.
- Some comments mention former script names (`inf_distr.sh`, `inf_wrapper.sh`) while the supplied files are `slurm_distr.sh` and `slurm_wrapper.sh`.
- Several Make/Python interfaces depend on matching human-readable strings (model compatibility text; planner completion text). These should become explicit machine-readable outputs.
- Site paths appear in both Make and Python, so changing only the top-level Makefile is insufficient for a full port.

