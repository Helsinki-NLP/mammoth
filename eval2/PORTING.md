## Porting 

### Same LUMI, different project

Audit every occurrence of:

```text
project_462...
/scratch/...
/appl/local/laifs/...
```

Then check:

- `DISKPROJECT` vs `JOBPROJECT` semantics;
- Singularity bind mounts;
- model aliases;
- Mammoth checkouts;
- shared venv/data locations;
- lang2vec path;
- Slurm account.

### Another Slurm cluster

In addition to paths:

- replace module/environment setup;
- replace container runtime/image and binds;
- rewrite partition names;
- rewrite allocation lookup tables;
- validate Slurm directives and GPU semantics;
- validate the `srun` launch layout;
- check job-id output parsing and `squeue` formats.

### Non-Slurm backend

Recommended refactoring boundary:

```text
planner -> call lists -> execution backend -> outputs -> planner
```

Keep `inf_plan.py` and call-list conventions. Introduce a backend interface that provides:

- submit/launch;
- rank/world distribution or local parallelism;
- active-job query;
- dependency/continuation scheduling;
- job/run identifier for logs.

Then reimplement the current Slurm-specific Make fragments behind that interface.

## 20. Suggested refactoring for portability

The current code can become substantially easier to port by moving site values out of the top-level Makefile and Python scripts.

A possible configuration split is:

```text
config/
    site.mk          # filesystem, account, scheduler, container, modules
    models.mk        # model aliases only
    datasets.mk      # benchmark root/layout
```

High-value variables to externalize include:

```text
DISKPROJECT
JOBPROJECT
SELFDIR
SIF
MAMMOTHDEF
MAMMOTH64
VENV
TESTINGDIR
DATADIR
L2V_REPO
GPU_PARTITION
CPU_PARTITION
DEV_GPU_PARTITION
CONTAINER_RUNTIME
```

For stronger portability, make the allocation planner data-driven rather than embedding LUMI tables in `slurm_distr.sh`.

### A. Moving to another LUMI project

This is the smallest port. Review at least:

```make
DISKPROJECT := ...
JOBPROJECT  := ...
SELFDIR     := ...
INSPECT_MODEL_FILES := ...
MAMMOTHDEF  := ...
MAMMOTH64   := ...
VENV        := ...
TESTINGDIR  := ...
DATADIR     := ...
```

Also update `mk-model.mk`, because its model roots are absolute `/scratch/...` paths.

Check the selected SIF image and the project bind mounts in `mk-basic.mk` and `slurm_inf.templ`. A change of project id is not complete until both **host paths** and **container binds** have been updated.

### B. Moving within the same LUMI project

If only the checkout/model location changes, usually update:

- `SELFDIR`;
- `INSPECT_MODEL_FILES`;
- model aliases in `mk-model.mk`;
- Mammoth checkout paths if they moved;
- shared venv/testing paths if they moved.

The LUMI partitions, account, container runtime, and allocation tables may remain usable.

### C. Moving to another Slurm cluster

Keep the planning layer, but audit the execution layer carefully:

1. Replace project/account names and filesystem paths.
2. Replace module commands (`module load cray-python`) with the target environment setup.
3. Replace the LUMI SIF path and container bind mounts.
4. Update `small-g`, `dev-g`, and `small` partition names.
5. Rewrite the resource tables in `slurm_distr.sh` for the new node/GPU topology and walltime limits.
6. Check Slurm options in `slurm_*.templ` and generated `sbatch` commands.
7. Confirm that `SLURM_PROCID`, `SLURM_NTASKS`, `SLURM_LOCALID`, and `SLURM_JOB_ID` have the expected meanings on the target cluster.
8. Test with one model and a very small call list before enabling `continue-all`.

### D. Moving to a non-Slurm machine

The planning and reporting pieces can still be reused, but the current high-level execution targets are not scheduler-neutral.

A practical port is:

1. run `mk-basic`, pair selection, and `mk-calls` after replacing LUMI-only setup commands;
2. execute lines from `inf_out/calls.out` locally or through another launcher;
3. rerun `mk-calls` to discover remaining scoring work;
4. execute `calls.sacre.out`;
5. use `summarize_sacre.py` / `compare.py` for reporting.

For a permanent non-Slurm port, replace `mk-slurm.mk`, the three `slurm_*.templ` files, `slurm_distr.sh`, `slurm_wrapper.sh`, and scheduler-dependent status/continuation checks with a scheduler-independent execution backend.

## 9. Current limitations worth knowing

The uploaded version is still under development. In particular:

- COMET call planning exists, but `mk-comet` currently reports that COMET evaluation is not implemented.
- `continue-eval` refers to a later visualization stage (`mk-viz` / `viz.done`) that is not defined in the supplied files.
- Some help text and older comments still use earlier target names such as `mk-yamls` or older script names. Prefer the targets documented above, which are present in the current Make fragments.
- Several paths are hard-coded and should eventually move into a site configuration file if this workflow is intended to be portable.
- `inf_pairs.py` currently contains a hard-coded lang2vec repository path under a LUMI project, and its setup comments assume `cray-python` and LUMI `/scratch`.
  For portability, make the lang2vec source/dependency configurable or use an installed package.

## 18. Portability matrix

| Component | Generic core | Slurm-specific | LUMI-specific |
|---|:---:|:---:|:---:|
| `inspect-model-files.py` analysis | ✓ |  |  |
| `mk-basic.mk` inspection launcher |  | partly | ✓ |
| `mk-model.mk` alias mechanism | ✓ |  | current paths ✓ |
| `inf_pairs.py` graph/ranking logic | ✓ |  | lang2vec path ✓ |
| `mk-pairs.mk` selection workflow | ✓ |  | Python module setup ✓ |
| `inf_plan.py` task/YAML planning | mostly ✓ | log job-id naming partly |  |
| `mk-calls.mk` orchestration | mostly ✓ |  | module command ✓ |
| generated `calls.out` concept | ✓ |  |  |
| generated scoring calls | ✓ |  |  |
| `mk-slurm.mk` |  | ✓ | resource parameters partly ✓ |
| `slurm_*.templ` |  | ✓ | partitions/account/binds/container ✓ |
| `slurm_distr.sh` strided distribution | ✓ | interface ✓ | allocation tables ✓ |
| `slurm_wrapper.sh` | concept ✓ | implementation ✓ |  |
| `mk-infer.mk` |  | ✓ |  |
| `mk-score.mk` |  | ✓ |  |
| `mk-contr.mk` replanning logic | ✓ | current locks/dependencies ✓ |  |
| `status.py` filesystem status | ✓ | live-job query ✓ |  |
| `summarize_sacre.py` | ✓ |  |  |
| `compare.py` | ✓ |  | configured paths/venv ✓ |
| dataset download scripts | ✓ |  | default local layout may be site-specific |


