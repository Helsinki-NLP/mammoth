# SBATCH Frontmatters Bundle (LUMI & Puhti)

This bundle contains ready-to-use SBATCH "frontmatter" scripts for Mammoth-NLP training. 
They **do not** include `#SBATCH` resource discovery/launch logic; they end with:
```
source "$PROJHOME/bin/slurm/sbatch-tail.sh"
```
Your `sbatch-tail.sh` handles environment, sanity checks, and the actual `srun` pattern.

## Patterns

- **Slurm pattern** (1 task/GPU): `--ntasks-per-node=4`, `--gpus-per-task=1`.  
  GPU mapping is done inside `srun` via `local-setup.sh` (e.g., `*_VISIBLE_DEVICES=$SLURM_LOCALID`).

- **Torchrun pattern** (1 task/node): `--ntasks-per-node=1`, `--gpus-per-node=4`.  
  The script sets `export PATTERN=torchrun` before sourcing your tail. Your entrypoint should invoke `torchrun`.

## Partitions & Sizes

- **LUMI**: `dev-g` (≈10–15 min sanity), `small-g` (≤4 nodes jobs), `standard-g` (long jobs / multi-node).
- **Puhti**: `gputest` (short sanity), `gpu` (regular GPU partition).

## Variables you must set

- `$ACCOUNT` — your project/account.
- `$PROJHOME` — root path where `bin/sbatch-tail.sh` lives.
- `$SYSTEM` — used by your tail scripts to pick LUMI vs Puhti behavior.

## Files

- `train-sbatch-<system>-<nodes>n<gpus>g-<time>.slurm` — Slurm pattern (1 task/GPU)
- `train-sbatch-<system>-tr-<nodes>n<gpus>g-<time>.slurm` — Torchrun pattern (1 task/node)

## Recommended setup for your experimemt

- call the experiment somthing like `2025-08-21_enfi_1n1g-30m`
    export EXP_ID=2025-08-21_enfi_1n1g-30m
    
- make directory and move to this directory
    mkdir $PROJDATA/exp/2025-08-21_enfi_1n1g-30m
    cd $PROJDATA/exp/2025-08-21_enfi_1n1g-30m

- make other directories
    mkdir -p data logs models vocab scripts
    mkdir -p tensorboard translations metrics

- copy config files
    cp $PROJHOME/bin/slurm/train-sbatch-lumi-1n1g-10m.slurm train-1n1g-2h.slurm
    # change this file: job-name, --time etc
  
    cp $PROJHOME/bin/slurm/job-setupm.sh .
    # change the EXP_ID
    
    cp $PROJHOME/bin/confs/conf.yml conf.yml.envsubst
    # adapt the config file
    envsubst < conf.yml.envsubst > conf.yml

- link vocabularies
    cd vocab
    ln -s $PROJDATA/vocab/TC/* .
    cd ..

- link data directories to data 
    cd data
    ln -s $PROJDATA/common/datadir/train .
    ln -s $PROJDATA/common/datadir/valid .
    ln -s $PROJDATA/common/datadir/test  .
    cd ..
  
- run the experiment
    sbatch train-1n1g-2h.slurm
  
# The overview:

$PROJDATA/
├─ exp/
│  ├─ ${EXP_ID}/
│  │  ├─ README.md           # lyhyt kuvaus hyperparametreista, datasta, commitista
│  │  ├─ config.yml          # mammothin konfiguraatiotiedosto
│  │  ├─ train-1n1g-2h.slurm # ajoskripti
│  │  ├─ logs/*.{out,err}    # Slurmin lokit 
│  │  ├─ data/               # linkit varsinaiseen dataan (ei kopioida isoa dataa)
│  │  │  ├─ train.en  -> /path/to/corpus/train.en
│  │  │  ├─ train.fi  -> /path/to/corpus/train.fi
│  │  │  ├─ valid.en  -> /path/to/corpus/valid.en
│  │  │  └─ valid.fi  -> /path/to/corpus/valid.fi
│  │  ├─ models/             # save_model-*.pt checkpoints
│  │  ├─ tensorboard/        # events.out.tfevents...
│  │  ├─ metrics/            # BLEU, chrF, json/csv-yhteenveto
│  │  └─ scripts/            # pienet apuskriptit tälle kokeelle
│  └─ latest -> ${EXP_ID}    # symlink tuoreimpaan kokeeseen
└─ corpora/                  # (valinn.) yhteinen raakadata, jos haluat pitää erillään

