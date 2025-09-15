# Data Handling with MAMMOTH Helpers

This directory contains tools that can be used to download, stat, and
remove datasets.  Later, we may include conversions and other
preprocessing tools for datasets to this very directory.

## Set a Shared Location for the Project's Data

You need to choose where to store your datasets.  Set the environment
variable PROJCOMM to point to the shared data files.  The recommended
location on CSC machines is `/scratch/$ACCOUNT/shared`.
```
export DATACOMM=/scratch/$ACCOUNT/shared
mkdir $DATACOMM
ln -s $DATACOMM $PROJHOME/shared
ln -s ../shared $PROJHOME/base/
```

## Dataset Download, Stat and Removal

Script `datatsets.sh` is handy for planning, downloading, stating and
stop downloaded datasets.  On LUMI, you need to have `module load
lumi-tools` before using the tool.

### Usage

If you do not remember how to use it, just write `cd
$PROJHOME/base/mammoth-helper/helper/bin/gets/; ./dataset.sh` and this
will list the usage information:

```
Usage:
  getdata.sh --check <dataset>   # show quota/space, plan, cost (default if <dataset> only)
  getdata.sh --get   <dataset>   # download/extract/clone only that dataset
  getdata.sh --rm    <dataset>   # remove dataset directory (asks to confirm)
  getdata.sh --stat  <dataset>   # on-disk stats + costs
  getdata.sh --list              # list supported datasets
  getdata.sh --help              # this help
```

### Datasets

Downloadable datasets are shown with `--list`:

```
Available datasets:
  europarl-3langs     → $PROJCOMM/europarl/3langs
  europarl-all|europarl
                       → $PROJCOMM/europarl/all
  vocab-opusTC.mul     → $PROJCOMM/vocab/{opusTC.mul.64k.spm, opusTC.mul.vocab.onmt}
  tatoeba              → $PROJCOMM/tatoeba/Tatoeba-Challenge (git)
  opus100-de-en        → $PROJCOMM/opus100/de-en
  opus100-zeroshot     → $PROJCOMM/opus100/zeroshot
```

### Planning

One of the key functionalities of the tool is to check the space usage and the remaining resources of the project.
Just write `./datasets.sh --check opus100-de-en` to get the resouce analysis and action recommendation:

```
== Space & quota for lumi on /scratch/project_462000964/common ==
Filesystem                                                                             Size  Used Avail Use% Mounted on
10.253.241.4@tcp16,10.253.241.5@tcp16:10.253.241.6@tcp16,10.253.241.7@tcp16:/lustrep1  500T  435T   66T  87% /pfs/lustrep1

== lfs quota ==
Disk quotas for usr aylijyra (uid 18896):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
/scratch/project_462000964/common
                 25.84G      0k      0k       -   15255       0       0       -
uid 18896 is using default block quota setting
uid 18896 is using default file quota setting
Disk quotas for grp pepr_aylijyra (gid 8018896):
     Filesystem    used   quota   limit   grace   files   quota   limit   grace
/scratch/project_462000964/common
                 502.7M      0k      0k       -    2055       0       0       -
gid 8018896 is using default block quota setting
gid 8018896 is using default file quota setting

== plan ==
dataset: opus100-de-en  kind: parallel
target : /scratch/project_462000964/common/opus100/de-en
url    : https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-de-en-v1.0.tar.gz
size   : 64.09M  (0.06 GiB)

== storage comparison (≈30 days) ==
Size : 0.06 GiB  (0.0001 TiB)
LUMI: BU/mo=N/A (set RATE_BU_PER_TIB_MONTH_LUMI to see BU)
LUMI: TiB=0.0001  TiB·h/mo=0.04  BU/mo=N/A

== capacity recommendation (from lumi-quota) ==
  projappl free:   54.00G (need ~64.09M)
  scratch  free:  550.00T (need ~64.09M)
  flash    free:    2.20T (need ~64.09M)
✔ Recommend storing on: /scratch (enough free space).

== storage budget (TiB·hours) ==
Project TiB·h (remaining): 3050667.15
This dataset for 30 days : 0.04 TiB·h
✔ Fits in budget for ~76266678.8 month(s) at current size.
```

### Downloads

We [Helsinki NLP] have trained a SentencePiece tokenizer on OPUS
Tatoeba Challenge data with 64k vocabulary size.  To download the
SentencePiece model and the vocabulary, you just write `/datasets.sh
--get vocab-opusTC.mul`.  This gives the output:

```
[datasets] curl -fL --retry 3 -o /scratch/project_462000964/common/.tmp-datasets/opusTC.mul.64k.spm https://mammoth101.a3s.fi/opusTC.mul.64k.spm
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 1337k  100 1337k    0     0  22.7M      0 --:--:-- --:--:-- --:--:-- 22.9M
[datasets] curl -fL --retry 3 -o /scratch/project_462000964/common/.tmp-datasets/opusTC.mul.vocab.onmt https://mammoth101.a3s.fi/opusTC.mul.vocab.onmt
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  605k  100  605k    0     0  4901k      0 --:--:-- --:--:-- --:--:-- 4921k
[datasets] done (--get vocab-opusTC.mul)
```

To download all languages of the Europarl dataset, give the command
`./datasets.sh --get europarl`.  This gives the output:

```
[datasets] mkdir -p /scratch/project_462000964/common/europarl/all
[datasets] curl -fL --retry 3 -o /scratch/project_462000964/common/.tmp-datasets/europarl.tar.gz https://mammoth101.a3s.fi/europarl.tar.gz
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100 5594M  100 5594M    0     0   186M      0  0:00:29  0:00:29 --:--:--  179M
[datasets] extracting /scratch/project_462000964/common/.tmp-datasets/europarl.tar.gz -> /scratch/project_462000964/common/europarl/all
[datasets] done (--get europarl)
```

Here is the obtained directory tree:
```
/scratch/project_462000964/common
├── europarl
│   └── all
│       └── europarl
│           ├── bg-en
│           ├── ...
│           └── sv-en
│               ├── europarl-v7.sv-en.en
│               ├── europarl-v7.sv-en.sv
│               ├── train.sv-en.en.sp
│               ├── train.sv-en.sv.sp
│               ├── valid.sv-en.en.sp
│               └── valid.sv-en.sv.sp
└── vocab
    ├── opusTC.mul.64k.spm
    └── opusTC.mul.vocab.onmt
```

### Status

If you want to check if the Europarl dataset consumes too much
resources from the project, you can do `./datasets.sh --stat europarl`
to get the ouput:

```
== stats for /scratch/project_462000964/common/europarl/all ==
19G	/scratch/project_462000964/common/europarl/all
files: 120

== storage comparison (≈30 days) ==
Size : 18.86 GiB  (0.0184 TiB)
LUMI: BU/mo=N/A (set RATE_BU_PER_TIB_MONTH_LUMI to see BU)
LUMI: TiB=0.0184  TiB·h/mo=13.26  BU/mo=N/A

== capacity recommendation (from lumi-quota) ==
  projappl free:   54.00G (need ~18.86G)
  scratch  free:  550.00T (need ~18.86G)
  flash    free:    2.20T (need ~18.86G)
✔ Recommend storing on: /scratch (enough free space).

== storage budget (TiB·hours) ==
Project TiB·h (remaining): 3050667.15
Current data for 30 days : 13.26 TiB·h
✔ Fits in budget for ~230065.4 month(s) at current size.
```

### Removing Datasets

You can also remove the Europarl dataset with the command `./datasets.sh --rm europarl`.





