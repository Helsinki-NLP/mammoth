# 🦣 MAMMOTH: Massively Multilingual Modular Open Translation @ Helsinki
This repository contains the code for 🦣 MAMMOTH, the modular translation toolkit from Helsinki-NLP.

This library is built on top of OpenNMT-py.
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation framework. It is designed to be research friendly to try out new ideas in translation, summary, morphology, and many other domains. Some companies have proven the code to be production ready.

### Documentation
The original ONMT-py documentation is available [here](https://opennmt.net/OpenNMT-py/).  
Our own modifications (currently-under-development) are documented [here](https://helsinki-nlp.github.io/mammoth/).

FINAL NOTE: We would appreciate it A LOT if you report issues with this repo and/or the documentation 

### Acknowledgements
We thank the NVIDIA AI Technology Center Finland for their help with the multi-gpu/node implementation.

### Quick Installation:

Follow the instructions on [this page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/conf).
This will create the virtual environment and a directory tree containing the MAMMOTH software.  The directory structure of your project home will look like this:
```
.
├── base
│   ├── git -> /project/project_462000964/members/aylijyra/git
│   ├── mammoth-helper -> git/mammoth-helper
│   ├── mammoth-hf -> git/mammoth-hf
│   └── venv -> /project/project_462000964/members/aylijyra/venv
├── data -> /scratch/project_462000964/members/aylijyra
├── git
│   ├── mammoth
│   ├── mammoth-helper
│   └── mammoth-hf
└── venv
    └── mammoth-hf
```

### Running Training Jobs

The helper features of MAMMOTH are new and meant for guarded,
sanity-checked and optimized deployment of supercomputers for MAMMOTH,
as well as cumulative creation of advancing job configurations
(directories, slurm scripts, datafiles, and yaml files).

That means that we are currently testing scripts that would facilitate
the creation of the job-specific local environment (directory trees
and configurations) for your training (and other) jobs.  According to
the current planning, the structure of a prototypical job will look
like this:

```
$PROJHOME/data/example-job
├── base -> /project/$PROJECT/members/$USER/base
│   ├── git -> /project/project_462000964/members/aylijyra/git
│   ├── mammoth-helper -> git/mammoth-helper
│   ├── mammoth-hf -> git/mammoth-hf
│   └── venv -> /project/project_462000964/members/aylijyra/venv
├── cfg
│   ├── conf.yml
│   └── sbatch-entry.slurm
├── in
│   ├── data
│   ├── models
│   └── vocab
├── log
│   ├── slurm
│   └── tb
└── out
    ├── checkpoints
    ├── metrics
    ├── models
    └── translations
```

The slurm launch helpers are available, but they are under testing.
Follow the instructions on [this
page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/slurm)
to use the readily made execution scripts and to create shorter and
safer slurm scripts with the helper feature.

### Translation, Evaluation and Efficient Inference

We are working on extending the domain of the helper features.


