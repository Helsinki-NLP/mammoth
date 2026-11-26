# Overview

**MAMMOTH** (Massively Multilingual Modular Open Translation @ Helsinki) is a toolkit for training modular, massively multilingual translation models.

## Installation

```bash
pip install mammoth-nlp
```

See the [installation guide](install) for PyTorch builds and installing from source, and the [LUMI & Roihu quickstart](CSC_quickstart) for the CSC clusters.

Take a look at the [quickstart](quickstart) to familiarize yourself with the main training workflow.

## Citation

When using MAMMOTH for research, please cite [Mickus et al. (2024)](https://aclanthology.org/2024.eacl-demo.14/):

```
@inproceedings{mickus-etal-2024-mammoth,
    title = "{MAMMOTH}: Massively Multilingual Modular Open Translation @ {H}elsinki",
    author = {Mickus, Timothee  and
      Gr{\"o}nroos, Stig-Arne  and
      Attieh, Joseph  and
      Boggia, Michele  and
      De Gibert, Ona  and
      Ji, Shaoxiong  and
      Loppi, Niki Andreas  and
      Raganato, Alessandro  and
      V{\'a}zquez, Ra{\'u}l  and
      Tiedemann, J{\"o}rg},
    editor = "Aletras, Nikolaos  and
      De Clercq, Orphee",
    booktitle = "Proceedings of the 18th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics: System Demonstrations",
    month = mar,
    year = "2024",
    address = "St. Julians, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-demo.14/",
    doi = "10.18653/v1/2024.eacl-demo.14",
    pages = "127--136"
}
```

MAMMOTH grew out of [FoTraNMT](https://github.com/Helsinki-NLP/FoTraNMT), introduced in [Boggia et al. (2023)](https://aclanthology.org/2023.nodalida-1.24):

```
@inproceedings{boggia-etal-2023-dozens,
    title = "Dozens of Translation Directions or Millions of Shared Parameters? Comparing Two Types of Multilinguality in Modular Machine Translation",
    author = {Boggia, Michele  and
      Gr{\"o}nroos, Stig-Arne  and
      Loppi, Niki  and
      Mickus, Timothee  and
      Raganato, Alessandro  and
      Tiedemann, J{\"o}rg  and
      V{\'a}zquez, Ra{\'u}l},
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.24",
    pages = "238--247"
}
```

## Acknowledgements

MAMMOTH started as a fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) ([Klein et al., 2017](https://doi.org/10.18653/v1/P17-4012)). The model and training code have since been rewritten, but parts of the data pipeline and decoding still come from it:

```
@inproceedings{opennmt,
  author    = {Guillaume Klein and
               Yoon Kim and
               Yuntian Deng and
               Jean Senellart and
               Alexander M. Rush},
  title     = {OpenNMT: Open-Source Toolkit for Neural Machine Translation},
  booktitle = {Proc. ACL},
  year      = {2017},
  url       = {https://doi.org/10.18653/v1/P17-4012},
  doi       = {10.18653/v1/P17-4012}
}
```
