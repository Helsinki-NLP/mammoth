# Overview


This portal provides a detailed documentation of the **MAMMOTH**: Modular Adaptable Massively Multilingual Open Translation @ Helsinki. 



## Installation

```bash
git clone https://github.com/Helsinki-NLP/mammoth.git
cd mammoth
pip3 install -e .
pip3 install sentencepiece==0.1.97 sacrebleu==2.3.1
```

Check out the [installation guide](install) to install in specific clusters.

Take a look at the [quickstart](quickstart) to familiarize yourself with the main training workflow.

## Citation

This project is based on [OpenNMT V2](https://opennmt.net).
When using OpenNMT-py for research please cite the
[OpenNMT technical report](https://doi.org/10.18653/v1/P17-4012)

We published [FoTraNMT](https://github.com/Helsinki-NLP/FoTraNMT) the ancestor of MAMMOTH, when using MAMMOTH for research, please cite [Boggia et al. (2023)](https://aclanthology.org/2023.nodalida-1.24).


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