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