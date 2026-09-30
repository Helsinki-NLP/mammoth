# 🦣 MAMMOTH: Massively Multilingual Modular Open Translation @ Helsinki

This repository contains the code for 🦣 MAMMOTH, the modular translation toolkit from Helsinki-NLP.

This library is built on top of OpenNMT-py.
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation framework. It is designed to be research friendly to try out new ideas in translation, summary, morphology, and many other domains. Some companies have proven the code to be production ready.

## Getting started

```bash
pip install mammoth-nlp
```

- [Installation guide](docs/source/install.md): install locally or on a specific cluster.
- [Quickstart](docs/source/quickstart.md): run a first training with synthetic toy data and a small translation model.
- [Documentation site](https://helsinki-nlp.github.io/mammoth/): the full Mammoth documentation (currently under development).
  The original ONMT-py documentation is available [here](https://opennmt.net/OpenNMT-py/).

## Guides
- [LUMI/Roihu quickstart guide](docs/source/CSC_quickstart.md): working on the LUMI/Roihu supercomputer environments.
- [Exporting Mammoth to Model Hub guide](docs/source/exporting_to_huggingface.md): export pretrained Mammoth model weights to the Hugging Face Model Hub.
- [HF Tokenizers guide](docs/source/HF_TOKENIZERS.md): we recommend the Hugging Face Tokenizer for training and inferencing in Mammoth.
- [Indexed dataset guide](docs/source/INDEXED_DATASET_GUIDE.md): tokenize once up front to remove tokenization cost during training.

**Note:** We would greatly appreciate issue reports for this repository and its documentation.

## Acknowledgements

We thank the NVIDIA AI Technology Center Finland for their help with the multi-gpu/node implementation.
