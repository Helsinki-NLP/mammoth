# 🦣 MAMMOTH: Massively Multilingual Modular Open Translation @ Helsinki

MAMMOTH is a toolkit from Helsinki-NLP for training modular, massively multilingual translation models.

A model is built from **components**: one embedding per language, and encoder and decoder layer stacks that you choose to share between languages, keep per language, or mix (e.g. a per-language stack followed by a shared one). Each translation direction (task) picks its components through the YAML config. Tasks are spread over many GPUs and nodes, so hundreds of directions can be trained in one run. Translating one direction later loads only the components it uses, which fits on a laptop CPU.

## Getting started

```bash
pip install mammoth-nlp
```

- [Installation](https://helsinki-nlp.github.io/mammoth/install.html): PyTorch builds, installing from source, troubleshooting.
- [Quickstart](https://helsinki-nlp.github.io/mammoth/quickstart.html): train and translate with one language pair on one GPU.
- [MAMMOTH 101](https://helsinki-nlp.github.io/mammoth/examples/train_mammoth_101.html): a multilingual model on Europarl, from one GPU to several nodes.
- [Documentation site](https://helsinki-nlp.github.io/mammoth/): all guides and the option reference.

## Guides

- [Sharing schemes](https://helsinki-nlp.github.io/mammoth/examples/sharing_schemes.html): fully shared, language-specific and modular encoders and decoders.
- [LUMI & Roihu quickstart](https://helsinki-nlp.github.io/mammoth/CSC_quickstart.html): running on the CSC supercomputers.
- [Extending a trained model with a new language](https://helsinki-nlp.github.io/mammoth/examples/extending_with_a_new_language.html) and [freezing components](https://helsinki-nlp.github.io/mammoth/freezing_components.html).
- [Counting parameters](https://helsinki-nlp.github.io/mammoth/counting_parameters.html): model size in total, per task and per component.
- [Exporting to Hugging Face](https://helsinki-nlp.github.io/mammoth/exporting_to_huggingface.html): share a model on the Model Hub and run it without MAMMOTH.
- [Translation server](https://helsinki-nlp.github.io/mammoth/serving.html): serve a model locally with a web page and an OPUS-MT compatible API (`mammoth_serve`).
- [Upgrading from 0.2](https://helsinki-nlp.github.io/mammoth/upgrading.html): config changes since the x-transformers backend was replaced.

We welcome issue reports for the code and the documentation.

## Citation

If you use MAMMOTH in research, please cite [Mickus et al. (2024)](https://aclanthology.org/2024.eacl-demo.14/). The BibTeX entry is on the [documentation site](https://helsinki-nlp.github.io/mammoth/main.html#citation).

## Acknowledgements

MAMMOTH started as a fork of [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py). We thank the NVIDIA AI Technology Center Finland for their help with the multi-GPU/node implementation.
