# 🦣 MAMMOTH: Massively Multilingual Modular Open Translation @ Helsinki

This repository contains the code for 🦣 MAMMOTH, the modular translation toolkit from Helsinki-NLP.

This library is built on top of OpenNMT-py.
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation framework. It is designed to be research friendly to try out new ideas in translation, summary, morphology, and many other domains. Some companies have proven the code to be production ready.

### Documentation

The original ONMT-py documentation is available [here](https://opennmt.net/OpenNMT-py/).  
The documentation to Mammoth (currently under development) is available [here](https://helsinki-nlp.github.io/mammoth/)

**Note:** We would greatly appreciate issue reports for this repository and its documentation.

### Acknowledgements

We thank the NVIDIA AI Technology Center Finland for their help with the multi-gpu/node implementation.


For working on LUMI/Roihu supercomputer environments, see the [LUMI/Roihu quickstart guide](csc_env/README.md).

We recommend to use the Hugging Face Tokenizer for training and inferencing in Mammoth, see the [HF Tokenizers guide](docs/HF_TOKENIZERS.md).

To export the Mammoth pretrained model weights to Hugging Face Model Hub, see the [Exporting Mammoth to Model Hub guide](docs/exporting_to_huggingface.md).

Please find the template config files for single & multi-node training inference and profiling under `csc_env/`.