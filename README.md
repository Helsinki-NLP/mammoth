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

## Quick Start: HuggingFace Integration

This guide shows how to quickly convert and use HuggingFace BART models with Mammoth.


### Setup

```bash
# Copy repository from feat/hf_integration branch
git clone -b feat/hf_integration https://github.com/Helsinki-NLP/mammoth.git

# cd into the directory
cd mammoth

# Install the dependencies
pip install -r requirements.txt
```


### HF Converter

#### Basic Usage

Convert a HuggingFace BART model to Mammoth format:

```bash
# Basic conversion from Hugging Face model hub (language pair needs to be specified)
python hf_converter.py vgaraujov/bart-base-translation-en-es ./models/en_es_model --src-lang en --tgt-lang es

# Local model conversion (language pair needs to be specified)
python hf_converter.py /path/to/local/bart/model ./models/converted_model --src-lang en --tgt-lang es
```

Note:
1. "my_bart_model" will be the prefix of the model (components) name and will not impact the inference later.
2. BART architecture is originally used in many tasks, but the current setup has been only tuned for and tested on translation tasks (HF model head: BartForConditionalGeneration). Different model heads have different weights hence might lead to unexpected results.
3. The HF BART model used in debugging is [vgaraujov/bart-base-translation-en-es](https://huggingface.co/vgaraujov/bart-base-translation-en-es).

#### Command Line Options

```bash
python hf_converter.py --help
```

**Required Arguments:**
- `hf_model_path`: Path to HuggingFace model (local path or HF model name)
- `save_path`: Where to save the converted Mammoth model

**Optional Arguments:**

- `--src-lang`: Source language code (default: `en`)
- `--tgt-lang`: Target language code (default: `es`)


#### What the Converter Does

1. **Downloads/Loads** the HF BART model and tokenizer
2. **Creates** an x-transformers model with matching architecture
3. **Maps weights** from HF format to x-transformers format
4. **Builds** a Mammoth model structure with task configuration
5. **Transfers weights** from x-transformers to Mammoth
6. **Saves** the complete Mammoth checkpoint

#### Output Files

After conversion, you'll find:
- `{save_path}/` - Main Mammoth model checkpoint 
- `src_vocab_{src_lang}.txt` - Vocab file of source language
- `tgt_vocab_{tgt_lang}.txt` - Vocab file of target language 
- `xt_model_keys.txt` - x-transformers model layer names (debug)
- `mammoth_model_keys.txt` - Mammoth model layer names (debug)

Note:

Mammoth models are saved in components so do not panic.

#### Supported Models

✅ **Tested Models:** (batch size: 1)
- [vgaraujov/bart-base-translation-en-es](https://huggingface.co/vgaraujov/bart-base-translation-en-es)
- [NYTK/translation-bart-128-en-hu](https://huggingface.co/NYTK/translation-bart-128-en-hu)
- [ahazeemi/bart-base-wmt-en-fr-finetuned](https://huggingface.co/ahazeemi/bart-base-wmt-en-fr-finetuned)
- [NYTK/translation-bart-hu-en](https://huggingface.co/NYTK/translation-bart-hu-en) (Not very stable)

⚠️ **Requirements:**
- Model must be BART-based architecture (BART has its own specific settings)
---

### 🛠️ Quick Start: Translation

#### Basic Usage

Use the provided `translation_config.yaml` as a template config:

```bash
# Run translation
python translate.py -config .../translation_config.yaml
```

#### Configuration template structure

```yaml
# translation_config.yaml

# Model and task settings
model: /path/to/converted/mammoth/model
task_id: bart_translation

# Vocabulary files (usually same for HF BART models)
src_vocab:
  en: /path/to/vocab.txt
tgt_vocab:
  es: /path/to/vocab.txt

# Input/output
src: /path/to/source.txt 
output: /path/to/output.txt

# Translation settings
batch_size: 1
beam_size: 1
max_length: 20

# HuggingFace integration
transforms: [huggingface] # on-the-fly tokenization using HF BART tokenizer
src_hf_model_name: /path/to/original/hf/model
tgt_hf_model_name: /path/to/original/hf/model

# Task configuration
tasks:
  bart_translation: 
    src_tgt: en-es
    enc_sharing_group: ["en"]
    dec_sharing_group: ["es"]
    introduce_at_training_step: 0
    weight: 1
```

#### Key Configuration Options

##### Model Settings
- `model`: Path to converted Mammoth model checkpoint
- `task_id`: Must match the `task_id` in `tasks` section

##### Vocabulary
- `src_vocab`/`tgt_vocab`: Language-specific vocabulary files generated during the conversion.
- For HF BART models, there is usually only one vocab shared by the source and target text.

##### HuggingFace Integration
- `transforms: [huggingface]`: Enables HF on-the-fly tokenization/detokenization
- `src_hf_model_name`: Original HF model for tokenization
- `tgt_hf_model_name`: Usually same as source for BART

For the update and file structure changes, see CHANGELOG.md.