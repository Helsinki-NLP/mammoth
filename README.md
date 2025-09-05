# 🦣 MAMMOTH: Massively Multilingual Modular Open Translation @ Helsinki
This repository contains the code for 🦣 MAMMOTH, the modular translation toolkit from Helsinki-NLP.

This library is built on top of OpenNMT-py.
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation framework. It is designed to be research friendly to try out new ideas in translation, summary, morphology, and many other domains. Some companies have proven the code to be production ready.

### Documentation
The original ONMT-py documentation is available [here](https://opennmt.net/OpenNMT-py/).  
Our own modifications (currently-under-development) are documented [here](https://helsinki-nlp.github.io/mammoth/).

**Note:** We would greatly appreciate issue reports for this repository and its documentation. 

### Acknowledgements
We thank the NVIDIA AI Technology Center Finland for their help with the multi-gpu/node implementation.

For the update history in this branch, see CHANGELOG.md.

## Quick Start: HuggingFace Integration

This guide shows: 
- how to quickly convert and use HuggingFace BART models with Mammoth
- how to inference with a converted (or general) Mammoth model
- how to continue training a Mammoth model from a converted checkpoint (or any checkpoint)
- how to push the Mammoth BART model to HuggingFace.

### Setup

```bash
# Copy repository from feat/hf_integration branch
git clone -b feat/hf_integration https://github.com/Helsinki-NLP/mammoth.git

# cd into the directory
cd mammoth

# Install the dependencies
pip install -r requirements.txt
```

### HF2Mammoth Converter: convert Huggingface BART model to Mammoth

#### Basic Usage

Convert a HuggingFace BART model to Mammoth format:

```bash
# Basic conversion from Hugging Face model hub (language pair needs to be specified)
python hf2mammoth.py vgaraujov/bart-base-translation-en-es ./save/path/converted_model --src-lang en --tgt-lang es

# Local model conversion (language pair needs to be specified)
python hf2mammoth.py /path/to/local/bart/model ./save/path/converted_model --src-lang en --tgt-lang es
```

**Important Notes:**
- The "converted_model" parameter serves as the prefix for converted model component names and does not affect inference.
- While BART architecture supports many tasks, this setup has been tuned and tested specifically for translation tasks (using BartForConditionalGeneration). Different model heads may have varying weights and could produce unexpected results.
- The HF BART model used for debugging is [vgaraujov/bart-base-translation-en-es](https://huggingface.co/vgaraujov/bart-base-translation-en-es).


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

**Note:** Mammoth models are saved as components, which is normal behavior.

#### Supported Models

✅ **Tested Models:** (batch size: 1)
- [vgaraujov/bart-base-translation-en-es](https://huggingface.co/vgaraujov/bart-base-translation-en-es)
- [NYTK/translation-bart-128-en-hu](https://huggingface.co/NYTK/translation-bart-128-en-hu)
- [ahazeemi/bart-base-wmt-en-fr-finetuned](https://huggingface.co/ahazeemi/bart-base-wmt-en-fr-finetuned)
- [NYTK/translation-bart-hu-en](https://huggingface.co/NYTK/translation-bart-hu-en) (Not very stable)

⚠️ **Requirements:**
- Model must be BART-based architecture (BART has its own specific settings)
---

### Translation

#### Basic Usage

Use the provided `translation_config.yaml` as a template config.

```bash
# Run translation
cd mammoth
python translate.py -config translation_config.yaml
```

**Note:** The `task_id` must match the `task_id` specified in the `tasks` section.


### Training from the converted model (or any checkpoint)

#### Basic Usage

Use the provided `training_ft.yaml` as a template config

```bash
# Run training
cd mammoth
python train.py -config training_ft.yaml
```

### Mammoth2HF Converter: convert Mammoth model to Huggingface BART model 

**Two Conversion Scenarios:**

1. **HF → Mammoth → HF:** When converting a Mammoth model that was originally converted from HuggingFace, the model uses the original HF tokenizer (the same tokenizer used during inference).

2. **Native Mammoth → HF:** When converting a model trained natively in the Mammoth framework to HuggingFace format, special considerations apply. Since HuggingFace BART natively uses a BPE tokenizer requiring both vocab and merges files, if the Mammoth model lacks the "merges" file (particularly when trained with a SentencePiece tokenizer), switch to LlamaTokenizer. This tokenizer requires a SentencePiece model that was created during the original model training (vocabulary building phase). 

#### Basic Usage

Scenario 1:

##### Convert Model Only

```bash
python hf2mammoth2hf.py \
  --mammoth_model /path/to/mammoth/checkpoint \
  --hf_model /path/to/output/hf_model \
  --tokenizer /path/to/original/tokenizer 
```

##### Convert and Push to HuggingFace Hub

```bash
python hf2mammoth2hf.py \
  --mammoth_model /path/to/mammoth/checkpoint \
  --hf_model your-username/model-name \
  --tokenizer /path/to/original/tokenizer \
  --push_to_hub
```