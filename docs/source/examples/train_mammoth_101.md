
# Training MAMMOTH 101


This example uses the [Europarl parallel corpus](https://www.statmt.org/europarl/) - a multilingual resource extracted from European Parliament proceedings, containing text in 21 European languages. If you use the data in your research, please cite the paper by Philipp Koehn, "Europarl: A Parallel Corpus for Statistical Machine Translation," presented at the MT Summit 2005.
The tokenization is done with [sentencepiece](https://github.com/google/sentencepiece).

## Step 0: Download the data and SentencePiece model

Download the Release v7 - a further expanded and improved version of the Europarl corpus on 15 May 2012 - from the original website or download the processed data by us:
```bash
wget https://mammoth101.a3s.fi/europarl.tar.gz
```

We use a SentencePiece model trained on OPUS Tatoeba Challenge data with 64k vocabulary size. Download the SentencePiece model and the vocabulary:
```bash
# Download the SentencePiece model
wget https://mammoth101.a3s.fi/opusTC.mul.64k.spm
# Download the vocabulary
wget https://mammoth101.a3s.fi/opusTC.mul.vocab.onmt
```

## Step 1: Prepare the data

Then, read parallel text data, processes it, and generate output files for training and validation sets. 
Here's a high-level summary of the main processing steps. For each language in 'langs,' 
- read parallel data files.
- clean the data by removing empty lines.
- shuffle the data randomly.
- tokenizes the text using SentencePiece and writes the tokenized data to separate output files for training and validation sets.

We use a positional argument 'lang' that can accept one or more values, for specifying the languages (e.g., `bg` and `cs` as used in Europarl) to process.

You're free to skip this step if you directly download the processed data.

```python
import argparse
import random

import tqdm
import sentencepiece as sp

parser = argparse.ArgumentParser()
parser.add_argument('lang', nargs='+')
langs = parser.parse_args().lang

sp_path = 'vocab/opusTC.mul.64k.spm'
spm = sp.SentencePieceProcessor(model_file=sp_path)

for lang in tqdm.tqdm(langs):
    en_side_in = f'{lang}-en/europarl-v7.{lang}-en.en'
    xx_side_in = f'{lang}-en/europarl-v7.{lang}-en.{lang}'
    with open(xx_side_in) as xx_stream, open(en_side_in) as en_stream:
        data = zip(map(str.strip, xx_stream), map(str.strip, en_stream))
        data = [(xx, en) for xx, en in tqdm.tqdm(data, leave=False, desc=f'read {lang}') if xx and en] # drop empty lines
        random.shuffle(data)
    en_side_out = f'{lang}-en/valid.{lang}-en.en.sp'
    xx_side_out = f'{lang}-en/valid.{lang}-en.{lang}.sp'
    with open(xx_side_out, 'w') as xx_stream, open(en_side_out, 'w') as en_stream:
        for xx, en in tqdm.tqdm(data[:1000], leave=False, desc=f'valid {lang}'):
            print(*spm.encode(xx, out_type=str), file=xx_stream)
            print(*spm.encode(en, out_type=str), file=en_stream)
    en_side_out = f'{lang}-en/train.{lang}-en.en.sp'
    xx_side_out = f'{lang}-en/train.{lang}-en.{lang}.sp'
    with open(xx_side_out, 'w') as xx_stream, open(en_side_out, 'w') as en_stream:
        for xx, en in tqdm.tqdm(data[1000:], leave=False, desc=f'train {lang}'):
            print(*spm.encode(xx, out_type=str), file=xx_stream)
            print(*spm.encode(en, out_type=str), file=en_stream)
```

## Step 3: Configuration

We can define a configuration for the model, sharing scheme, and training arguments. You can choose to manually write your config in a yaml file, or use our automatic config generation tool.
Here, we provide two configuration examples for training a dummy transformer model in single-node and multi-node settings.

<details>
<summary>Single-node configuration</summary>

```yaml
src_vocab:
  'bg': path_to_vocab/opusTC.mul.vocab.onmt
  'cs': path_to_vocab/opusTC.mul.vocab.onmt
  'da': path_to_vocab/opusTC.mul.vocab.onmt
  'de': path_to_vocab/opusTC.mul.vocab.onmt
  'el': path_to_vocab/opusTC.mul.vocab.onmt
  'en': path_to_vocab/opusTC.mul.vocab.onmt
  'es': path_to_vocab/opusTC.mul.vocab.onmt
  'et': path_to_vocab/opusTC.mul.vocab.onmt
  'fi': path_to_vocab/opusTC.mul.vocab.onmt
  'fr': path_to_vocab/opusTC.mul.vocab.onmt
  'hu': path_to_vocab/opusTC.mul.vocab.onmt
  'it': path_to_vocab/opusTC.mul.vocab.onmt
  'lt': path_to_vocab/opusTC.mul.vocab.onmt
  'lv': path_to_vocab/opusTC.mul.vocab.onmt
  'nl': path_to_vocab/opusTC.mul.vocab.onmt
  'pl': path_to_vocab/opusTC.mul.vocab.onmt
  'pt': path_to_vocab/opusTC.mul.vocab.onmt
  'ro': path_to_vocab/opusTC.mul.vocab.onmt
  'sk': path_to_vocab/opusTC.mul.vocab.onmt
  'sl': path_to_vocab/opusTC.mul.vocab.onmt
  'sv': path_to_vocab/opusTC.mul.vocab.onmt
tgt_vocab:
  'bg': path_to_vocab/opusTC.mul.vocab.onmt
  'cs': path_to_vocab/opusTC.mul.vocab.onmt
  'da': path_to_vocab/opusTC.mul.vocab.onmt
  'de': path_to_vocab/opusTC.mul.vocab.onmt
  'el': path_to_vocab/opusTC.mul.vocab.onmt
  'en': path_to_vocab/opusTC.mul.vocab.onmt
  'es': path_to_vocab/opusTC.mul.vocab.onmt
  'et': path_to_vocab/opusTC.mul.vocab.onmt
  'fi': path_to_vocab/opusTC.mul.vocab.onmt
  'fr': path_to_vocab/opusTC.mul.vocab.onmt
  'hu': path_to_vocab/opusTC.mul.vocab.onmt
  'it': path_to_vocab/opusTC.mul.vocab.onmt
  'lt': path_to_vocab/opusTC.mul.vocab.onmt
  'lv': path_to_vocab/opusTC.mul.vocab.onmt
  'nl': path_to_vocab/opusTC.mul.vocab.onmt
  'pl': path_to_vocab/opusTC.mul.vocab.onmt
  'pt': path_to_vocab/opusTC.mul.vocab.onmt
  'ro': path_to_vocab/opusTC.mul.vocab.onmt
  'sk': path_to_vocab/opusTC.mul.vocab.onmt
  'sl': path_to_vocab/opusTC.mul.vocab.onmt
  'sv': path_to_vocab/opusTC.mul.vocab.onmt

overwrite: False
tasks:
  # GPU 0:0
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg]
    dec_sharing_group: [en]
    node_gpu: 0:0
    path_src: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_tgt: path_to_europarl/bg-en/train.bg-en.en.sp
    path_valid_src: path_to_europarl/bg-en/valid.bg-en.bg.sp
    path_valid_tgt: path_to_europarl/bg-en/valid.bg-en.en.sp
    transforms: [filtertoolong]
  train_bg-bg:
    src_tgt: bg-bg
    enc_sharing_group: [bg]
    dec_sharing_group: [bg]
    node_gpu: 0:0
    path_src: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_tgt: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_valid_src: path_to_europarl/bg-en/valid.bg-en.bg.sp
    path_valid_tgt: path_to_europarl/bg-en/valid.bg-en.bg.sp
    transforms: [filtertoolong, denoising]
  train_en-bg:
    src_tgt: en-bg
    enc_sharing_group: [en]
    dec_sharing_group: [bg]
    node_gpu: 0:0
    path_src: path_to_europarl/bg-en/train.bg-en.en.sp
    path_tgt: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_valid_src: path_to_europarl/bg-en/valid.bg-en.en.sp
    path_valid_tgt: path_to_europarl/bg-en/valid.bg-en.bg.sp
    transforms: [filtertoolong]
  # GPU 0:1
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs]
    dec_sharing_group: [en]
    node_gpu: 0:1
    path_src: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_tgt: path_to_europarl/cs-en/train.cs-en.en.sp
    path_valid_src: path_to_europarl/cs-en/valid.cs-en.cs.sp
    path_valid_tgt: path_to_europarl/cs-en/valid.cs-en.en.sp
    transforms: [filtertoolong]
  train_cs-cs:
    src_tgt: cs-cs
    enc_sharing_group: [cs]
    dec_sharing_group: [cs]
    node_gpu: 0:1
    path_src: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_tgt: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_valid_src: path_to_europarl/cs-en/valid.cs-en.cs.sp
    path_valid_tgt: path_to_europarl/cs-en/valid.cs-en.cs.sp
    transforms: [filtertoolong, denoising]
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en]
    dec_sharing_group: [cs]
    node_gpu: 0:1
    path_src: path_to_europarl/cs-en/train.cs-en.en.sp
    path_tgt: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_valid_src: path_to_europarl/cs-en/valid.cs-en.en.sp
    path_valid_tgt: path_to_europarl/cs-en/valid.cs-en.cs.sp
    transforms: [filtertoolong]
  # GPU 0:2
  train_da-en:
    src_tgt: da-en
    enc_sharing_group: [da]
    dec_sharing_group: [en]
    node_gpu: 0:2
    path_src: path_to_europarl/da-en/train.da-en.da.sp
    path_tgt: path_to_europarl/da-en/train.da-en.en.sp
    path_valid_src: path_to_europarl/da-en/valid.da-en.da.sp
    path_valid_tgt: path_to_europarl/da-en/valid.da-en.en.sp
    transforms: [filtertoolong]
  train_da-da:
    src_tgt: da-da
    enc_sharing_group: [da]
    dec_sharing_group: [da]
    node_gpu: 0:2
    path_src: path_to_europarl/da-en/train.da-en.da.sp
    path_tgt: path_to_europarl/da-en/train.da-en.da.sp
    path_valid_src: path_to_europarl/da-en/valid.da-en.da.sp
    path_valid_tgt: path_to_europarl/da-en/valid.da-en.da.sp
    transforms: [filtertoolong, denoising]
  train_en-da:
    src_tgt: en-da
    enc_sharing_group: [en]
    dec_sharing_group: [da]
    node_gpu: 0:2
    path_src: path_to_europarl/da-en/train.da-en.en.sp
    path_tgt: path_to_europarl/da-en/train.da-en.da.sp
    path_valid_src: path_to_europarl/da-en/valid.da-en.en.sp
    path_valid_tgt: path_to_europarl/da-en/valid.da-en.da.sp
    transforms: [filtertoolong]
  # GPU 0:3
  train_de-en:
    src_tgt: de-en
    enc_sharing_group: [de]
    dec_sharing_group: [en]
    node_gpu: 0:3
    path_src: path_to_europarl/de-en/train.de-en.de.sp
    path_tgt: path_to_europarl/de-en/train.de-en.en.sp
    path_valid_src: path_to_europarl/de-en/valid.de-en.de.sp
    path_valid_tgt: path_to_europarl/de-en/valid.de-en.en.sp
    transforms: [filtertoolong]
  train_de-de:
    src_tgt: de-de
    enc_sharing_group: [de]
    dec_sharing_group: [de]
    node_gpu: 0:3
    path_src: path_to_europarl/de-en/train.de-en.de.sp
    path_tgt: path_to_europarl/de-en/train.de-en.de.sp
    path_valid_src: path_to_europarl/de-en/valid.de-en.de.sp
    path_valid_tgt: path_to_europarl/de-en/valid.de-en.de.sp
    transforms: [filtertoolong, denoising]
  train_en-de:
    src_tgt: en-de
    enc_sharing_group: [en]
    dec_sharing_group: [de]
    node_gpu: 0:3
    path_src: path_to_europarl/de-en/train.de-en.en.sp
    path_tgt: path_to_europarl/de-en/train.de-en.de.sp
    path_valid_src: path_to_europarl/de-en/valid.de-en.en.sp
    path_valid_tgt: path_to_europarl/de-en/valid.de-en.de.sp
    transforms: [filtertoolong]
  # GPU 0:0
  train_el-en:
    src_tgt: el-en
    enc_sharing_group: [el]
    dec_sharing_group: [en]
    node_gpu: 0:0
    path_src: path_to_europarl/el-en/train.el-en.el.sp
    path_tgt: path_to_europarl/el-en/train.el-en.en.sp
    path_valid_src: path_to_europarl/el-en/valid.el-en.el.sp
    path_valid_tgt: path_to_europarl/el-en/valid.el-en.en.sp
    transforms: [filtertoolong]
  train_el-el:
    src_tgt: el-el
    enc_sharing_group: [el]
    dec_sharing_group: [el]
    node_gpu: 0:0
    path_src: path_to_europarl/el-en/train.el-en.el.sp
    path_tgt: path_to_europarl/el-en/train.el-en.el.sp
    path_valid_src: path_to_europarl/el-en/valid.el-en.el.sp
    path_valid_tgt: path_to_europarl/el-en/valid.el-en.el.sp
    transforms: [filtertoolong, denoising]
  train_en-el:
    src_tgt: en-el
    enc_sharing_group: [en]
    dec_sharing_group: [el]
    node_gpu: 0:0
    path_src: path_to_europarl/el-en/train.el-en.en.sp
    path_tgt: path_to_europarl/el-en/train.el-en.el.sp
    path_valid_src: path_to_europarl/el-en/valid.el-en.en.sp
    path_valid_tgt: path_to_europarl/el-en/valid.el-en.el.sp
    transforms: [filtertoolong]
  # GPU 0:1
  train_es-en:
    src_tgt: es-en
    enc_sharing_group: [es]
    dec_sharing_group: [en]
    node_gpu: 0:1
    path_src: path_to_europarl/es-en/train.es-en.es.sp
    path_tgt: path_to_europarl/es-en/train.es-en.en.sp
    path_valid_src: path_to_europarl/es-en/valid.es-en.es.sp
    path_valid_tgt: path_to_europarl/es-en/valid.es-en.en.sp
    transforms: [filtertoolong]
  train_es-es:
    src_tgt: es-es
    enc_sharing_group: [es]
    dec_sharing_group: [es]
    node_gpu: 0:1
    path_src: path_to_europarl/es-en/train.es-en.es.sp
    path_tgt: path_to_europarl/es-en/train.es-en.es.sp
    path_valid_src: path_to_europarl/es-en/valid.es-en.es.sp
    path_valid_tgt: path_to_europarl/es-en/valid.es-en.es.sp
    transforms: [filtertoolong, denoising]
  train_en-es:
    src_tgt: en-es
    enc_sharing_group: [en]
    dec_sharing_group: [es]
    node_gpu: 0:1
    path_src: path_to_europarl/es-en/train.es-en.en.sp
    path_tgt: path_to_europarl/es-en/train.es-en.es.sp
    path_valid_src: path_to_europarl/es-en/valid.es-en.en.sp
    path_valid_tgt: path_to_europarl/es-en/valid.es-en.es.sp
    transforms: [filtertoolong]
  # GPU 0:2
  train_et-en:
    src_tgt: et-en
    enc_sharing_group: [et]
    dec_sharing_group: [en]
    node_gpu: 0:2
    path_src: path_to_europarl/et-en/train.et-en.et.sp
    path_tgt: path_to_europarl/et-en/train.et-en.en.sp
    path_valid_src: path_to_europarl/et-en/valid.et-en.et.sp
    path_valid_tgt: path_to_europarl/et-en/valid.et-en.en.sp
    transforms: [filtertoolong]
  train_et-et:
    src_tgt: et-et
    enc_sharing_group: [et]
    dec_sharing_group: [et]
    node_gpu: 0:2
    path_src: path_to_europarl/et-en/train.et-en.et.sp
    path_tgt: path_to_europarl/et-en/train.et-en.et.sp
    path_valid_src: path_to_europarl/et-en/valid.et-en.et.sp
    path_valid_tgt: path_to_europarl/et-en/valid.et-en.et.sp
    transforms: [filtertoolong, denoising]
  train_en-et:
    src_tgt: en-et
    enc_sharing_group: [en]
    dec_sharing_group: [et]
    node_gpu: 0:2
    path_src: path_to_europarl/et-en/train.et-en.en.sp
    path_tgt: path_to_europarl/et-en/train.et-en.et.sp
    path_valid_src: path_to_europarl/et-en/valid.et-en.en.sp
    path_valid_tgt: path_to_europarl/et-en/valid.et-en.et.sp
    transforms: [filtertoolong]
  # GPU 0:3
  train_fi-en:
    src_tgt: fi-en
    enc_sharing_group: [fi]
    dec_sharing_group: [en]
    node_gpu: 0:3
    path_src: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_tgt: path_to_europarl/fi-en/train.fi-en.en.sp
    path_valid_src: path_to_europarl/fi-en/valid.fi-en.fi.sp
    path_valid_tgt: path_to_europarl/fi-en/valid.fi-en.en.sp
    transforms: [filtertoolong]
  train_fi-fi:
    src_tgt: fi-fi
    enc_sharing_group: [fi]
    dec_sharing_group: [fi]
    node_gpu: 0:3
    path_src: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_tgt: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_valid_src: path_to_europarl/fi-en/valid.fi-en.fi.sp
    path_valid_tgt: path_to_europarl/fi-en/valid.fi-en.fi.sp
    transforms: [filtertoolong, denoising]
  train_en-fi:
    src_tgt: en-fi
    enc_sharing_group: [en]
    dec_sharing_group: [fi]
    node_gpu: 0:3
    path_src: path_to_europarl/fi-en/train.fi-en.en.sp
    path_tgt: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_valid_src: path_to_europarl/fi-en/valid.fi-en.en.sp
    path_valid_tgt: path_to_europarl/fi-en/valid.fi-en.fi.sp
    transforms: [filtertoolong]
  # GPU 0:0
  train_fr-en:
    src_tgt: fr-en
    enc_sharing_group: [fr]
    dec_sharing_group: [en]
    node_gpu: 0:0
    path_src: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_tgt: path_to_europarl/fr-en/train.fr-en.en.sp
    path_valid_src: path_to_europarl/fr-en/valid.fr-en.fr.sp
    path_valid_tgt: path_to_europarl/fr-en/valid.fr-en.en.sp
    transforms: [filtertoolong]
  train_fr-fr:
    src_tgt: fr-fr
    enc_sharing_group: [fr]
    dec_sharing_group: [fr]
    node_gpu: 0:0
    path_src: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_tgt: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_valid_src: path_to_europarl/fr-en/valid.fr-en.fr.sp
    path_valid_tgt: path_to_europarl/fr-en/valid.fr-en.fr.sp
    transforms: [filtertoolong, denoising]
  train_en-fr:
    src_tgt: en-fr
    enc_sharing_group: [en]
    dec_sharing_group: [fr]
    node_gpu: 0:0
    path_src: path_to_europarl/fr-en/train.fr-en.en.sp
    path_tgt: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_valid_src: path_to_europarl/fr-en/valid.fr-en.en.sp
    path_valid_tgt: path_to_europarl/fr-en/valid.fr-en.fr.sp
    transforms: [filtertoolong]  
  # GPU 0:1
  train_hu-en:
    src_tgt: hu-en
    enc_sharing_group: [hu]
    dec_sharing_group: [en]
    node_gpu: 0:1
    path_src: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_tgt: path_to_europarl/hu-en/train.hu-en.en.sp
    path_valid_src: path_to_europarl/hu-en/valid.hu-en.hu.sp
    path_valid_tgt: path_to_europarl/hu-en/valid.hu-en.en.sp
    transforms: [filtertoolong]
  train_hu-hu:
    src_tgt: hu-hu
    enc_sharing_group: [hu]
    dec_sharing_group: [hu]
    node_gpu: 0:1
    path_src: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_tgt: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_valid_src: path_to_europarl/hu-en/valid.hu-en.hu.sp
    path_valid_tgt: path_to_europarl/hu-en/valid.hu-en.hu.sp
    transforms: [filtertoolong, denoising]
  train_en-hu:
    src_tgt: en-hu
    enc_sharing_group: [en]
    dec_sharing_group: [hu]
    node_gpu: 0:1
    path_src: path_to_europarl/hu-en/train.hu-en.en.sp
    path_tgt: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_valid_src: path_to_europarl/hu-en/valid.hu-en.en.sp
    path_valid_tgt: path_to_europarl/hu-en/valid.hu-en.hu.sp
    transforms: [filtertoolong]
  # GPU 0:2
  train_it-en:
    src_tgt: it-en
    enc_sharing_group: [it]
    dec_sharing_group: [en]
    node_gpu: 0:2
    path_src: path_to_europarl/it-en/train.it-en.it.sp
    path_tgt: path_to_europarl/it-en/train.it-en.en.sp
    path_valid_src: path_to_europarl/it-en/valid.it-en.it.sp
    path_valid_tgt: path_to_europarl/it-en/valid.it-en.en.sp
    transforms: [filtertoolong]
  train_it-it:
    src_tgt: it-it
    enc_sharing_group: [it]
    dec_sharing_group: [it]
    node_gpu: 0:2
    path_src: path_to_europarl/it-en/train.it-en.it.sp
    path_tgt: path_to_europarl/it-en/train.it-en.it.sp
    path_valid_src: path_to_europarl/it-en/valid.it-en.it.sp
    path_valid_tgt: path_to_europarl/it-en/valid.it-en.it.sp
    transforms: [filtertoolong, denoising]
  train_en-it:
    src_tgt: en-it
    enc_sharing_group: [en]
    dec_sharing_group: [it]
    node_gpu: 0:2
    path_src: path_to_europarl/it-en/train.it-en.en.sp
    path_tgt: path_to_europarl/it-en/train.it-en.it.sp
    path_valid_src: path_to_europarl/it-en/valid.it-en.en.sp
    path_valid_tgt: path_to_europarl/it-en/valid.it-en.it.sp
    transforms: [filtertoolong]
  # GPU 0:3
  train_lt-en:
    src_tgt: lt-en
    enc_sharing_group: [lt]
    dec_sharing_group: [en]
    node_gpu: 0:3
    path_src: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_tgt: path_to_europarl/lt-en/train.lt-en.en.sp
    path_valid_src: path_to_europarl/lt-en/valid.lt-en.lt.sp
    path_valid_tgt: path_to_europarl/lt-en/valid.lt-en.en.sp
    transforms: [filtertoolong]
  train_lt-lt:
    src_tgt: lt-lt
    enc_sharing_group: [lt]
    dec_sharing_group: [lt]
    node_gpu: 0:3
    path_src: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_tgt: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_valid_src: path_to_europarl/lt-en/valid.lt-en.lt.sp
    path_valid_tgt: path_to_europarl/lt-en/valid.lt-en.lt.sp
    transforms: [filtertoolong, denoising]
  train_en-lt:
    src_tgt: en-lt
    enc_sharing_group: [en]
    dec_sharing_group: [lt]
    node_gpu: 0:3
    path_src: path_to_europarl/lt-en/train.lt-en.en.sp
    path_tgt: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_valid_src: path_to_europarl/lt-en/valid.lt-en.en.sp
    path_valid_tgt: path_to_europarl/lt-en/valid.lt-en.lt.sp
    transforms: [filtertoolong]
  # GPU 0:0
  train_lv-en:
    src_tgt: lv-en
    enc_sharing_group: [lv]
    dec_sharing_group: [en]
    node_gpu: 0:0
    path_src: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_tgt: path_to_europarl/lv-en/train.lv-en.en.sp
    path_valid_src: path_to_europarl/lv-en/valid.lv-en.lv.sp
    path_valid_tgt: path_to_europarl/lv-en/valid.lv-en.en.sp
    transforms: [filtertoolong]
  train_lv-lv:
    src_tgt: lv-lv
    enc_sharing_group: [lv]
    dec_sharing_group: [lv]
    node_gpu: 0:0
    path_src: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_tgt: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_valid_src: path_to_europarl/lv-en/valid.lv-en.lv.sp
    path_valid_tgt: path_to_europarl/lv-en/valid.lv-en.lv.sp
    transforms: [filtertoolong, denoising]
  train_en-lv:
    src_tgt: en-lv
    enc_sharing_group: [en]
    dec_sharing_group: [lv]
    node_gpu: 0:0
    path_src: path_to_europarl/lv-en/train.lv-en.en.sp
    path_tgt: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_valid_src: path_to_europarl/lv-en/valid.lv-en.en.sp
    path_valid_tgt: path_to_europarl/lv-en/valid.lv-en.lv.sp
    transforms: [filtertoolong]
  # GPU 0:1
  train_nl-en:
    src_tgt: nl-en
    enc_sharing_group: [nl]
    dec_sharing_group: [en]
    node_gpu: 0:1
    path_src: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_tgt: path_to_europarl/nl-en/train.nl-en.en.sp
    path_valid_src: path_to_europarl/nl-en/valid.nl-en.nl.sp
    path_valid_tgt: path_to_europarl/nl-en/valid.nl-en.en.sp
    transforms: [filtertoolong]
  train_nl-nl:
    src_tgt: nl-nl
    enc_sharing_group: [nl]
    dec_sharing_group: [nl]
    node_gpu: 0:1
    path_src: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_tgt: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_valid_src: path_to_europarl/nl-en/valid.nl-en.nl.sp
    path_valid_tgt: path_to_europarl/nl-en/valid.nl-en.nl.sp
    transforms: [filtertoolong, denoising]
  train_en-nl:
    src_tgt: en-nl
    enc_sharing_group: [en]
    dec_sharing_group: [nl]
    node_gpu: 0:1
    path_src: path_to_europarl/nl-en/train.nl-en.en.sp
    path_tgt: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_valid_src: path_to_europarl/nl-en/valid.nl-en.en.sp
    path_valid_tgt: path_to_europarl/nl-en/valid.nl-en.nl.sp
    transforms: [filtertoolong]
  # GPU 0:2
  train_pl-en:
    src_tgt: pl-en
    enc_sharing_group: [pl]
    dec_sharing_group: [en]
    node_gpu: 0:2
    path_src: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_tgt: path_to_europarl/pl-en/train.pl-en.en.sp
    path_valid_src: path_to_europarl/pl-en/valid.pl-en.pl.sp
    path_valid_tgt: path_to_europarl/pl-en/valid.pl-en.en.sp
    transforms: [filtertoolong]
  train_pl-pl:
    src_tgt: pl-pl
    enc_sharing_group: [pl]
    dec_sharing_group: [pl]
    node_gpu: 0:2
    path_src: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_tgt: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_valid_src: path_to_europarl/pl-en/valid.pl-en.pl.sp
    path_valid_tgt: path_to_europarl/pl-en/valid.pl-en.pl.sp
    transforms: [filtertoolong, denoising]
  train_en-pl:
    src_tgt: en-pl
    enc_sharing_group: [en]
    dec_sharing_group: [pl]
    node_gpu: 0:2
    path_src: path_to_europarl/pl-en/train.pl-en.en.sp
    path_tgt: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_valid_src: path_to_europarl/pl-en/valid.pl-en.en.sp
    path_valid_tgt: path_to_europarl/pl-en/valid.pl-en.pl.sp
    transforms: [filtertoolong]
  # GPU 0:3
  train_pt-en:
    src_tgt: pt-en
    enc_sharing_group: [pt]
    dec_sharing_group: [en]
    node_gpu: 0:3
    path_src: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_tgt: path_to_europarl/pt-en/train.pt-en.en.sp
    path_valid_src: path_to_europarl/pt-en/valid.pt-en.pt.sp
    path_valid_tgt: path_to_europarl/pt-en/valid.pt-en.en.sp
    transforms: [filtertoolong]
  train_pt-pt:
    src_tgt: pt-pt
    enc_sharing_group: [pt]
    dec_sharing_group: [pt]
    node_gpu: 0:3
    path_src: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_tgt: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_valid_src: path_to_europarl/pt-en/valid.pt-en.pt.sp
    path_valid_tgt: path_to_europarl/pt-en/valid.pt-en.pt.sp
    transforms: [filtertoolong, denoising]
  train_en-pt:
    src_tgt: en-pt
    enc_sharing_group: [en]
    dec_sharing_group: [pt]
    node_gpu: 0:3
    path_src: path_to_europarl/pt-en/train.pt-en.en.sp
    path_tgt: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_valid_src: path_to_europarl/pt-en/valid.pt-en.en.sp
    path_valid_tgt: path_to_europarl/pt-en/valid.pt-en.pt.sp
    transforms: [filtertoolong]
  # GPU 0:0
  train_ro-en:
    src_tgt: ro-en
    enc_sharing_group: [ro]
    dec_sharing_group: [en]
    node_gpu: 0:0
    path_src: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_tgt: path_to_europarl/ro-en/train.ro-en.en.sp
    path_valid_src: path_to_europarl/ro-en/valid.ro-en.ro.sp
    path_valid_tgt: path_to_europarl/ro-en/valid.ro-en.en.sp
    transforms: [filtertoolong]
  train_ro-ro:
    src_tgt: ro-ro
    enc_sharing_group: [ro]
    dec_sharing_group: [ro]
    node_gpu: 0:0
    path_src: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_tgt: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_valid_src: path_to_europarl/ro-en/valid.ro-en.ro.sp
    path_valid_tgt: path_to_europarl/ro-en/valid.ro-en.ro.sp
    transforms: [filtertoolong, denoising]
  train_en-ro:
    src_tgt: en-ro
    enc_sharing_group: [en]
    dec_sharing_group: [ro]
    node_gpu: 0:0
    path_src: path_to_europarl/ro-en/train.ro-en.en.sp
    path_tgt: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_valid_src: path_to_europarl/ro-en/valid.ro-en.en.sp
    path_valid_tgt: path_to_europarl/ro-en/valid.ro-en.ro.sp
    transforms: [filtertoolong]
  # GPU 0:1
  train_sk-en:
    src_tgt: sk-en
    enc_sharing_group: [sk]
    dec_sharing_group: [en]
    node_gpu: 0:1
    path_src: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_tgt: path_to_europarl/sk-en/train.sk-en.en.sp
    path_valid_src: path_to_europarl/sk-en/valid.sk-en.sk.sp
    path_valid_tgt: path_to_europarl/sk-en/valid.sk-en.en.sp
    transforms: [filtertoolong]
  train_sk-sk:
    src_tgt: sk-sk
    enc_sharing_group: [sk]
    dec_sharing_group: [sk]
    node_gpu: 0:1
    path_src: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_tgt: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_valid_src: path_to_europarl/sk-en/valid.sk-en.sk.sp
    path_valid_tgt: path_to_europarl/sk-en/valid.sk-en.sk.sp
    transforms: [filtertoolong, denoising]
  train_en-sk:
    src_tgt: en-sk
    enc_sharing_group: [en]
    dec_sharing_group: [sk]
    node_gpu: 0:1
    path_src: path_to_europarl/sk-en/train.sk-en.en.sp
    path_tgt: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_valid_src: path_to_europarl/sk-en/valid.sk-en.en.sp
    path_valid_tgt: path_to_europarl/sk-en/valid.sk-en.sk.sp
    transforms: [filtertoolong]
  # GPU 0:2
  train_sl-en:
    src_tgt: sl-en
    enc_sharing_group: [sl]
    dec_sharing_group: [en]
    node_gpu: 0:2
    path_src: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_tgt: path_to_europarl/sl-en/train.sl-en.en.sp
    path_valid_src: path_to_europarl/sl-en/valid.sl-en.sl.sp
    path_valid_tgt: path_to_europarl/sl-en/valid.sl-en.en.sp
    transforms: [filtertoolong]
  train_sl-sl:
    src_tgt: sl-sl
    enc_sharing_group: [sl]
    dec_sharing_group: [sl]
    node_gpu: 0:2
    path_src: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_tgt: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_valid_src: path_to_europarl/sl-en/valid.sl-en.sl.sp
    path_valid_tgt: path_to_europarl/sl-en/valid.sl-en.sl.sp
    transforms: [filtertoolong, denoising]
  train_en-sl:
    src_tgt: en-sl
    enc_sharing_group: [en]
    dec_sharing_group: [sl]
    node_gpu: 0:2
    path_src: path_to_europarl/sl-en/train.sl-en.en.sp
    path_tgt: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_valid_src: path_to_europarl/sl-en/valid.sl-en.en.sp
    path_valid_tgt: path_to_europarl/sl-en/valid.sl-en.sl.sp
    transforms: [filtertoolong]
  # GPU 0:3
  train_sv-en:
    src_tgt: sv-en
    enc_sharing_group: [sv]
    dec_sharing_group: [en]
    node_gpu: 0:3
    path_src: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_tgt: path_to_europarl/sv-en/train.sv-en.en.sp
    path_valid_src: path_to_europarl/sv-en/valid.sv-en.sv.sp
    path_valid_tgt: path_to_europarl/sv-en/valid.sv-en.en.sp
    transforms: [filtertoolong]
  train_sv-sv:
    src_tgt: sv-sv
    enc_sharing_group: [sv]
    dec_sharing_group: [sv]
    node_gpu: 0:3
    path_src: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_tgt: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_valid_src: path_to_europarl/sv-en/valid.sv-en.sv.sp
    path_valid_tgt: path_to_europarl/sv-en/valid.sv-en.sv.sp
    transforms: [filtertoolong, denoising]
  train_en-sv:
    src_tgt: en-sv
    enc_sharing_group: [en]
    dec_sharing_group: [sv]
    node_gpu: 0:3
    path_src: path_to_europarl/sv-en/train.sv-en.en.sp
    path_tgt: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_valid_src: path_to_europarl/sv-en/valid.sv-en.en.sp
    path_valid_tgt: path_to_europarl/sv-en/valid.sv-en.sv.sp
    transforms: [filtertoolong]

        
### Transform related opts:
#### Filter
src_seq_length: 200
tgt_seq_length: 200
#### Bart
src_subword_type: sentencepiece
tgt_subword_type: sentencepiece
mask_ratio: 0.2
replace_length: 1

# silently ignore empty lines in the data
skip_empty_level: silent

batch_size: 4096
batch_type: tokens
normalization: tokens
valid_batch_size: 4096
max_generator_batches: 2
src_vocab_size: 100000
tgt_vocab_size: 100000
encoder_type: transformer
decoder_type: transformer
model_dim: 512
transformer_ff: 2048
heads: 8
enc_layers: [6]
dec_layers: [6]
dropout: 0.1
label_smoothing: 0.1
param_init: 0.0
param_init_glorot: true
position_encoding: true
valid_steps: 10000
warmup_steps: 10000
report_every: 100
save_checkpoint_steps: 5000000
# save_checkpoint_steps: 50000
keep_checkpoint: -1
accum_count: 1
optim: adafactor
decay_method: none
learning_rate: 3.0
max_grad_norm: 0.0
seed: 3435
model_type: text
save_all_gpus: false

world_size: 4
gpu_ranks: [0, 1, 2, 3]
node_rank: 0

early_stopping: 5
early_stopping_criteria: accuracy
```
</details>

<details>
<summary>Multi-node configuration</summary>

```yaml

src_vocab:
  'bg': path_to_vocab/opusTC.mul.vocab.onmt
  'cs': path_to_vocab/opusTC.mul.vocab.onmt
  'da': path_to_vocab/opusTC.mul.vocab.onmt
  'de': path_to_vocab/opusTC.mul.vocab.onmt
  'el': path_to_vocab/opusTC.mul.vocab.onmt
  'en': path_to_vocab/opusTC.mul.vocab.onmt
  'es': path_to_vocab/opusTC.mul.vocab.onmt
  'et': path_to_vocab/opusTC.mul.vocab.onmt
  'fi': path_to_vocab/opusTC.mul.vocab.onmt
  'fr': path_to_vocab/opusTC.mul.vocab.onmt
  'hu': path_to_vocab/opusTC.mul.vocab.onmt
  'it': path_to_vocab/opusTC.mul.vocab.onmt
  'lt': path_to_vocab/opusTC.mul.vocab.onmt
  'lv': path_to_vocab/opusTC.mul.vocab.onmt
  'nl': path_to_vocab/opusTC.mul.vocab.onmt
  'pl': path_to_vocab/opusTC.mul.vocab.onmt
  'pt': path_to_vocab/opusTC.mul.vocab.onmt
  'ro': path_to_vocab/opusTC.mul.vocab.onmt
  'sk': path_to_vocab/opusTC.mul.vocab.onmt
  'sl': path_to_vocab/opusTC.mul.vocab.onmt
  'sv': path_to_vocab/opusTC.mul.vocab.onmt
tgt_vocab:
  'bg': path_to_vocab/opusTC.mul.vocab.onmt
  'cs': path_to_vocab/opusTC.mul.vocab.onmt
  'da': path_to_vocab/opusTC.mul.vocab.onmt
  'de': path_to_vocab/opusTC.mul.vocab.onmt
  'el': path_to_vocab/opusTC.mul.vocab.onmt
  'en': path_to_vocab/opusTC.mul.vocab.onmt
  'es': path_to_vocab/opusTC.mul.vocab.onmt
  'et': path_to_vocab/opusTC.mul.vocab.onmt
  'fi': path_to_vocab/opusTC.mul.vocab.onmt
  'fr': path_to_vocab/opusTC.mul.vocab.onmt
  'hu': path_to_vocab/opusTC.mul.vocab.onmt
  'it': path_to_vocab/opusTC.mul.vocab.onmt
  'lt': path_to_vocab/opusTC.mul.vocab.onmt
  'lv': path_to_vocab/opusTC.mul.vocab.onmt
  'nl': path_to_vocab/opusTC.mul.vocab.onmt
  'pl': path_to_vocab/opusTC.mul.vocab.onmt
  'pt': path_to_vocab/opusTC.mul.vocab.onmt
  'ro': path_to_vocab/opusTC.mul.vocab.onmt
  'sk': path_to_vocab/opusTC.mul.vocab.onmt
  'sl': path_to_vocab/opusTC.mul.vocab.onmt
  'sv': path_to_vocab/opusTC.mul.vocab.onmt

overwrite: False
tasks:
  # GPU 0:0
  train_bg-en:
    src_tgt: bg-en
    enc_sharing_group: [bg]
    dec_sharing_group: [en]
    node_gpu: "0:0"
    path_src: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_tgt: path_to_europarl/bg-en/train.bg-en.en.sp
    path_valid_src: path_to_europarl/bg-en/valid.bg-en.bg.sp
    path_valid_tgt: path_to_europarl/bg-en/valid.bg-en.en.sp
    transforms: [filtertoolong]
  train_bg-bg:
    src_tgt: bg-bg
    enc_sharing_group: [bg]
    dec_sharing_group: [bg]
    node_gpu: "0:0"
    path_src: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_tgt: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_valid_src: path_to_europarl/bg-en/valid.bg-en.bg.sp
    path_valid_tgt: path_to_europarl/bg-en/valid.bg-en.bg.sp
    transforms: [filtertoolong, denoising]
  train_en-bg:
    src_tgt: en-bg
    enc_sharing_group: [en]
    dec_sharing_group: [bg]
    node_gpu: "0:0"
    path_src: path_to_europarl/bg-en/train.bg-en.en.sp
    path_tgt: path_to_europarl/bg-en/train.bg-en.bg.sp
    path_valid_src: path_to_europarl/bg-en/valid.bg-en.en.sp
    path_valid_tgt: path_to_europarl/bg-en/valid.bg-en.bg.sp
    transforms: [filtertoolong]
  # GPU 0:1
  train_cs-en:
    src_tgt: cs-en
    enc_sharing_group: [cs]
    dec_sharing_group: [en]
    node_gpu: "0:1"
    path_src: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_tgt: path_to_europarl/cs-en/train.cs-en.en.sp
    path_valid_src: path_to_europarl/cs-en/valid.cs-en.cs.sp
    path_valid_tgt: path_to_europarl/cs-en/valid.cs-en.en.sp
    transforms: [filtertoolong]
  train_cs-cs:
    src_tgt: cs-cs
    enc_sharing_group: [cs]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_tgt: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_valid_src: path_to_europarl/cs-en/valid.cs-en.cs.sp
    path_valid_tgt: path_to_europarl/cs-en/valid.cs-en.cs.sp
    transforms: [filtertoolong, denoising]
  train_en-cs:
    src_tgt: en-cs
    enc_sharing_group: [en]
    dec_sharing_group: [cs]
    node_gpu: "0:1"
    path_src: path_to_europarl/cs-en/train.cs-en.en.sp
    path_tgt: path_to_europarl/cs-en/train.cs-en.cs.sp
    path_valid_src: path_to_europarl/cs-en/valid.cs-en.en.sp
    path_valid_tgt: path_to_europarl/cs-en/valid.cs-en.cs.sp
    transforms: [filtertoolong]
  # GPU 0:2
  train_da-en:
    src_tgt: da-en
    enc_sharing_group: [da]
    dec_sharing_group: [en]
    node_gpu: "0:2"
    path_src: path_to_europarl/da-en/train.da-en.da.sp
    path_tgt: path_to_europarl/da-en/train.da-en.en.sp
    path_valid_src: path_to_europarl/da-en/valid.da-en.da.sp
    path_valid_tgt: path_to_europarl/da-en/valid.da-en.en.sp
    transforms: [filtertoolong]
  train_da-da:
    src_tgt: da-da
    enc_sharing_group: [da]
    dec_sharing_group: [da]
    node_gpu: "0:2"
    path_src: path_to_europarl/da-en/train.da-en.da.sp
    path_tgt: path_to_europarl/da-en/train.da-en.da.sp
    path_valid_src: path_to_europarl/da-en/valid.da-en.da.sp
    path_valid_tgt: path_to_europarl/da-en/valid.da-en.da.sp
    transforms: [filtertoolong, denoising]
  train_en-da:
    src_tgt: en-da
    enc_sharing_group: [en]
    dec_sharing_group: [da]
    node_gpu: "0:2"
    path_src: path_to_europarl/da-en/train.da-en.en.sp
    path_tgt: path_to_europarl/da-en/train.da-en.da.sp
    path_valid_src: path_to_europarl/da-en/valid.da-en.en.sp
    path_valid_tgt: path_to_europarl/da-en/valid.da-en.da.sp
    transforms: [filtertoolong]
  # GPU 0:3
  train_de-en:
    src_tgt: de-en
    enc_sharing_group: [de]
    dec_sharing_group: [en]
    node_gpu: "0:3"
    path_src: path_to_europarl/de-en/train.de-en.de.sp
    path_tgt: path_to_europarl/de-en/train.de-en.en.sp
    path_valid_src: path_to_europarl/de-en/valid.de-en.de.sp
    path_valid_tgt: path_to_europarl/de-en/valid.de-en.en.sp
    transforms: [filtertoolong]
  train_de-de:
    src_tgt: de-de
    enc_sharing_group: [de]
    dec_sharing_group: [de]
    node_gpu: "0:3"
    path_src: path_to_europarl/de-en/train.de-en.de.sp
    path_tgt: path_to_europarl/de-en/train.de-en.de.sp
    path_valid_src: path_to_europarl/de-en/valid.de-en.de.sp
    path_valid_tgt: path_to_europarl/de-en/valid.de-en.de.sp
    transforms: [filtertoolong, denoising]
  train_en-de:
    src_tgt: en-de
    enc_sharing_group: [en]
    dec_sharing_group: [de]
    node_gpu: "0:3"
    path_src: path_to_europarl/de-en/train.de-en.en.sp
    path_tgt: path_to_europarl/de-en/train.de-en.de.sp
    path_valid_src: path_to_europarl/de-en/valid.de-en.en.sp
    path_valid_tgt: path_to_europarl/de-en/valid.de-en.de.sp
    transforms: [filtertoolong]
  # GPU 1:0
  train_el-en:
    src_tgt: el-en
    enc_sharing_group: [el]
    dec_sharing_group: [en]
    node_gpu: "1:0"
    path_src: path_to_europarl/el-en/train.el-en.el.sp
    path_tgt: path_to_europarl/el-en/train.el-en.en.sp
    path_valid_src: path_to_europarl/el-en/valid.el-en.el.sp
    path_valid_tgt: path_to_europarl/el-en/valid.el-en.en.sp
    transforms: [filtertoolong]
  train_el-el:
    src_tgt: el-el
    enc_sharing_group: [el]
    dec_sharing_group: [el]
    node_gpu: "1:0"
    path_src: path_to_europarl/el-en/train.el-en.el.sp
    path_tgt: path_to_europarl/el-en/train.el-en.el.sp
    path_valid_src: path_to_europarl/el-en/valid.el-en.el.sp
    path_valid_tgt: path_to_europarl/el-en/valid.el-en.el.sp
    transforms: [filtertoolong, denoising]
  train_en-el:
    src_tgt: en-el
    enc_sharing_group: [en]
    dec_sharing_group: [el]
    node_gpu: "1:0"
    path_src: path_to_europarl/el-en/train.el-en.en.sp
    path_tgt: path_to_europarl/el-en/train.el-en.el.sp
    path_valid_src: path_to_europarl/el-en/valid.el-en.en.sp
    path_valid_tgt: path_to_europarl/el-en/valid.el-en.el.sp
    transforms: [filtertoolong]
  # GPU 1:1
  train_es-en:
    src_tgt: es-en
    enc_sharing_group: [es]
    dec_sharing_group: [en]
    node_gpu: "1:1"
    path_src: path_to_europarl/es-en/train.es-en.es.sp
    path_tgt: path_to_europarl/es-en/train.es-en.en.sp
    path_valid_src: path_to_europarl/es-en/valid.es-en.es.sp
    path_valid_tgt: path_to_europarl/es-en/valid.es-en.en.sp
    transforms: [filtertoolong]
  train_es-es:
    src_tgt: es-es
    enc_sharing_group: [es]
    dec_sharing_group: [es]
    node_gpu: "1:1"
    path_src: path_to_europarl/es-en/train.es-en.es.sp
    path_tgt: path_to_europarl/es-en/train.es-en.es.sp
    path_valid_src: path_to_europarl/es-en/valid.es-en.es.sp
    path_valid_tgt: path_to_europarl/es-en/valid.es-en.es.sp
    transforms: [filtertoolong, denoising]
  train_en-es:
    src_tgt: en-es
    enc_sharing_group: [en]
    dec_sharing_group: [es]
    node_gpu: "1:1"
    path_src: path_to_europarl/es-en/train.es-en.en.sp
    path_tgt: path_to_europarl/es-en/train.es-en.es.sp
    path_valid_src: path_to_europarl/es-en/valid.es-en.en.sp
    path_valid_tgt: path_to_europarl/es-en/valid.es-en.es.sp
    transforms: [filtertoolong]
  # GPU 1:2
  train_et-en:
    src_tgt: et-en
    enc_sharing_group: [et]
    dec_sharing_group: [en]
    node_gpu: "1:2"
    path_src: path_to_europarl/et-en/train.et-en.et.sp
    path_tgt: path_to_europarl/et-en/train.et-en.en.sp
    path_valid_src: path_to_europarl/et-en/valid.et-en.et.sp
    path_valid_tgt: path_to_europarl/et-en/valid.et-en.en.sp
    transforms: [filtertoolong]
  train_et-et:
    src_tgt: et-et
    enc_sharing_group: [et]
    dec_sharing_group: [et]
    node_gpu: "1:2"
    path_src: path_to_europarl/et-en/train.et-en.et.sp
    path_tgt: path_to_europarl/et-en/train.et-en.et.sp
    path_valid_src: path_to_europarl/et-en/valid.et-en.et.sp
    path_valid_tgt: path_to_europarl/et-en/valid.et-en.et.sp
    transforms: [filtertoolong, denoising]
  train_en-et:
    src_tgt: en-et
    enc_sharing_group: [en]
    dec_sharing_group: [et]
    node_gpu: "1:2"
    path_src: path_to_europarl/et-en/train.et-en.en.sp
    path_tgt: path_to_europarl/et-en/train.et-en.et.sp
    path_valid_src: path_to_europarl/et-en/valid.et-en.en.sp
    path_valid_tgt: path_to_europarl/et-en/valid.et-en.et.sp
    transforms: [filtertoolong]
  # GPU 1:3
  train_fi-en:
    src_tgt: fi-en
    enc_sharing_group: [fi]
    dec_sharing_group: [en]
    node_gpu: "1:3"
    path_src: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_tgt: path_to_europarl/fi-en/train.fi-en.en.sp
    path_valid_src: path_to_europarl/fi-en/valid.fi-en.fi.sp
    path_valid_tgt: path_to_europarl/fi-en/valid.fi-en.en.sp
    transforms: [filtertoolong]
  train_fi-fi:
    src_tgt: fi-fi
    enc_sharing_group: [fi]
    dec_sharing_group: [fi]
    node_gpu: "1:3"
    path_src: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_tgt: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_valid_src: path_to_europarl/fi-en/valid.fi-en.fi.sp
    path_valid_tgt: path_to_europarl/fi-en/valid.fi-en.fi.sp
    transforms: [filtertoolong, denoising]
  train_en-fi:
    src_tgt: en-fi
    enc_sharing_group: [en]
    dec_sharing_group: [fi]
    node_gpu: "1:3"
    path_src: path_to_europarl/fi-en/train.fi-en.en.sp
    path_tgt: path_to_europarl/fi-en/train.fi-en.fi.sp
    path_valid_src: path_to_europarl/fi-en/valid.fi-en.en.sp
    path_valid_tgt: path_to_europarl/fi-en/valid.fi-en.fi.sp
    transforms: [filtertoolong]
  # GPU 2:0
  train_fr-en:
    src_tgt: fr-en
    enc_sharing_group: [fr]
    dec_sharing_group: [en]
    node_gpu: "2:0"
    path_src: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_tgt: path_to_europarl/fr-en/train.fr-en.en.sp
    path_valid_src: path_to_europarl/fr-en/valid.fr-en.fr.sp
    path_valid_tgt: path_to_europarl/fr-en/valid.fr-en.en.sp
    transforms: [filtertoolong]
  train_fr-fr:
    src_tgt: fr-fr
    enc_sharing_group: [fr]
    dec_sharing_group: [fr]
    node_gpu: "2:0"
    path_src: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_tgt: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_valid_src: path_to_europarl/fr-en/valid.fr-en.fr.sp
    path_valid_tgt: path_to_europarl/fr-en/valid.fr-en.fr.sp
    transforms: [filtertoolong, denoising]
  train_en-fr:
    src_tgt: en-fr
    enc_sharing_group: [en]
    dec_sharing_group: [fr]
    node_gpu: "2:0"
    path_src: path_to_europarl/fr-en/train.fr-en.en.sp
    path_tgt: path_to_europarl/fr-en/train.fr-en.fr.sp
    path_valid_src: path_to_europarl/fr-en/valid.fr-en.en.sp
    path_valid_tgt: path_to_europarl/fr-en/valid.fr-en.fr.sp
    transforms: [filtertoolong]  
  # GPU 2:1
  train_hu-en:
    src_tgt: hu-en
    enc_sharing_group: [hu]
    dec_sharing_group: [en]
    node_gpu: "2:1"
    path_src: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_tgt: path_to_europarl/hu-en/train.hu-en.en.sp
    path_valid_src: path_to_europarl/hu-en/valid.hu-en.hu.sp
    path_valid_tgt: path_to_europarl/hu-en/valid.hu-en.en.sp
    transforms: [filtertoolong]
  train_hu-hu:
    src_tgt: hu-hu
    enc_sharing_group: [hu]
    dec_sharing_group: [hu]
    node_gpu: "2:1"
    path_src: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_tgt: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_valid_src: path_to_europarl/hu-en/valid.hu-en.hu.sp
    path_valid_tgt: path_to_europarl/hu-en/valid.hu-en.hu.sp
    transforms: [filtertoolong, denoising]
  train_en-hu:
    src_tgt: en-hu
    enc_sharing_group: [en]
    dec_sharing_group: [hu]
    node_gpu: "2:1"
    path_src: path_to_europarl/hu-en/train.hu-en.en.sp
    path_tgt: path_to_europarl/hu-en/train.hu-en.hu.sp
    path_valid_src: path_to_europarl/hu-en/valid.hu-en.en.sp
    path_valid_tgt: path_to_europarl/hu-en/valid.hu-en.hu.sp
    transforms: [filtertoolong]
  # GPU 2:2
  train_it-en:
    src_tgt: it-en
    enc_sharing_group: [it]
    dec_sharing_group: [en]
    node_gpu: "2:2"
    path_src: path_to_europarl/it-en/train.it-en.it.sp
    path_tgt: path_to_europarl/it-en/train.it-en.en.sp
    path_valid_src: path_to_europarl/it-en/valid.it-en.it.sp
    path_valid_tgt: path_to_europarl/it-en/valid.it-en.en.sp
    transforms: [filtertoolong]
  train_it-it:
    src_tgt: it-it
    enc_sharing_group: [it]
    dec_sharing_group: [it]
    node_gpu: "2:2"
    path_src: path_to_europarl/it-en/train.it-en.it.sp
    path_tgt: path_to_europarl/it-en/train.it-en.it.sp
    path_valid_src: path_to_europarl/it-en/valid.it-en.it.sp
    path_valid_tgt: path_to_europarl/it-en/valid.it-en.it.sp
    transforms: [filtertoolong, denoising]
  train_en-it:
    src_tgt: en-it
    enc_sharing_group: [en]
    dec_sharing_group: [it]
    node_gpu: "2:2"
    path_src: path_to_europarl/it-en/train.it-en.en.sp
    path_tgt: path_to_europarl/it-en/train.it-en.it.sp
    path_valid_src: path_to_europarl/it-en/valid.it-en.en.sp
    path_valid_tgt: path_to_europarl/it-en/valid.it-en.it.sp
    transforms: [filtertoolong]
  # GPU 2:3
  train_lt-en:
    src_tgt: lt-en
    enc_sharing_group: [lt]
    dec_sharing_group: [en]
    node_gpu: "2:3"
    path_src: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_tgt: path_to_europarl/lt-en/train.lt-en.en.sp
    path_valid_src: path_to_europarl/lt-en/valid.lt-en.lt.sp
    path_valid_tgt: path_to_europarl/lt-en/valid.lt-en.en.sp
    transforms: [filtertoolong]
  train_lt-lt:
    src_tgt: lt-lt
    enc_sharing_group: [lt]
    dec_sharing_group: [lt]
    node_gpu: "2:3"
    path_src: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_tgt: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_valid_src: path_to_europarl/lt-en/valid.lt-en.lt.sp
    path_valid_tgt: path_to_europarl/lt-en/valid.lt-en.lt.sp
    transforms: [filtertoolong, denoising]
  train_en-lt:
    src_tgt: en-lt
    enc_sharing_group: [en]
    dec_sharing_group: [lt]
    node_gpu: "2:3"
    path_src: path_to_europarl/lt-en/train.lt-en.en.sp
    path_tgt: path_to_europarl/lt-en/train.lt-en.lt.sp
    path_valid_src: path_to_europarl/lt-en/valid.lt-en.en.sp
    path_valid_tgt: path_to_europarl/lt-en/valid.lt-en.lt.sp
    transforms: [filtertoolong]
  # GPU 3:0
  train_lv-en:
    src_tgt: lv-en
    enc_sharing_group: [lv]
    dec_sharing_group: [en]
    node_gpu: "3:0"
    path_src: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_tgt: path_to_europarl/lv-en/train.lv-en.en.sp
    path_valid_src: path_to_europarl/lv-en/valid.lv-en.lv.sp
    path_valid_tgt: path_to_europarl/lv-en/valid.lv-en.en.sp
    transforms: [filtertoolong]
  train_lv-lv:
    src_tgt: lv-lv
    enc_sharing_group: [lv]
    dec_sharing_group: [lv]
    node_gpu: "3:0"
    path_src: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_tgt: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_valid_src: path_to_europarl/lv-en/valid.lv-en.lv.sp
    path_valid_tgt: path_to_europarl/lv-en/valid.lv-en.lv.sp
    transforms: [filtertoolong, denoising]
  train_en-lv:
    src_tgt: en-lv
    enc_sharing_group: [en]
    dec_sharing_group: [lv]
    node_gpu: "3:0"
    path_src: path_to_europarl/lv-en/train.lv-en.en.sp
    path_tgt: path_to_europarl/lv-en/train.lv-en.lv.sp
    path_valid_src: path_to_europarl/lv-en/valid.lv-en.en.sp
    path_valid_tgt: path_to_europarl/lv-en/valid.lv-en.lv.sp
    transforms: [filtertoolong]
  # GPU 3:1
  train_nl-en:
    src_tgt: nl-en
    enc_sharing_group: [nl]
    dec_sharing_group: [en]
    node_gpu: "3:1"
    path_src: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_tgt: path_to_europarl/nl-en/train.nl-en.en.sp
    path_valid_src: path_to_europarl/nl-en/valid.nl-en.nl.sp
    path_valid_tgt: path_to_europarl/nl-en/valid.nl-en.en.sp
    transforms: [filtertoolong]
  train_nl-nl:
    src_tgt: nl-nl
    enc_sharing_group: [nl]
    dec_sharing_group: [nl]
    node_gpu: "3:1"
    path_src: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_tgt: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_valid_src: path_to_europarl/nl-en/valid.nl-en.nl.sp
    path_valid_tgt: path_to_europarl/nl-en/valid.nl-en.nl.sp
    transforms: [filtertoolong, denoising]
  train_en-nl:
    src_tgt: en-nl
    enc_sharing_group: [en]
    dec_sharing_group: [nl]
    node_gpu: "3:1"
    path_src: path_to_europarl/nl-en/train.nl-en.en.sp
    path_tgt: path_to_europarl/nl-en/train.nl-en.nl.sp
    path_valid_src: path_to_europarl/nl-en/valid.nl-en.en.sp
    path_valid_tgt: path_to_europarl/nl-en/valid.nl-en.nl.sp
    transforms: [filtertoolong]
  # GPU 3:2
  train_pl-en:
    src_tgt: pl-en
    enc_sharing_group: [pl]
    dec_sharing_group: [en]
    node_gpu: "3:2"
    path_src: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_tgt: path_to_europarl/pl-en/train.pl-en.en.sp
    path_valid_src: path_to_europarl/pl-en/valid.pl-en.pl.sp
    path_valid_tgt: path_to_europarl/pl-en/valid.pl-en.en.sp
    transforms: [filtertoolong]
  train_pl-pl:
    src_tgt: pl-pl
    enc_sharing_group: [pl]
    dec_sharing_group: [pl]
    node_gpu: "3:2"
    path_src: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_tgt: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_valid_src: path_to_europarl/pl-en/valid.pl-en.pl.sp
    path_valid_tgt: path_to_europarl/pl-en/valid.pl-en.pl.sp
    transforms: [filtertoolong, denoising]
  train_en-pl:
    src_tgt: en-pl
    enc_sharing_group: [en]
    dec_sharing_group: [pl]
    node_gpu: "3:2"
    path_src: path_to_europarl/pl-en/train.pl-en.en.sp
    path_tgt: path_to_europarl/pl-en/train.pl-en.pl.sp
    path_valid_src: path_to_europarl/pl-en/valid.pl-en.en.sp
    path_valid_tgt: path_to_europarl/pl-en/valid.pl-en.pl.sp
    transforms: [filtertoolong]
  # GPU 3:3
  train_pt-en:
    src_tgt: pt-en
    enc_sharing_group: [pt]
    dec_sharing_group: [en]
    node_gpu: "3:3"
    path_src: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_tgt: path_to_europarl/pt-en/train.pt-en.en.sp
    path_valid_src: path_to_europarl/pt-en/valid.pt-en.pt.sp
    path_valid_tgt: path_to_europarl/pt-en/valid.pt-en.en.sp
    transforms: [filtertoolong]
  train_pt-pt:
    src_tgt: pt-pt
    enc_sharing_group: [pt]
    dec_sharing_group: [pt]
    node_gpu: "3:3"
    path_src: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_tgt: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_valid_src: path_to_europarl/pt-en/valid.pt-en.pt.sp
    path_valid_tgt: path_to_europarl/pt-en/valid.pt-en.pt.sp
    transforms: [filtertoolong, denoising]
  train_en-pt:
    src_tgt: en-pt
    enc_sharing_group: [en]
    dec_sharing_group: [pt]
    node_gpu: "3:3"
    path_src: path_to_europarl/pt-en/train.pt-en.en.sp
    path_tgt: path_to_europarl/pt-en/train.pt-en.pt.sp
    path_valid_src: path_to_europarl/pt-en/valid.pt-en.en.sp
    path_valid_tgt: path_to_europarl/pt-en/valid.pt-en.pt.sp
    transforms: [filtertoolong]
  # GPU 4:0
  train_ro-en:
    src_tgt: ro-en
    enc_sharing_group: [ro]
    dec_sharing_group: [en]
    node_gpu: "4:0"
    path_src: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_tgt: path_to_europarl/ro-en/train.ro-en.en.sp
    path_valid_src: path_to_europarl/ro-en/valid.ro-en.ro.sp
    path_valid_tgt: path_to_europarl/ro-en/valid.ro-en.en.sp
    transforms: [filtertoolong]
  train_ro-ro:
    src_tgt: ro-ro
    enc_sharing_group: [ro]
    dec_sharing_group: [ro]
    node_gpu: "4:0"
    path_src: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_tgt: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_valid_src: path_to_europarl/ro-en/valid.ro-en.ro.sp
    path_valid_tgt: path_to_europarl/ro-en/valid.ro-en.ro.sp
    transforms: [filtertoolong, denoising]
  train_en-ro:
    src_tgt: en-ro
    enc_sharing_group: [en]
    dec_sharing_group: [ro]
    node_gpu: "4:0"
    path_src: path_to_europarl/ro-en/train.ro-en.en.sp
    path_tgt: path_to_europarl/ro-en/train.ro-en.ro.sp
    path_valid_src: path_to_europarl/ro-en/valid.ro-en.en.sp
    path_valid_tgt: path_to_europarl/ro-en/valid.ro-en.ro.sp
    transforms: [filtertoolong]
  # GPU 4:1
  train_sk-en:
    src_tgt: sk-en
    enc_sharing_group: [sk]
    dec_sharing_group: [en]
    node_gpu: "4:1"
    path_src: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_tgt: path_to_europarl/sk-en/train.sk-en.en.sp
    path_valid_src: path_to_europarl/sk-en/valid.sk-en.sk.sp
    path_valid_tgt: path_to_europarl/sk-en/valid.sk-en.en.sp
    transforms: [filtertoolong]
  train_sk-sk:
    src_tgt: sk-sk
    enc_sharing_group: [sk]
    dec_sharing_group: [sk]
    node_gpu: "4:1"
    path_src: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_tgt: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_valid_src: path_to_europarl/sk-en/valid.sk-en.sk.sp
    path_valid_tgt: path_to_europarl/sk-en/valid.sk-en.sk.sp
    transforms: [filtertoolong, denoising]
  train_en-sk:
    src_tgt: en-sk
    enc_sharing_group: [en]
    dec_sharing_group: [sk]
    node_gpu: "4:1"
    path_src: path_to_europarl/sk-en/train.sk-en.en.sp
    path_tgt: path_to_europarl/sk-en/train.sk-en.sk.sp
    path_valid_src: path_to_europarl/sk-en/valid.sk-en.en.sp
    path_valid_tgt: path_to_europarl/sk-en/valid.sk-en.sk.sp
    transforms: [filtertoolong]
  # GPU 4:2
  train_sl-en:
    src_tgt: sl-en
    enc_sharing_group: [sl]
    dec_sharing_group: [en]
    node_gpu: "4:2"
    path_src: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_tgt: path_to_europarl/sl-en/train.sl-en.en.sp
    path_valid_src: path_to_europarl/sl-en/valid.sl-en.sl.sp
    path_valid_tgt: path_to_europarl/sl-en/valid.sl-en.en.sp
    transforms: [filtertoolong]
  train_sl-sl:
    src_tgt: sl-sl
    enc_sharing_group: [sl]
    dec_sharing_group: [sl]
    node_gpu: "4:2"
    path_src: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_tgt: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_valid_src: path_to_europarl/sl-en/valid.sl-en.sl.sp
    path_valid_tgt: path_to_europarl/sl-en/valid.sl-en.sl.sp
    transforms: [filtertoolong, denoising]
  train_en-sl:
    src_tgt: en-sl
    enc_sharing_group: [en]
    dec_sharing_group: [sl]
    node_gpu: "4:2"
    path_src: path_to_europarl/sl-en/train.sl-en.en.sp
    path_tgt: path_to_europarl/sl-en/train.sl-en.sl.sp
    path_valid_src: path_to_europarl/sl-en/valid.sl-en.en.sp
    path_valid_tgt: path_to_europarl/sl-en/valid.sl-en.sl.sp
    transforms: [filtertoolong]
  # GPU 4:3
  train_sv-en:
    src_tgt: sv-en
    enc_sharing_group: [sv]
    dec_sharing_group: [en]
    node_gpu: "4:3"
    path_src: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_tgt: path_to_europarl/sv-en/train.sv-en.en.sp
    path_valid_src: path_to_europarl/sv-en/valid.sv-en.sv.sp
    path_valid_tgt: path_to_europarl/sv-en/valid.sv-en.en.sp
    transforms: [filtertoolong]
  train_sv-sv:
    src_tgt: sv-sv
    enc_sharing_group: [sv]
    dec_sharing_group: [sv]
    node_gpu: "4:3"
    path_src: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_tgt: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_valid_src: path_to_europarl/sv-en/valid.sv-en.sv.sp
    path_valid_tgt: path_to_europarl/sv-en/valid.sv-en.sv.sp
    transforms: [filtertoolong, denoising]
  train_en-sv:
    src_tgt: en-sv
    enc_sharing_group: [en]
    dec_sharing_group: [sv]
    node_gpu: "4:3"
    path_src: path_to_europarl/sv-en/train.sv-en.en.sp
    path_tgt: path_to_europarl/sv-en/train.sv-en.sv.sp
    path_valid_src: path_to_europarl/sv-en/valid.sv-en.en.sp
    path_valid_tgt: path_to_europarl/sv-en/valid.sv-en.sv.sp
    transforms: [filtertoolong]

        
### Transform related opts:
#### Filter
src_seq_length: 200
tgt_seq_length: 200
#### Bart
src_subword_type: sentencepiece
tgt_subword_type: sentencepiece
mask_ratio: 0.2
replace_length: 1

# silently ignore empty lines in the data
skip_empty_level: silent

batch_size: 4096
batch_type: tokens
normalization: tokens
valid_batch_size: 4096
max_generator_batches: 2
src_vocab_size: 100000
tgt_vocab_size: 100000
encoder_type: transformer
decoder_type: transformer
model_dim: 512
transformer_ff: 2048
heads: 8
enc_layers: [6]
dec_layers: [6]
dropout: 0.1
label_smoothing: 0.1
param_init: 0.0
param_init_glorot: true
position_encoding: true
valid_steps: 10000
warmup_steps: 10000
report_every: 100
save_checkpoint_steps: 50000
keep_checkpoint: -1
accum_count: 1
optim: adafactor
decay_method: none
learning_rate: 3.0
max_grad_norm: 0.0
seed: 3435
model_type: text
save_all_gpus: false

n_nodes: 5
world_size: 20
gpu_ranks: [0, 1, 2, 3]

early_stopping: 5
early_stopping_criteria: accuracy
```
</details>


### Data Configuration:
- Vocabularies for the source and target languages is need to be specified. In the example, we used a shared vocabulary.
- Specifies options related to data transformation, including filtering and BART-specific denoising parameters.

### Task Configuration:
- Translation tasks are defined in this section, such as `bg-en` for Bulgarian to English translation. 
- Each task includes details such as source and target file paths, sharing groups, GPU assignments, and data transforms.
- For GPU assignments, the task defines the ranks of nodes and GPUs. For example, `4:0` indicates the first GPU on the fifth node.

### Training Configuration:
- Batch size, normalization, and other training parameters are set.
- Model parameters such as dimensions, transformer layers, dropout, label smoothing, and more are specified.
- The training uses the Adafactor optimizer with a learning rate of 3.0 and no gradient clipping.
- Early stopping is enabled with a criterion of accuracy and a patience of 5 steps.
- The training process is distributed across 4 GPUs (`world_size: 4`, `gpu_ranks: [0, 1, 2, 3]`) on a single node (`node_rank: 0`) for single node job. For the 5-node job, job is distributed across 20 GPUs. 

## Step 4: Train your MAMMOTH model

Finally, we can start the training process now. Here we provide an example script that sets several environment variables, creates necessary directories, and then runs a training job for a MAMMOTH machine translation model. 

```bash
export PYTHONUSERBASE=/path_to_your_env/mammoth/

# pointer to codebase
export MAMMOTH=/path_to_codebase/mammoth

# pointer to config file
export CONFIG_DIR=path_to_europarl/config

# pointer to slurm multinode wrapper.
export SCRIPT_DIR=path_to_europarl/scripts/

# info for model and log saving
export SAVE_DIR=your_path/models/europarl
export LOG_DIR=${SAVE_DIR}/logs
export EXP_ID=example-1-node

mkdir -p  ${SAVE_DIR}/{logs,models}

srun ${SCRIPT_DIR}/wrapper.sh -u ${MAMMOTH}/train.py \
    -config ${CONFIG_DIR}/europarl-1node-4gpu.yml \
    -save_model ${SAVE_DIR}/models/${EXP_ID} \
    -master_port 9973 \
    -tensorboard -tensorboard_log_dir ${LOG_DIR}/${EXP_ID}
```


### Environment Variable Setup:
   - `PYTHONUSERBASE`: Specifies the base directory for Python user-specific packages. You can also specify the python environment in your favorite way and check the installation guide for more information.
   - `MAMMOTH`: Points to the codebase directory for a project named "mammoth."
   - `CONFIG_DIR`: Points to a directory containing configuration files.
   - `SCRIPT_DIR`: Points to a directory containing Slurm multinode wrapper scripts.
   - `SAVE_DIR`: Specifies the base directory for saving model-related files.
   - `LOG_DIR`: Specifies the directory for saving logs related to the model training.
   - `EXP_ID`: Represents an experiment identifier, set to "example-1-node."

### Directory Creation:
   - Creates the "logs" and "models" directories inside `SAVE_DIR` if they do not already exist. You will find the logs and saved models there.

### Training Job Submission:
   - We utilize Slurm for resource allocation. `srun`: Initiates a Slurm job.
   - `${SCRIPT_DIR}/wrapper.sh`: Calls a wrapper script for managing Slurm settings, monitoring GPU usage, and etc.
An example of wrapper script can be:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
nvidia-smi dmon -s mu -d 5 -o TD > "${LOG_DIR}/gpu_load-${EXP_ID}-${PPID}.log" &
echo python -u "$@" --node_rank $SLURM_NODEID
python -u "$@" --node_rank $SLURM_NODEID
```
   - `-u ${MAMMOTH}/train.py`: Specifies the Python script for training, located in the "mammoth" codebase.
   - `-config ${CONFIG_DIR}/europarl-1node-4gpu.yml`: Specifies the configuration file for the training job.
   - `-save_model ${SAVE_DIR}/models/${EXP_ID}`: Specifies the directory to save the trained model.
   - `-master_port 9973`: Specifies the master port for communication.
   - `-tensorboard -tensorboard_log_dir ${LOG_DIR}/${EXP_ID}`: Enables TensorBoard logging and specifies the directory for TensorBoard logs.


Hooray! Take a moment to celebrate the progress you've made. Wait for hours and the model training should be completed soon.