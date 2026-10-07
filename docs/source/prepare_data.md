# Prepare Data

MAMMOTH trains on plain parallel text: one sentence per line, with the source file and the target file line-aligned. Before following this page, make sure that you have [installed](install) MAMMOTH, which includes the dependencies required below.

There are two ways to handle tokenization:

- **Option 1: HuggingFace tokenizer (recommended).** MAMMOTH tokenizes on the fly, so there is nothing to pre-tokenize. You only need to check that your parallel files are aligned and split off a validation set.
- **Option 2: SentencePiece (legacy).** You tokenize the data yourself with SentencePiece before training.


## Option 1: HuggingFace tokenizer (recommended)

With a HuggingFace tokenizer (a `.json` file in your vocab config), MAMMOTH tokenizes the raw text as it loads the data. You do not need to tokenize or encode your corpus. What you need is:

1. aligned raw text for each language pair,
2. a separate validation set for each task (`path_valid_src` / `path_valid_tgt`),
3. a trained tokenizer `.json`.

The steps below use the [Europarl](https://www.statmt.org/europarl/) corpus as an example.

### Step 1: Download the data

[Europarl parallel corpus](https://www.statmt.org/europarl/) is a multilingual resource extracted from European Parliament proceedings and contains texts in 21 European languages. Download the Release v7 - a further expanded and improved version of the Europarl corpus on 15 May 2012 - from the original website or download the processed data by us:
```bash
wget https://mammoth101.a3s.fi/europarl.tar.gz
mkdir europarl_data
tar -xvzf europarl.tar.gz -C europarl_data
```
Note that the extracted dataset will require around 30GB of disk space. Alternatively, you can only download the data for the three example languages (666M).
```bash
wget https://mammoth101.a3s.fi/europarl-3langs.tar.gz
mkdir europarl_data
tar -xvzf europarl-3langs.tar.gz -C europarl_data
```

### Step 2: Check, clean and split train/valid

Save the following script as `prepare_parallel.py`. It does three things, in this order:

1. **Check alignment.** Each source file must have exactly as many lines as its target file. If the counts differ, sentences get paired with the wrong translations and training degrades without any error message, so the script stops with an error.
2. **Clean (Optional)** It strips leading and trailing whitespace and collapses runs of whitespace, then drops pairs where either side is empty and drops exact duplicate pairs. Removing duplicates before the split also keeps the same sentence pair from landing in both the training and the validation set. Invalid UTF-8 bytes are replaced with `�`, so the output is always valid UTF-8. *Note: this preprocessing is for demonstration purposes only. Please apply your own data-cleaning procedures in your actual implementation.*
3. **Shuffle and split.** MAMMOTH reads your training files in order and does not shuffle them for you, so the script shuffles the pairs before holding out `--valid_size` of them for validation. Otherwise the validation set would be one contiguous chunk of the corpus.

It writes `train.<lang>` and `val.<lang>` files for both sides.

```python
"""Check that a parallel corpus is line-aligned, clean it, then shuffle and split it into train/valid files."""
import argparse
import os
import random


def read_lines(path):
    # errors="replace": invalid UTF-8 becomes U+FFFD here instead of crashing training later
    with open(path, encoding="utf-8", errors="replace") as f:
        return list(f)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--src", required=True, help="source-side text file")
parser.add_argument("--tgt", required=True, help="target-side text file")
parser.add_argument("--src_lang", required=True, help="used in the output file names, e.g. bg")
parser.add_argument("--tgt_lang", required=True, help="used in the output file names, e.g. en")
parser.add_argument("--out_dir", required=True)
parser.add_argument("--valid_size", type=int, default=1000, help="number of sentence pairs held out for validation")
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

if args.src_lang == args.tgt_lang:
    raise SystemExit("--src_lang and --tgt_lang must differ, otherwise the output files overwrite each other")

src, tgt = read_lines(args.src), read_lines(args.tgt)

# 1. both sides must have the same number of lines
if len(src) != len(tgt):
    raise SystemExit(f"MISMATCH: {args.src} has {len(src)} lines, {args.tgt} has {len(tgt)} lines")
print(f"OK: {len(src)} aligned sentence pairs")

# 2. clean: normalize whitespace, then drop pairs with an empty side and exact duplicate pairs
pairs = [(" ".join(s.split()), " ".join(t.split())) for s, t in zip(src, tgt)]
n_total = len(pairs)
pairs = [(s, t) for s, t in pairs if s and t]
n_empty = n_total - len(pairs)
pairs = list(dict.fromkeys(pairs))  # keeps the first occurrence of each pair
print(f"cleaned: dropped {n_empty} pairs with an empty side and {n_total - n_empty - len(pairs)} duplicates")
if args.valid_size >= len(pairs):
    raise SystemExit(f"--valid_size {args.valid_size} leaves no training data ({len(pairs)} pairs after cleaning)")

# 3. shuffle the pairs (MAMMOTH reads the files in order) and split
random.Random(args.seed).shuffle(pairs)
splits = {"val": pairs[:args.valid_size], "train": pairs[args.valid_size:]}

os.makedirs(args.out_dir, exist_ok=True)
for name, rows in splits.items():
    for lang, side in ((args.src_lang, 0), (args.tgt_lang, 1)):
        with open(os.path.join(args.out_dir, f"{name}.{lang}"), "w", encoding="utf-8") as f:
            f.writelines(row[side] + "\n" for row in rows)
    print(f"{name}: {len(rows)} pairs")
```

Then run it once per language pair. Here for Bulgarian-English and Czech-English:

```bash
for lang in bg cs; do
    python prepare_parallel.py \
        --src europarl_data/europarl/${lang}-en/europarl-v7.${lang}-en.${lang} \
        --tgt europarl_data/europarl/${lang}-en/europarl-v7.${lang}-en.en \
        --src_lang ${lang} --tgt_lang en \
        --out_dir europarl_data/split/${lang}-en \
        --valid_size 1000
done
```

This produces `europarl_data/split/bg-en/{train,val}.{bg,en}` and the same for `cs-en`. The script prints how many pairs each cleaning step dropped, so check that the numbers look reasonable for your corpus. It holds both files in memory, which is fine for a corpus like Europarl (a few million lines per pair). For much larger corpora, use a streaming tool instead.

The script only does basic cleaning. For further filtering, for example by length or length ratio, MAMMOTH has on-the-fly filter transforms (`filtertoolong`, `filterwordratio`, `filterrepetitions`, `filterterminalpunct` and `filternonzeronumerals`) that you add to a task's `transforms` list.

### Step 3: Train a tokenizer

Train a BPE tokenizer on the **training** files with `mammoth/bin/build_vocab.py` (run from the root of the MAMMOTH repository). It needs the `tokenizers` and `transformers` packages, both listed in `requirements.txt`.

```bash
python mammoth/bin/build_vocab.py \
    --input_file europarl_data/split/bg-en/train.bg europarl_data/split/bg-en/train.en \
                 europarl_data/split/cs-en/train.cs europarl_data/split/cs-en/train.en \
    --output_dir tokenizer \
    --vocab_size 32000
```

`--input_file` accepts any number of files, so one command trains a single tokenizer shared by all your languages. The tokenizer is saved to `tokenizer/tokenizer.json`. The special tokens (`</s>`, `<pad>`, `<s>`, `<unk>`, `<mask>`) are fixed for MAMMOTH, and BOS/EOS are added by MAMMOTH during data loading, not by the tokenizer. See the [HF tokenizers guide](HF_TOKENIZERS.md) for details and for using a pretrained tokenizer instead.

### Step 4: Point the config at your files

```yaml
use_hf_tokenizer: true

src_vocab:
  bg: tokenizer/tokenizer.json
tgt_vocab:
  en: tokenizer/tokenizer.json

tasks:
  train_bg-en:
    src_tgt: bg-en
    path_src: europarl_data/split/bg-en/train.bg
    path_tgt: europarl_data/split/bg-en/train.en
    path_valid_src: europarl_data/split/bg-en/val.bg
    path_valid_tgt: europarl_data/split/bg-en/val.en
    # ... plus the usual task keys (weight, node_gpu, sharing groups, ...)
```

Transforms such as `filtertoolong` still work with this setup, since they run on the fly too. If you would rather tokenize once up front, see the [indexed dataset guide](INDEXED_DATASET_GUIDE.md).


## Option 2: SentencePiece (legacy)

SentencePiece is the older way of tokenizing in MAMMOTH. It is kept here for models that already use a SentencePiece vocabulary. The [MAMMOTH 101](examples/train_mammoth_101.md) example now uses Hugging Face tokenizers (Option 1). For new projects, use Option 1.

With SentencePiece, the data is tokenized ahead of time with the scripts below.

### Europarl

Download the raw Europarl data as in Option 1, Step 1. Then:

#### Step 1: Download the SentencePiece model

We use a SentencePiece tokenizer trained on OPUS Tatoeba Challenge data with 64k vocabulary size. Download the SentencePiece model and the vocabulary:
```bash
# Download the SentencePiece model
wget https://mammoth101.a3s.fi/opusTC.mul.64k.spm
# Download the vocabulary
wget https://mammoth101.a3s.fi/opusTC.mul.vocab.onmt

mkdir vocab
mv opusTC.mul.64k.spm vocab/.
mv opusTC.mul.vocab.onmt vocab/.
```
If you would like to create and use a custom sentencepiece tokenizer, take a look at the OPUS tutorial below.

#### Step 2: Tokenization
Then, read parallel text data, processes it, and generates output files for training and validation sets. 
Here's a high-level summary of the main processing steps. For each language in 'langs,' 
- read parallel data files.
- clean the data by removing empty lines.
- shuffle the data randomly.
- tokenizes the text using SentencePiece and writes the tokenized data to separate output files for training and validation sets.

You're free to skip this step if you directly download the processed data.

```python
import random
import pathlib

import tqdm
import sentencepiece as sp

langs = ["bg", "cs"]

sp_path = 'vocab/opusTC.mul.64k.spm'
spm = sp.SentencePieceProcessor(model_file=sp_path)

input_dir = 'europarl_data/europarl'
output_dir = 'europarl_data/encoded'

for lang in tqdm.tqdm(langs):
    en_side_in = f'{input_dir}/{lang}-en/europarl-v7.{lang}-en.en'
    xx_side_in = f'{input_dir}/{lang}-en/europarl-v7.{lang}-en.{lang}'
    with open(xx_side_in) as xx_stream, open(en_side_in) as en_stream:
        data = zip(map(str.strip, xx_stream), map(str.strip, en_stream))
        data = [(xx, en) for xx, en in tqdm.tqdm(data, leave=False, desc=f'read {lang}') if xx and en] # drop empty lines
        random.shuffle(data)
    pathlib.Path(output_dir).mkdir(exist_ok=True) 
    en_side_out = f'{output_dir}/valid.{lang}-en.en.sp'
    xx_side_out = f'{output_dir}/valid.{lang}-en.{lang}.sp'
    with open(xx_side_out, 'w') as xx_stream, open(en_side_out, 'w') as en_stream:
        for xx, en in tqdm.tqdm(data[:1000], leave=False, desc=f'valid {lang}'):
            print(*spm.encode(xx, out_type=str), file=xx_stream)
            print(*spm.encode(en, out_type=str), file=en_stream)
    en_side_out = f'{output_dir}/train.{lang}-en.en.sp'
    xx_side_out = f'{output_dir}/train.{lang}-en.{lang}.sp'
    with open(xx_side_out, 'w') as xx_stream, open(en_side_out, 'w') as en_stream:
        for xx, en in tqdm.tqdm(data[1000:], leave=False, desc=f'train {lang}'):
            print(*spm.encode(xx, out_type=str), file=xx_stream)
            print(*spm.encode(en, out_type=str), file=en_stream)
```

The script will produce encoded datasets in `europarl_data/encoded` that you can further use for the training.

### UNPC
[UNPC](https://opus.nlpl.eu/UNPC/corpus/version/UNPC) consists of manually translated UN documents from the last 25 years (1990 to 2014) for the six official UN languages, Arabic, Chinese, English, French, Russian, and Spanish. 
We preprocess the data. You can download the processed data by:
```bash
wget https://mammoth-share.a3s.fi/unpc.tar
```
Or you can use the scripts provided by the tarball to process the data yourself. 

For references, please cite this reference: Ziemski, M., Junczys-Dowmunt, M., and Pouliquen, B., (2016), The United Nations Parallel Corpus, Language Resources and Evaluation (LREC’16), Portorož, Slovenia, May 2016.


### OPUS 100

In this guideline, we will also create our custom sentencepiece tokenizer.

To do that, you will also need to compile a sentencepiece installation in your environment (not just pip install). 
Follow the instructions on [sentencepiece github](https://github.com/google/sentencepiece?tab=readme-ov-file#build-and-install-sentencepiece-command-line-tools-from-c-source).

After that, download the opus 100 dataset from [OPUS 100](https://opus.nlpl.eu/legacy/opus-100.php)

#### Step 1: Set relevant paths, variables and download

```
SP_PATH=your/sentencepiece/path/build/src
DATA_PATH=your/path/to/save/dataset
# Download the default datasets into the $DATA_PATH; mkdir if it doesn't exist
mkdir -p $DATA_PATH

CUR_DIR=$(pwd)

# set vocabulary size and language pairs
vocab_sizes=(32000 16000 8000 4000 2000 1000)
input_sentence_size=10000000

cd $DATA_PATH
echo "Downloading and extracting Opus100"
wget -q --trust-server-names https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz
tar -xzvf opus-100-corpus-v1.0.tar.gz
cd $CUR_DIR

language_pairs=( $( ls $DATA_PATH/opus-100-corpus/v1.0/supervised/ ) )
```

#### Step 2: Train SentencePiece models and get vocabs

Starting from here, original files are supposed to be in `$DATA_PATH`

```
echo "$0: Training SentencePiece models"
rm -f $DATA_PATH/train.txt
rm -f $DATA_PATH/train.en.txt
for lp in "${language_pairs[@]}"
do
IFS=- read sl tl <<< $lp
if [[ $sl = "en" ]]
then
other_lang=$tl
else
other_lang=$sl
fi
# train the SentencePiece model over the language other than english
sort -u $DATA_PATH/opus-100-corpus/v1.0/supervised/$lp/opus.$lp-train.$other_lang | shuf > $DATA_PATH/train.txt
for vocab_size in "${vocab_sizes[@]}"
do
echo "Training SentencePiece model for $other_lang with vocab size $vocab_size"
cd $SP_PATH
./spm_train --input=$DATA_PATH/train.txt \
            --model_prefix=$DATA_PATH/opus.$other_lang \
            --vocab_size=$vocab_size --character_coverage=0.98 \
            --input_sentence_size=$input_sentence_size --shuffle_input_sentence=true # to use a subset of sentences sampled from the entire training set
cd $CUR_DIR
if [ -f "$DATA_PATH"/opus."$other_lang".vocab ]
then
    # get vocab in onmt format
    cut -f 1 "$DATA_PATH"/opus."$other_lang".vocab > "$DATA_PATH"/opus."$other_lang".vocab.onmt
    break
fi
done
rm $DATA_PATH/train.txt
# append the english data to a file
cat $DATA_PATH/opus-100-corpus/v1.0/supervised/$lp/opus.$lp-train.en >> $DATA_PATH/train.en.txt

 # train the SentencePiece model for english
 echo "Training SentencePiece model for en"
 sort -u $DATA_PATH/train.en.txt | shuf -n $input_sentence_size > $DATA_PATH/train.txt
 rm $DATA_PATH/train.en.txt
 cd $SP_PATH
 ./spm_train --input=$DATA_PATH/train.txt --model_prefix=$DATA_PATH/opus.en \
             --vocab_size=${vocab_sizes[0]} --character_coverage=0.98 \
             --input_sentence_size=$input_sentence_size --shuffle_input_sentence=true # to use a subset of sentences sampled from the entire training set
# Other options to consider:
# --max_sentence_length=  # to set max length when filtering sentences
# --train_extremely_large_corpus=true
 cd $CUR_DIR
 rm $DATA_PATH/train.txt
fi
```

#### Step 3: Parse train, valid and test sets for supervised translation directions
```
mkdir -p $DATA_PATH/supervised
for lp in "${language_pairs[@]}"
do
mkdir -p $DATA_PATH/supervised/$lp
IFS=- read sl tl <<< $lp

echo "$lp: parsing train data"
dir=$DATA_PATH/opus-100-corpus/v1.0/supervised
cd $SP_PATH
./spm_encode --model=$DATA_PATH/opus.$sl.model \
                < $dir/$lp/opus.$lp-train.$sl \
                > $DATA_PATH/supervised/$lp/opus.$lp-train.$sl.sp
./spm_encode --model=$DATA_PATH/opus.$tl.model \
                < $dir/$lp/opus.$lp-train.$tl \
                > $DATA_PATH/supervised/$lp/opus.$lp-train.$tl.sp
cd $CUR_DIR

if [ -f $dir/$lp/opus.$lp-dev.$sl ]
then
    echo "$lp: parsing dev data"
    cd $SP_PATH
    ./spm_encode --model=$DATA_PATH/opus.$sl.model \
                < $dir/$lp/opus.$lp-dev.$sl \
                > $DATA_PATH/supervised/$lp/opus.$lp-dev.$sl.sp
    ./spm_encode --model=$DATA_PATH/opus.$tl.model \
                < $dir/$lp/opus.$lp-dev.$tl \
                > $DATA_PATH/supervised/$lp/opus.$lp-dev.$tl.sp
    cd $CUR_DIR
else
    echo "$lp: dev data not found"
fi

if [ -f $dir/$lp/opus.$lp-test.$sl ]
then
    echo "$lp: parsing test data"
    cd $SP_PATH
    ./spm_encode --model=$DATA_PATH/opus.$sl.model \
                < $dir/$lp/opus.$lp-test.$sl \
                > $DATA_PATH/supervised/$lp/opus.$lp-test.$sl.sp
    ./spm_encode --model=$DATA_PATH/opus.$tl.model \
                < $dir/$lp/opus.$lp-test.$tl \
                > $DATA_PATH/supervised/$lp/opus.$lp-test.$tl.sp
    cd $CUR_DIR
else
    echo "$lp: test data not found"
fi
done
```

#### Step 4: Parse the test sets for zero-shot translation directions
```
mkdir -p $DATA_PATH/zero-shot
for dir in $DATA_PATH/opus-100-corpus/v1.0/zero-shot/*
do
lp=$(basename $dir)  # get name of dir from full path
mkdir -p $DATA_PATH/zero-shot/$lp
echo "$lp: parsing zero-shot test data"
IFS=- read sl tl <<< $lp
cd $SP_PATH
./spm_encode --model=$DATA_PATH/opus.$sl.model \
            < $dir/opus.$lp-test.$sl \
            > $DATA_PATH/zero-shot/$lp/opus.$lp-test.$sl.sp
./spm_encode --model=$DATA_PATH/opus.$tl.model \
            < $dir/opus.$lp-test.$tl \
            > $DATA_PATH/zero-shot/$lp/opus.$lp-test.$tl.sp
cd $CUR_DIR
done
```
