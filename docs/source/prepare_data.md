# Prepare Data

Before running these scripts, make sure that you have [installed](quickstart) Mamooth, which includes the dependencies required below.


## Quickstart: Europarl

In the [Quickstart tutorial](quickstart), we assume that you will download and preprocess the Europarl data by following the steps below.

### Step 1: Download the data
[Europarl parallel corpus](https://www.statmt.org/europarl/) is a multilingual resource extracted from European Parliament proceedings and contains texts in 21 European languages. Download the Release v7 - a further expanded and improved version of the Europarl corpus on 15 May 2012 - from the original website or 
download the processed data by us:
```bash
wget https://mammoth101.a3s.fi/europarl.tar.gz
mkdir europarl_data
tar –xvzf europarl.tar.gz -C europarl_data
```
Note that the extracted dataset will require around 30GB of memory. Alternatively, you can only download the data for the three example languages (666M).
```bash
wget https://mammoth101.a3s.fi/europarl-3langs.tar.gz
mkdir europarl_data
tar –xvzf europarl-3langs.tar.gz -C europarl_data
```

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

### Step 2: Tokenization
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

## UNPC
[UNPC](https://opus.nlpl.eu/UNPC/corpus/version/UNPC) consists of manually translated UN documents from the last 25 years (1990 to 2014) for the six official UN languages, Arabic, Chinese, English, French, Russian, and Spanish. 
We preprocess the data. You can download the processed data by:
```bash
wget https://mammoth-share.a3s.fi/unpc.tar
```
Or you can use the scripts provided by the tarball to process the data yourself. 

For references, please cite this reference: Ziemski, M., Junczys-Dowmunt, M., and Pouliquen, B., (2016), The United Nations Parallel Corpus, Language Resources and Evaluation (LREC’16), Portorož, Slovenia, May 2016.


## OPUS 100

In this guideline, we will also create our custom sentencepiece tokenizer.

To do that, you will also need to compile a sentencepiece installation in your environment (not just pip install). 
Follow the instructions on [sentencepiece github](https://github.com/google/sentencepiece?tab=readme-ov-file#build-and-install-sentencepiece-command-line-tools-from-c-source).

After that, download the opus 100 dataset from [OPUS 100](https://opus.nlpl.eu/legacy/opus-100.php)

### Step 1: Set relevant paths, variables and download

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

### Step 2: Train SentencePiece models and get vocabs

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

### Step 3: Parse train, valid and test sets for supervised translation directions
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

### Step 4: Parse the test sets for zero-shot translation directions
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
