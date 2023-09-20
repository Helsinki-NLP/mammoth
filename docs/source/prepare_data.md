
# Prepare Data

## Tatoeba Challenge

TODO

## OPUS 100 
To get started, download the opus 100 dataset from [OPUS 100](https://opus.nlpl.eu/opus-100.php)

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