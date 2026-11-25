# Prepare Data

MAMMOTH trains on plain parallel text: one sentence per line, with the source file and the target file line-aligned. Before following this page, make sure that you have [installed](install) MAMMOTH, which includes the dependencies required below.

MAMMOTH tokenizes with HuggingFace tokenizers, on the fly as it loads the data, so there is nothing to pre-tokenize. You only need to check that your parallel files are aligned and split off a validation set. (SentencePiece is deprecated and will be removed.)


## Example: Europarl

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
# the full archive has an extra top-level europarl/ folder; --strip-components=1 drops it
tar -xvzf europarl.tar.gz -C europarl_data --strip-components=1
```
Note that the extracted dataset will require around 30GB of disk space. Alternatively, you can only download the data for the three example languages (666M).
```bash
wget https://mammoth101.a3s.fi/europarl-3langs.tar.gz
mkdir europarl_data
tar -xvzf europarl-3langs.tar.gz -C europarl_data
```

Either way, each language pair ends up in its own folder, e.g. `europarl_data/bg-en/europarl-v7.bg-en.bg` and `europarl_data/bg-en/europarl-v7.bg-en.en`.

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
        --src europarl_data/${lang}-en/europarl-v7.${lang}-en.${lang} \
        --tgt europarl_data/${lang}-en/europarl-v7.${lang}-en.en \
        --src_lang ${lang} --tgt_lang en \
        --out_dir europarl_data/split/${lang}-en \
        --valid_size 1000
done
```

This produces `europarl_data/split/bg-en/{train,val}.{bg,en}` and the same for `cs-en`. The script prints how many pairs each cleaning step dropped, so check that the numbers look reasonable for your corpus. It holds both files in memory, which is fine for a corpus like Europarl (a few million lines per pair). For much larger corpora, use a streaming tool instead.

The script only does basic cleaning. For further filtering, for example by length or length ratio, MAMMOTH has on-the-fly filter transforms (`filtertoolong`, `filterwordratio`, `filterrepetitions`, `filterterminalpunct` and `filternonzeronumerals`) that you add to a task's `transforms` list.

### Step 3: Train a tokenizer

Train a BPE tokenizer on the **training** files with the `mammoth_build_vocab` command, which is installed with the package. It needs the `tokenizers` and `transformers` packages, both listed in `requirements.txt`.

```bash
mammoth_build_vocab \
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

Transforms such as `filtertoolong` still work with this setup, since they run on the fly too. If you would rather tokenize once up front, see the [indexed dataset guide](indexed_datasets.md).


## Other corpora

The same steps work for any parallel corpus. Two multilingual corpora that are easy to start with, both from [OPUS](https://opus.nlpl.eu/):

- [UNPC](https://opus.nlpl.eu/UNPC/corpus/version/UNPC): manually translated UN documents (1990 to 2014) in the six official UN languages, Arabic, Chinese, English, French, Russian and Spanish. Please cite Ziemski, M., Junczys-Dowmunt, M., and Pouliquen, B. (2016), *The United Nations Parallel Corpus*, LREC 2016.
- [OPUS-100](https://opus.nlpl.eu/legacy/opus-100.php): English paired with 99 languages, with train/dev/test splits and zero-shot test sets between non-English languages:

  ```bash
  wget https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-v1.0.tar.gz
  tar -xzf opus-100-corpus-v1.0.tar.gz
  ```
