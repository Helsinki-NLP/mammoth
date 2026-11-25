# Indexed Datasets (Pretokenization)

By default MAMMOTH tokenizes the text while it trains. For large corpora you can tokenize once up front instead: each text file is turned into an indexed dataset, a `.bin` file with the token ids and an `.idx` file with the sentence offsets. Training then reads token ids directly.

- Enable it per task with `data_type: indexed`.
- Transforms (`filtertoolong`, `prefix`, `denoising`, …) do **not** run on indexed data; they are ignored with a warning. Filter the data before you index it (see [Length filtering](#length-filtering) below).
- Length filtering is not part of indexing. Keep every sentence within `max_length` tokens.

## Automatic indexing

Point `path_src` / `path_tgt` (and the validation paths) at the text files as usual and add `data_type: indexed`. Before training, MAMMOTH looks for `.bin` / `.idx` files next to each text file. If they are missing, it creates them there; later runs reuse them. The indexed files are named after the text file with a `.txt`, `.gz` or `.txt.gz` extension removed, e.g. `train.ar.txt` → `train.ar.bin` + `train.ar.idx`.

## Manual indexing

To index a file yourself, e.g. in a separate CPU job:

```bash
python -m mammoth.scripts.preprocess_indexed \
  --input data/ar-en/train.ar.txt \
  --output_prefix data/ar-en/train.ar \
  --vocab_path tokenizers/ar/tokenizer.json \
  --workers 8
```

This writes `data/ar-en/train.ar.bin` and `data/ar-en/train.ar.idx`. In the config, give the prefix (or the original text path, which maps to the same prefix) as the data path.

## Configuration example

```yaml
max_length: 512

tasks:
  ar-en:
    src_tgt: ara-eng
    data_type: indexed            # no transforms for this task
    node_gpu: "0:0"
    enc_sharing_group: [ara, all]
    dec_sharing_group: [all, eng]
    path_src: data/ar-en/train.ar.txt
    path_tgt: data/ar-en/train.en.txt
    path_valid_src: data/ar-en/val.ar.txt
    path_valid_tgt: data/ar-en/val.en.txt
  fi-fi:
    src_tgt: fin-fin
    node_gpu: "0:1"
    enc_sharing_group: [fin, all]
    dec_sharing_group: [all, fin]
    transforms: [denoising, filtertoolong]   # plain text task: transforms work
    path_src: data/fi/train.fi.txt
    path_tgt: data/fi/train.fi.txt
    path_valid_src: data/fi/val.fi.txt
    path_valid_tgt: data/fi/val.fi.txt
```

Indexed and plain-text tasks can be mixed in one config.

## Length filtering

`filter_by_length` drops sentence pairs whose tokenized length is outside a range, using the same HuggingFace tokenizers as training:

```bash
python -m mammoth.scripts.filter_by_length \
  --src_input data/ar-en/train.ar.raw --tgt_input data/ar-en/train.en.raw \
  --src_output data/ar-en/train.ar.txt --tgt_output data/ar-en/train.en.txt \
  --src_tokenizer tokenizers/ar/tokenizer.json \
  --tgt_tokenizer tokenizers/en/tokenizer.json \
  --min_length 10 \
  --max_length 512
```

If both sides use the same tokenizer, pass `--tokenizer` once instead of `--src_tokenizer` / `--tgt_tokenizer`. Leave a little room below `max_length` for the `<s>` / `</s>` tokens MAMMOTH adds.
