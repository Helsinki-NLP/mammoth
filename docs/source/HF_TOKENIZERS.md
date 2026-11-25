# HuggingFace Tokenizers

MAMMOTH tokenizes text with HuggingFace [`tokenizers`](https://huggingface.co/docs/tokenizers/index) (installed with MAMMOTH). Each language has its own `tokenizer.json`, and tokenization happens on the fly while the data is loaded, so the training files stay plain text. (SentencePiece is deprecated and will be removed.)

## 1. Train a tokenizer

```bash
mammoth_build_vocab --input_file corpus.txt --output_dir tokenizers/en --vocab_size 32000
```

| Option | Meaning |
|---|---|
| `--input_file` | One or more training text files, one sentence per line. Files with invalid UTF-8 are cleaned first (invalid bytes become `�`) into `<file>.utf8_cleaned`. |
| `--output_dir` | Where to write `tokenizer.json` (plus `tokenizer_config.json`). Default: the current directory. |
| `--vocab_size` | Vocabulary size, including the special tokens. Default: 32000. |

The tokenizer it trains:

- **Model**: BPE with byte fallback, so any character can be encoded even if it never appeared in training.
- **Word boundaries**: Metaspace pre-tokenizer: a word-initial piece starts with `▁` (U+2581), e.g. `"Hello, world!"` → `['▁Hello', ',', '▁world', '!']`. Punctuation is not split off before BPE.
- **Special tokens**: `</s>`, `<pad>`, `<s>`, `<unk>`, `<mask>`. Do not change them: MAMMOTH looks these tokens up by name.
- **No BOS/EOS template**: the tokenizer only splits text into subwords. MAMMOTH adds `<s>` and `</s>` itself when it loads the data.

You can train one tokenizer per language, or one shared tokenizer on all languages and point every language at it.

## 2. Use it in the config

```yaml
use_hf_tokenizer: true

src_vocab:
  en: tokenizers/en/tokenizer.json
tgt_vocab:
  fi: tokenizers/fi/tokenizer.json
```

MAMMOTH uses **separate source and target vocabularies**: `src_vocab` tokenizes the input of a task, `tgt_vocab` its output. The keys are the language codes used in `tasks.<id>.src_tgt`. Paths ending in `.json` are loaded as HuggingFace tokenizers.

A copy of each tokenizer is saved in the checkpoint, and translation uses that copy, so a model keeps working if the original `tokenizer.json` files are moved or changed.

## Using other tokenizers

Any `tokenizer.json` works if it contains the tokens `<s>`, `</s>`, `<pad>` and `<unk>`. Tokenizers that use other names for them (e.g. BERT's `[CLS]`, `[SEP]`, `[PAD]`) do not work as they are. To use a different algorithm (WordPiece, Unigram, …), write your own training script with the [`tokenizers` library](https://huggingface.co/docs/tokenizers/index), keeping the special tokens above and no post-processor template.
