# JSONL input and paragraph-level NMT

Branch `jsonl-input`, compared to `pytorch_backend`.

## Summary

MAMMOTH can now read JSONL corpora, optionally gzipped, for tokenizer training, model training, validation and inference. Only the `"text"` key of each record is used. Because a JSONL `"text"` can contain line breaks, this branch also makes MAMMOTH work for **paragraph-level** translation. The same tokenizer and the same code serve both sentence-level and paragraph-level data; there is no mode switch.

```
{"text": "Erster Absatz.\n\nZweiter Absatz.", "id": 17, "source": "web"}
```

## What changed for users

| Area | Before | Now |
|---|---|---|
| Corpus files (`path_src`, `path_tgt`, `path_valid_*`, `translate.py -src/-tgt`) | Plain text, optionally `.gz` | Also `.jsonl` / `.jsonl.gz`; only `"text"` is used, other keys are ignored |
| Plain-text examples | Included the trailing `\n` of the line | The line ending is dropped; it separates examples and is not part of them |
| Indexed datasets (`data_type: indexed`) | Plain uncompressed text only | Any supported format, including `.gz` and JSONL; `train.jsonl.gz` → `train.bin` / `train.idx` |
| `mammoth_build_vocab` input | Raw files given to the tokenizer library: no `.gz`, no JSONL (JSON syntax was learned as text, and gzip bytes were "cleaned" into garbage) | Read like training data, so all formats work |
| `mammoth_build_vocab` invalid UTF-8 | First 10M characters checked, a `.utf8_cleaned` copy written; bad bytes later in the file crashed training | Bad bytes anywhere are replaced with `�` while reading; no extra file |
| Tokenizer: newlines | `\n` glued to words (`▁today.\n`, ~330 such tokens in a 32k vocab); in-text newlines became `<unk>` | Newline runs are their own tokens (`\n`, `\n\n`), never glued to words; `\r\n` is normalized to `\n` |
| Tokenizer: rare characters | `byte_fallback=True` but no byte tokens in the vocab, so emoji and rare characters became `<unk>` | 256 byte tokens (`<0x00>`…`<0xFF>`) are regular vocabulary; no `<unk>`; decoding reproduces the input exactly |
| `--vocab_size` | Learned tokens + specials | Total size, including the 256 byte tokens |
| Translations | All whitespace (including newlines) collapsed to single spaces | Newlines kept |
| `translate.py` output | One line per translation | Default unchanged: one line per translation, newlines turned into spaces (stays line-aligned with the input). New `-jsonl` flag → one `{"text": ...}` record per translation, newlines kept. `.gz` output paths are gzip-compressed |

## Design decisions

### One reader for every path
`mammoth/inputters/corpus_io.py` (new) has `open_corpus_lines(path)`. It yields one example string per line: plain text without its line ending, or the `"text"` value of a JSONL record. The format is chosen by file extension, and a `.gz` suffix means gzipped. Every place that reads corpora goes through it:

- `read_examples_from_files` (training, validation)
- `preprocess_text_to_indexed` (indexed datasets)
- `split_corpus` (inference sharding)
- `mammoth_build_vocab`

As a result, the tokenizer is trained on exactly the strings the model later sees.

Malformed records stop loading with the file name and line number: invalid JSON, a missing `"text"` key, or a non-string `"text"`. Blank lines count as malformed, because skipping them silently would misalign source and target.

### Inference shards are lists, not re-joined text
`split_corpus` used to join N lines into a `StringIO`, which was then split again on `\n`. A JSONL text containing newlines would be cut apart, and texts without a final newline would be glued together. Shards are now lists of example strings.

### Line endings are not content
Old `.txt`-trained tokenizers learned the line ending as part of the last word of every sentence: `▁today.` and `▁today.\n` were separate tokens. This wasted vocabulary and split training signal between the two forms. HF and Android inference never send that `\n`, so those paths already ran slightly mismatched. Dropping line endings in the reader makes `.txt`, JSONL, `translate.py`, HF and Android consistent.

### Tokenizer: newline split before Metaspace (option A), not byte-level BPE
DeepSeek's byte-level BPE (option B) was considered. Both were prototyped on paragraph data: both gave exact round trips and no `<unk>`, and B used about 4% fewer tokens on toy data. A was chosen for these reasons:

- It keeps the `▁` word-marker convention that `transforms/denoising.py`, `transforms/tokenize.py` and `utils/alignment.py` rely on. Mammoth still detects the subword type as `sentencepiece`.
- Its base units are whole characters, with bytes only as a fallback, so tokens stay readable for accented and Cyrillic text.

The tokenizer pipeline:

```
normalizer:    \r\n?  ->  \n
pre-tokenizer: Split(\n+, isolated)  ->  Metaspace(▁, prepend_scheme="first")
model:         BPE(byte_fallback=True) + 256 byte tokens + "\n" in the initial alphabet
decoder:       Replace(▁ -> " ")  ->  ByteFallback  ->  Fuse  ->  Strip(1 leading space)
```

`prepend_scheme="first"` adds `▁` only at the start of the text, not after each newline, so decoding is exact. As a consequence, a word at the start of a line is `The` rather than `▁The`, just like a word at the very start of a text.

### Byte tokens must not be special tokens
If the byte tokens are registered as trainer `special_tokens`, `decode(skip_special_tokens=True)`, which MAMMOTH uses, silently drops them: `'café 🦣'` decodes to `'a '`. They are therefore appended to the BPE vocabulary after training (`_add_byte_tokens`).

### gzip output is closed explicitly
`translate.py` never closed its output file. That was harmless for plain text, which is flushed after each write, but a `.gz` file only gets its end marker (CRC and length) on close. The output file is now closed after the last shard.

### Metrics unchanged
sacrebleu treats newlines as whitespace, so BLEU and chrF on paragraph output need no change. I checked this.

## Compatibility

- **Existing tokenizers and checkpoints** load and run unchanged. The only difference: `translate.py` on `.txt` input no longer passes the trailing `\n`, so these models see `▁today.` instead of `▁today.\n`. That is a small mismatch for models trained before this branch, and the same one the HF and Android paths already had.
- **New tokenizers** built with `mammoth_build_vocab` differ from old ones: they have newline tokens, 256 byte tokens and a new decoder. Do not mix an old tokenizer with a model trained on a new one.
- **`translate.py` output** stays plain text, one translation per line, unless `-jsonl` is given. The format is not inferred from the `-output` extension; a `.jsonl` output path without `-jsonl` gets plain text and a warning.
- **The `Translator` API** has a new `out_jsonl` argument (default `False`). `Translator.from_opts` sets it from the `-jsonl` flag.

## Files

| File | Change |
|---|---|
| `mammoth/inputters/corpus_io.py` | **New.** `is_jsonl_path`, `open_corpus_lines`, `open_corpus_output`, `format_output_line` |
| `mammoth/inputters/dataset.py` | `read_examples_from_files` uses the reader; accepts any iterable of strings; only closes files it opened. Fixed an existing bug where an open target handle was replaced by the source handle. `_get_indexed_path` strips `.jsonl(.gz)` |
| `mammoth/inputters/indexed_dataset.py` | Pretokenization reads through the reader |
| `mammoth/utils/misc.py` | `split_corpus` yields lists of examples |
| `mammoth/bin/build_vocab.py` | Reads through the reader; new tokenizer pipeline; byte tokens; `train_tokenizer` / `train_tokenizer_from_texts` split out for testing; `.utf8_cleaned` copies removed |
| `mammoth/translate/translation.py` | Keeps newlines in decoded translations |
| `mammoth/translate/translator.py` | Writes output through `format_output_line`; `out_jsonl` option set from `-jsonl`; output opened via `open_corpus_output` (gzip support) |
| `mammoth/opts.py` | New `-jsonl` translate flag (default off) |
| `mammoth/bin/translate.py` | Closes the output file |
| `docs/source/prepare_data.md` | JSONL input and paragraph-level sections |
| `mammoth/tests/test_jsonl_input.py` | **New.** 47 tests: reading, errors, stride/offset, inference shards, indexed preprocessing, output format |
| `mammoth/tests/test_build_vocab.py` | **New.** 27 tests: formats, round trips, newline and byte tokens, vocab size, `HFTokenizerVocab` integration, translation output |

The full suite gives 346 passed, 9 skipped on macOS (`pytorch_backend`: 272 passed, 9 skipped).

## Not covered yet

- A real paragraph-level training run and a `translate.py` run on LUMI.
- The HF `inference.py` and the Android `MammothTranslator.kt` may clean up whitespace in a way that removes newlines from the output. Not checked.
- Paragraphs are longer than sentences: check `max_length` / `filtertoolong`, and re-convert LiteRT models with larger `--enc-max-len` / `--dec-max-len`.
