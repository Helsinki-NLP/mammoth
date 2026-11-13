"""Reading corpus files: plain text (one example per line) or JSONL (one JSON object per line).

The format is chosen by file extension:
    *.jsonl, *.jsonl.gz      -> each line is a JSON object; the example is its "text" value
    anything else            -> each line is an example
A trailing `.gz` means gzip-compressed in both cases.

In both formats the line ending only separates examples and is not part of them,
so "text" values may contain newlines (paragraph-level data) while a plain-text
example never does (sentence-level data).
"""

import gzip
import json

JSONL_TEXT_KEY = 'text'
JSONL_EXTENSIONS = ('.jsonl', '.jsonl.gz')


def is_jsonl_path(path):
    return str(path).lower().endswith(JSONL_EXTENSIONS)


def _open_text(path, errors):
    path = str(path)
    if path.endswith('.gz'):
        return gzip.open(path, 'rt', encoding='utf-8', errors=errors)
    return open(path, 'rt', encoding='utf-8', errors=errors)


def _parse_jsonl_line(line, path, line_no):
    try:
        record = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f'{path}: line {line_no} is not valid JSON ({e})') from e
    if not isinstance(record, dict) or JSONL_TEXT_KEY not in record:
        raise ValueError(f'{path}: line {line_no} has no "{JSONL_TEXT_KEY}" key')
    text = record[JSONL_TEXT_KEY]
    if not isinstance(text, str):
        raise ValueError(
            f'{path}: line {line_no} "{JSONL_TEXT_KEY}" must be a string, got {type(text).__name__}'
        )
    return text


def open_corpus_lines(path, errors='strict'):
    """Yield one example string per line of `path`.

    Plain-text lines are yielded without their line ending (`\n` or `\r\n`).
    JSONL records yield their "text" value verbatim; other keys are ignored.
    `errors` is the UTF-8 decoding error policy (see `open()`), e.g. 'replace'.
    Calling `.close()` on the returned generator closes the underlying file.
    """
    with _open_text(path, errors) as f:
        if not is_jsonl_path(path):
            # text mode already turned \r\n into \n
            for line in f:
                yield line[:-1] if line.endswith('\n') else line
            return
        for line_no, line in enumerate(f, start=1):
            yield _parse_jsonl_line(line, path, line_no)


def open_corpus_output(path):
    """Open `path` for writing UTF-8 text, gzip-compressed if it ends in `.gz`."""
    path = str(path)
    if path.endswith('.gz'):
        return gzip.open(path, 'wt', encoding='utf-8')
    return open(path, 'w', encoding='utf-8')


def format_output_line(text, jsonl):
    """Format one translation as a line of an output corpus.

    JSONL keeps the text intact, newlines included. Plain text must stay one
    example per line to remain aligned with the input, so whitespace runs
    (including newlines) are collapsed to single spaces.
    """
    if jsonl:
        return json.dumps({JSONL_TEXT_KEY: text}, ensure_ascii=False) + '\n'
    return ' '.join(text.split()) + '\n'
