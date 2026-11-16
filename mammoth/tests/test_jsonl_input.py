"""Tests for JSONL corpus input (training, validation and inference).

A corpus path ending in `.jsonl` or `.jsonl.gz` is read as one JSON object per
line; only the "text" key is used as the example, every other key is ignored.
Plain-text paths (`.txt`, `.gz`, anything else) are read one example per line.
In both formats the line ending is a separator, not part of the example, so a
plain-text sentence and the same sentence as a JSONL "text" are identical.
"""

import gzip
import json
from io import StringIO

import pytest

from mammoth.inputters.corpus_io import format_output_line, is_jsonl_path, open_corpus_lines, open_corpus_output
from mammoth.inputters.dataset import _get_indexed_path, read_examples_from_files
from mammoth.utils.misc import split_corpus


SRC_RECORDS = [
    {"text": "hello world", "id": 1, "lang": "eng"},
    {"id": 2, "text": "überall grüße"},
    {"text": "the third line", "meta": {"source": "web"}},
]
TGT_RECORDS = [
    {"text": "hola mundo", "id": 1},
    {"text": "saludos por todas partes", "id": 2},
    {"text": "la tercera línea", "id": 3},
]


def _write_jsonl(path, records, gz):
    opener = gzip.open if gz else open
    with opener(path, 'wt', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return str(path)


def _texts(records):
    return [rec["text"] for rec in records]


@pytest.fixture(params=[True, False], ids=['jsonl.gz', 'jsonl'])
def jsonl_pair(tmp_path, request):
    gz = request.param
    ext = '.jsonl.gz' if gz else '.jsonl'
    src = _write_jsonl(tmp_path / f'train.src{ext}', SRC_RECORDS, gz)
    tgt = _write_jsonl(tmp_path / f'train.tgt{ext}', TGT_RECORDS, gz)
    return src, tgt


# --- path detection -------------------------------------------------------------

@pytest.mark.parametrize('path,expected', [
    ('data/train.jsonl', True),
    ('data/train.jsonl.gz', True),
    ('data/train.JSONL.GZ', True),
    ('data/train.txt', False),
    ('data/train.txt.gz', False),
    ('data/train.gz', False),
    ('data/train.json', False),
    ('data/jsonl/train.txt', False),
])
def test_is_jsonl_path(path, expected):
    assert is_jsonl_path(path) is expected


# --- open_corpus_lines ----------------------------------------------------------

def test_open_corpus_lines_extracts_text_key_in_order(jsonl_pair):
    src, tgt = jsonl_pair
    assert list(open_corpus_lines(src)) == _texts(SRC_RECORDS)
    assert list(open_corpus_lines(tgt)) == _texts(TGT_RECORDS)


@pytest.mark.parametrize('content', [
    'first line\nsecond line\n',
    'first line\nsecond line',          # no newline at end of file
    'first line\r\nsecond line\r\n',  # Windows line endings
], ids=['lf', 'no-final-newline', 'crlf'])
def test_open_corpus_lines_plain_text_drops_line_endings(tmp_path, content):
    plain = tmp_path / 'a.txt'
    plain.write_bytes(content.encode('utf-8'))
    gzipped = tmp_path / 'a.txt.gz'
    with gzip.open(gzipped, 'wb') as f:
        f.write(content.encode('utf-8'))

    expected = ['first line', 'second line']
    assert list(open_corpus_lines(str(plain))) == expected
    assert list(open_corpus_lines(str(gzipped))) == expected


def test_open_corpus_lines_plain_text_keeps_other_whitespace(tmp_path):
    plain = tmp_path / 'a.txt'
    plain.write_text('  indented\ttab  \n\n', encoding='utf-8')
    assert list(open_corpus_lines(str(plain))) == ['  indented\ttab  ', '']


def test_plain_text_and_jsonl_give_identical_examples(tmp_path):
    texts = ['hello world', 'überall grüße']
    plain = tmp_path / 'a.txt'
    plain.write_text(''.join(t + '\n' for t in texts), encoding='utf-8')
    jsonl = _write_jsonl(tmp_path / 'a.jsonl.gz', [{"text": t} for t in texts], gz=True)
    assert list(open_corpus_lines(str(plain))) == list(open_corpus_lines(jsonl)) == texts


def test_open_corpus_lines_keeps_internal_newlines_verbatim(tmp_path):
    path = _write_jsonl(tmp_path / 'doc.jsonl.gz', [{"text": "line one\nline two"}], gz=True)
    assert list(open_corpus_lines(path)) == ["line one\nline two"]


def test_open_corpus_lines_missing_text_key_raises_with_location(tmp_path):
    path = _write_jsonl(tmp_path / 'bad.jsonl.gz', [{"text": "ok"}, {"content": "no text key"}], gz=True)
    with pytest.raises(ValueError, match=r'bad\.jsonl\.gz.*line 2.*"text"'):
        list(open_corpus_lines(path))


def test_open_corpus_lines_invalid_json_raises_with_location(tmp_path):
    path = tmp_path / 'bad.jsonl'
    path.write_text('{"text": "ok"}\n{not json}\n', encoding='utf-8')
    with pytest.raises(ValueError, match=r'bad\.jsonl.*line 2'):
        list(open_corpus_lines(str(path)))


def test_open_corpus_lines_non_string_text_raises(tmp_path):
    path = _write_jsonl(tmp_path / 'bad.jsonl', [{"text": ["a", "list"]}], gz=False)
    with pytest.raises(ValueError, match=r'line 1.*string'):
        list(open_corpus_lines(str(path)))


def test_open_corpus_lines_closes_file_when_closed_early(jsonl_pair):
    src, _ = jsonl_pair
    lines = open_corpus_lines(src)
    assert next(lines) == SRC_RECORDS[0]["text"]
    lines.close()  # must not raise; releases the underlying file handle
    with pytest.raises(StopIteration):
        next(lines)


# --- training / validation: read_examples_from_files ----------------------------

def _tokenize(string, side):
    return string.split()


def test_read_examples_from_jsonl_pair(jsonl_pair):
    src, tgt = jsonl_pair
    examples = list(read_examples_from_files(src, tgt, tokenize_fn=_tokenize))
    assert [ex['src'] for ex in examples] == [t.split() for t in _texts(SRC_RECORDS)]
    assert [ex['tgt'] for ex in examples] == [t.split() for t in _texts(TGT_RECORDS)]
    assert [ex['line_idx'] for ex in examples] == [0, 1, 2]


def test_read_examples_from_jsonl_respects_stride_and_offset(jsonl_pair):
    """Multi-GPU data sharding (stride/offset) must index JSONL records like text lines."""
    src, tgt = jsonl_pair
    examples = list(read_examples_from_files(src, tgt, tokenize_fn=_tokenize, stride=2, offset=1))
    assert [ex['src'] for ex in examples] == [SRC_RECORDS[1]["text"].split()]
    assert [ex['line_idx'] for ex in examples] == [1]


def test_read_examples_from_jsonl_src_only(jsonl_pair):
    src, _ = jsonl_pair
    examples = list(read_examples_from_files(src, None, tokenize_fn=_tokenize))
    assert [ex['src'] for ex in examples] == [t.split() for t in _texts(SRC_RECORDS)]
    assert all(ex['tgt'] is None for ex in examples)


def test_read_examples_mixed_jsonl_src_and_text_tgt(tmp_path):
    src = _write_jsonl(tmp_path / 'src.jsonl.gz', SRC_RECORDS, gz=True)
    tgt = tmp_path / 'tgt.txt'
    tgt.write_text(''.join(t + '\n' for t in _texts(TGT_RECORDS)), encoding='utf-8')
    examples = list(read_examples_from_files(src, str(tgt), tokenize_fn=_tokenize))
    assert [ex['src'] for ex in examples] == [t.split() for t in _texts(SRC_RECORDS)]
    assert [ex['tgt'] for ex in examples] == [t.split() for t in _texts(TGT_RECORDS)]


def test_read_examples_from_in_memory_src_and_tgt_handles():
    """Already-open handles (as used by inference shards) are read as-is, src and tgt separately."""
    src = StringIO('a b\nc d\n')
    tgt = StringIO('x y\nz w\n')
    examples = list(read_examples_from_files(src, tgt, tokenize_fn=_tokenize))
    assert [ex['src'] for ex in examples] == [['a', 'b'], ['c', 'd']]
    assert [ex['tgt'] for ex in examples] == [['x', 'y'], ['z', 'w']]


# --- inference: split_corpus ----------------------------------------------------

def _read_shards_lazily(path, shard_size):
    """Consume each shard before requesting the next, as bin/translate.py does."""
    n_shards, examples = 0, []
    for shard in split_corpus(path, shard_size):
        n_shards += 1
        examples += [ex['src'] for ex in read_examples_from_files(shard, None, tokenize_fn=_tokenize)]
    return n_shards, examples


@pytest.mark.parametrize('shard_size', [0, 2])
def test_split_corpus_jsonl_yields_text_only(jsonl_pair, shard_size):
    src, _ = jsonl_pair
    n_shards, examples = _read_shards_lazily(src, shard_size)
    assert n_shards == (2 if shard_size > 0 else 1)  # 3 records, 2 per shard
    assert examples == [t.split() for t in _texts(SRC_RECORDS)]


@pytest.mark.parametrize('shard_size', [0, 2])
def test_split_corpus_jsonl_shards_keep_multiline_records_whole(tmp_path, shard_size):
    records = [{"text": "one\ntwo"}, {"text": "three"}, {"text": "four"}]
    path = _write_jsonl(tmp_path / 'multi.jsonl.gz', records, gz=True)
    _, examples = _read_shards_lazily(path, shard_size)
    assert examples == [['one', 'two'], ['three'], ['four']]


# --- indexed (pre-tokenized) datasets -------------------------------------------

@pytest.mark.parametrize('text_path,expected', [
    ('/data/train.jsonl.gz', '/data/train'),
    ('/data/train.jsonl', '/data/train'),
    ('/data/train.txt.gz', '/data/train'),
    ('/data/train.txt', '/data/train'),
])
def test_get_indexed_path_strips_jsonl_extensions(text_path, expected):
    assert _get_indexed_path(text_path) == expected


def _write_word_level_tokenizer(path, words):
    tokenizers = pytest.importorskip('tokenizers')
    specials = ['<pad>', '<unk>', '<s>', '</s>', '<mask>']
    vocab = {tok: i for i, tok in enumerate(specials + words)}
    tok = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='<unk>'))
    tok.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tok.save(str(path))
    return str(path), vocab


def test_preprocess_jsonl_gz_to_indexed(tmp_path):
    from mammoth.inputters.indexed_dataset import IndexedDataset, preprocess_text_to_indexed

    words = sorted({w for t in _texts(SRC_RECORDS) for w in t.split()})
    vocab_path, vocab = _write_word_level_tokenizer(tmp_path / 'tokenizer.json', words)
    src = _write_jsonl(tmp_path / 'train.src.jsonl.gz', SRC_RECORDS, gz=True)
    prefix = str(tmp_path / 'train.src')

    assert preprocess_text_to_indexed(src, prefix, vocab_path, workers=1)

    dataset = IndexedDataset(prefix)
    assert len(dataset) == len(SRC_RECORDS)
    for i, text in enumerate(_texts(SRC_RECORDS)):
        assert list(dataset[i]) == [vocab[w] for w in text.split()]


# --- inference output -----------------------------------------------------------

def test_format_output_line_jsonl_keeps_paragraph_intact():
    text = 'Erster Absatz.\n\nZweiter Absatz: "zitiert" 🦣'
    line = format_output_line(text, jsonl=True)
    assert line.endswith('\n') and line.count('\n') == 1  # exactly one record per line
    assert json.loads(line) == {"text": text}
    assert '🦣' in line  # written as UTF-8, not \\u escapes


def test_format_output_line_plain_text_stays_one_line():
    """Plain-text output must stay line-aligned with the input, so newlines become spaces."""
    assert format_output_line('first para.\n\nsecond  para.', jsonl=False) == 'first para. second para.\n'
    assert format_output_line('a sentence.', jsonl=False) == 'a sentence.\n'


@pytest.mark.parametrize('name', ['out.jsonl', 'out.jsonl.gz', 'out.txt', 'out.txt.gz'])
def test_output_round_trips_through_reader(tmp_path, name):
    """What translate.py writes can be read back as a corpus (e.g. for back-translation)."""
    path = str(tmp_path / name)
    texts = ['one paragraph.\nsecond line.', 'überall grüße']
    jsonl = is_jsonl_path(path)  # i.e. translate.py run with -jsonl and a matching output name
    with open_corpus_output(path) as f:
        for t in texts:
            f.write(format_output_line(t, jsonl=jsonl))
    expected = texts if jsonl else ['one paragraph. second line.', 'überall grüße']
    assert list(open_corpus_lines(path)) == expected


def test_translate_output_format_defaults_to_plain_text():
    """Backward compatible: JSONL output only with -jsonl, whatever the -output extension."""
    from mammoth.bin.translate import _get_parser

    required = ['-tasks', 't.yaml', '-src_vocab', 'v', '-task_id', 'eng-spa', '-seed', '1', '-max_length', '10']
    base = required + ['-model', 'm', '-src', 's.jsonl', '-output', 'pred.jsonl']
    assert _get_parser().parse_args(base).jsonl is False
    assert _get_parser().parse_args(base + ['-jsonl']).jsonl is True
