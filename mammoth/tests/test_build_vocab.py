"""Tests for `mammoth_build_vocab`.

One tokenizer design serves both sentence-level and paragraph-level NMT:
- input is read like training data: plain text one example per line (line endings
  dropped), JSONL by its "text" value only;
- newlines inside an example are kept, isolated as their own tokens (never glued to
  words), so paragraphs round-trip exactly;
- all 256 byte tokens are regular vocabulary, so byte fallback replaces `<unk>`.
"""

import gzip
import json
import random
import sys

import pytest

pytest.importorskip('tokenizers')

from mammoth.bin.build_vocab import BYTE_TOKENS, main, train_tokenizer, train_tokenizer_from_texts  # noqa: E402


TEXTS = [
    'the quick brown fox jumps over the lazy dog',
    'a lazy dog sleeps in the warm sun',
    'quick thinking saves the day',
]
VOCAB_SIZE = 400

PARAGRAPHS = [
    'The lazy dog sleeps.\nThe warm sun shines.',
    'First paragraph here.\n\nSecond paragraph here.',
    'Results:\n- the quick fox\n- the lazy dog',
    'The dog sleeps. \n\n The sun shines.',
]
ROUND_TRIP_CASES = PARAGRAPHS + [
    'Emoji 🦣 and café!',
    'Кириллица и ещё «кавычки».\nВторая строка.',
    'Tabs\tand  double  spaces.',
    '\nLeading newline and trailing newline\n',
]


def _write_plain(path, texts, gz=False, newline='\n'):
    opener = gzip.open if gz else open
    with opener(path, 'wb') as f:
        f.write(''.join(t + newline for t in texts).encode('utf-8'))
    return str(path)


def _write_jsonl(path, texts, gz=False):
    opener = gzip.open if gz else open
    with opener(path, 'wt', encoding='utf-8') as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({'id': i, 'text': t, 'source': 'unittest'}) + '\n')
    return str(path)


def _vocab(tokenizer):
    return set(tokenizer.get_vocab())


def _paragraph_corpus(n=400, seed=0):
    rng = random.Random(seed)
    words = 'the quick brown fox jumps over lazy dog a warm sun sleeps in day results show model'.split()

    def sentence():
        return ' '.join(rng.choices(words, k=rng.randint(3, 8))).capitalize() + rng.choice('.!?:')

    return [
        rng.choice(['\n', '\n\n']).join(' '.join(sentence() for _ in range(rng.randint(1, 3)))
                                        for _ in range(rng.randint(1, 4)))
        for _ in range(n)
    ]


@pytest.fixture(scope='module')
def paragraph_tokenizer():
    return train_tokenizer_from_texts(_paragraph_corpus() + PARAGRAPHS, VOCAB_SIZE)


# --- reading input files ---------------------------------------------------------

def test_line_endings_are_not_learned_from_plain_text(tmp_path):
    vocab = _vocab(train_tokenizer([_write_plain(tmp_path / 'train.txt', TEXTS)], VOCAB_SIZE))
    assert not any('\n' in tok for tok in vocab if tok != '\n')


@pytest.mark.parametrize('name,writer,kwargs', [
    ('train.txt.gz', _write_plain, {'gz': True}),
    ('train.crlf.txt', _write_plain, {'newline': '\r\n'}),
    ('train.jsonl', _write_jsonl, {}),
    ('train.jsonl.gz', _write_jsonl, {'gz': True}),
], ids=['txt.gz', 'crlf', 'jsonl', 'jsonl.gz'])
def test_all_formats_give_the_same_vocab_as_plain_text(tmp_path, name, writer, kwargs):
    reference = _vocab(train_tokenizer([_write_plain(tmp_path / 'ref.txt', TEXTS)], VOCAB_SIZE))
    path = writer(tmp_path / name, TEXTS, **kwargs)
    assert _vocab(train_tokenizer([path], VOCAB_SIZE)) == reference


def test_jsonl_syntax_and_other_keys_are_not_learned(tmp_path):
    path = _write_jsonl(tmp_path / 'train.jsonl.gz', TEXTS, gz=True)
    vocab = _vocab(train_tokenizer([path], VOCAB_SIZE))
    for char in '{}":':
        assert char not in vocab
    assert not any('source' in tok or 'unittest' in tok for tok in vocab)
    assert '�' not in vocab  # gzip bytes were decompressed, not decoded as garbage


def test_multiple_mixed_format_files(tmp_path):
    files = [
        _write_jsonl(tmp_path / 'a.jsonl.gz', TEXTS[:1], gz=True),
        _write_plain(tmp_path / 'b.txt', TEXTS[1:2]),
        _write_plain(tmp_path / 'c.gz', TEXTS[2:], gz=True),
    ]
    assert _vocab(train_tokenizer(files, VOCAB_SIZE)) == _vocab(train_tokenizer_from_texts(TEXTS, VOCAB_SIZE))


def test_invalid_utf8_is_replaced_without_writing_a_cleaned_copy(tmp_path):
    path = tmp_path / 'bad.txt'
    path.write_bytes(b'good line\nbad \xff byte\n')
    vocab = _vocab(train_tokenizer([str(path)], VOCAB_SIZE))
    assert '�' in vocab
    assert sorted(p.name for p in tmp_path.iterdir()) == ['bad.txt']


# --- paragraph handling ----------------------------------------------------------

def test_newline_is_always_a_token(tmp_path):
    """Even a tokenizer trained only on sentences has a dedicated newline token."""
    tok = train_tokenizer_from_texts(TEXTS, VOCAB_SIZE)
    assert tok.token_to_id('\n') is not None


def test_newlines_are_never_glued_to_other_characters(paragraph_tokenizer):
    newline_tokens = [t for t in paragraph_tokenizer.get_vocab() if '\n' in t]
    assert '\n' in newline_tokens and '\n\n' in newline_tokens  # learned from paragraph data
    assert all(set(t) == {'\n'} for t in newline_tokens)


def test_word_tokens_do_not_depend_on_a_following_newline(paragraph_tokenizer):
    mid = paragraph_tokenizer.encode('The dog sleeps. The sun.', add_special_tokens=False).tokens
    end = paragraph_tokenizer.encode('The dog sleeps.\nThe sun.', add_special_tokens=False).tokens
    assert mid[:mid.index('▁The', 1)] == end[:end.index('\n')]


@pytest.mark.parametrize('text', ROUND_TRIP_CASES)
def test_exact_round_trip(paragraph_tokenizer, text):
    ids = paragraph_tokenizer.encode(text, add_special_tokens=False).ids
    assert paragraph_tokenizer.decode(ids, skip_special_tokens=True) == text


def test_windows_line_endings_are_normalized(paragraph_tokenizer):
    ids = paragraph_tokenizer.encode('First line.\r\nSecond line.', add_special_tokens=False).ids
    assert paragraph_tokenizer.decode(ids) == 'First line.\nSecond line.'


# --- byte fallback ---------------------------------------------------------------

def test_byte_tokens_are_regular_vocabulary(paragraph_tokenizer):
    vocab = paragraph_tokenizer.get_vocab()
    assert all(b in vocab for b in BYTE_TOKENS) and len(BYTE_TOKENS) == 256
    added = {t.content for t in paragraph_tokenizer.get_added_tokens_decoder().values()}
    assert not added & set(BYTE_TOKENS)  # special tokens would be dropped by decode(skip_special_tokens=True)


def test_rare_characters_fall_back_to_bytes_not_unk(paragraph_tokenizer):
    tokens = paragraph_tokenizer.encode('a 🦣 b', add_special_tokens=False).tokens
    assert '<unk>' not in tokens
    assert '<0xF0>' in tokens


def test_vocab_size_includes_byte_tokens():
    tok = train_tokenizer_from_texts(_paragraph_corpus(n=2000, seed=1), VOCAB_SIZE)
    assert tok.get_vocab_size() == VOCAB_SIZE


# --- integration with MAMMOTH ----------------------------------------------------

@pytest.fixture(scope='module')
def saved_tokenizer(tmp_path_factory):
    pytest.importorskip('transformers')
    out = tmp_path_factory.mktemp('tok')
    path = _write_jsonl(out / 'train.jsonl.gz', _paragraph_corpus() + PARAGRAPHS, gz=True)
    argv = sys.argv
    sys.argv = ['mammoth_build_vocab', '--input_file', path, '--output_dir', str(out), '--vocab_size', str(VOCAB_SIZE)]
    try:
        main()
    finally:
        sys.argv = argv
    return str(out / 'tokenizer.json')


def test_main_writes_tokenizer_with_mammoth_specials(saved_tokenizer):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(saved_tokenizer)
    for special in ['</s>', '<pad>', '<s>', '<unk>', '<mask>']:
        assert tok.token_to_id(special) is not None


def test_hf_tokenizer_vocab_round_trips_paragraphs(saved_tokenizer):
    """MAMMOTH's own wrapper (used for training and translation) must keep paragraphs intact."""
    from mammoth.inputters.vocab import HFTokenizerVocab
    vocab = HFTokenizerVocab(saved_tokenizer)
    assert vocab.subword_type == 'sentencepiece'  # ▁-based transforms keep working
    for text in ROUND_TRIP_CASES:
        ids = [vocab[t] for t in vocab.tokenize(text)]
        assert vocab.decode_tokens(ids, skip_special_tokens=True) == text


def test_translation_output_keeps_paragraphs(saved_tokenizer):
    """translate.py decodes predictions through TranslationBuilder; newlines must survive it."""
    import torch
    from mammoth.constants import DefaultTokens
    from mammoth.inputters.vocab import HFTokenizerVocab
    from mammoth.translate.translation import TranslationBuilder

    vocab = HFTokenizerVocab(saved_tokenizer)
    builder = TranslationBuilder(data=None, vocabs={'tgt': vocab})
    text = 'First paragraph here.\n\nSecond paragraph here.'
    ids = [vocab[t] for t in vocab.tokenize(text)] + [vocab[DefaultTokens.EOS]]
    assert builder._build_target_tokens(None, None, None, torch.tensor(ids)) == [text]


def test_missing_input_file_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['mammoth_build_vocab', '--input_file', str(tmp_path / 'nope.jsonl.gz')])
    with pytest.raises(FileNotFoundError):
        main()
