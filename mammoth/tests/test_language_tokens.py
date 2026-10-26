import json
import os
from types import SimpleNamespace

import pytest

from mammoth.inputters.language_tokens import (
    add_language_tokens_to_tokenizer,
    save_tokenizers_with_language_tokens,
)


def _write_tokenizer(path, vocab):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {'model': {'type': 'WordLevel', 'vocab': vocab, 'unk_token': '<unk>'}, 'added_tokens': []}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    with open(path, 'rb') as f:
        return f.read()


def _read(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def test_add_language_tokens_writes_copy_and_keeps_original(tmp_path):
    original = str(tmp_path / 'orig' / 'tokenizer.json')
    original_bytes = _write_tokenizer(original, {'<unk>': 0, 'a': 1, '<en>': 2})
    output = str(tmp_path / 'out' / 'tokenizer.json')

    n_added = add_language_tokens_to_tokenizer(original, {'<en>', '<fi>', '<de>'}, output)

    assert n_added == 2
    with open(original, 'rb') as f:
        assert f.read() == original_bytes
    assert not os.path.exists(original.replace('.json', '.json.backup'))
    data = _read(output)
    assert data['model']['vocab'] == {'<unk>': 0, 'a': 1, '<en>': 2, '<de>': 3, '<fi>': 4}
    assert [(t['content'], t['id']) for t in data['added_tokens']] == [('<de>', 3), ('<fi>', 4)]


def test_add_language_tokens_nothing_to_add_writes_nothing(tmp_path):
    original = str(tmp_path / 'orig' / 'tokenizer.json')
    _write_tokenizer(original, {'<unk>': 0, '<en>': 1})
    output = str(tmp_path / 'out' / 'tokenizer.json')

    assert add_language_tokens_to_tokenizer(original, {'<en>'}, output) == 0
    assert not os.path.exists(output)


def test_add_language_tokens_refuses_to_overwrite_original(tmp_path):
    original = str(tmp_path / 'orig' / 'tokenizer.json')
    _write_tokenizer(original, {'<unk>': 0})
    with pytest.raises(ValueError):
        add_language_tokens_to_tokenizer(original, {'<en>'}, original)


def test_save_tokenizers_writes_one_copy_per_side_and_lang(tmp_path):
    # deu and eng tokenizers live in directories with the same name (".../32000/")
    deu = str(tmp_path / 'deu' / '32000' / 'tokenizer.json')
    _write_tokenizer(deu, {'<unk>': 0, 'a': 1})
    eng = str(tmp_path / 'eng' / '32000' / 'tokenizer.json')
    _write_tokenizer(eng, {'<unk>': 0, 'b': 1})
    # Already contains the tokens -> keeps original path
    complete = str(tmp_path / 'complete' / 'tokenizer.json')
    _write_tokenizer(complete, {'<unk>': 0, '<en>': 1, '<fi>': 2})

    opts = SimpleNamespace(
        src_vocab={'deu': deu, 'eng': eng},
        tgt_vocab={'eng': eng, 'fin': deu, 'dan': complete},
    )
    out_dir = str(tmp_path / 'model' / 'tokenizers')

    new_paths = save_tokenizers_with_language_tokens(
        opts, {'src': ['deu', 'eng'], 'tgt': ['eng', 'fin', 'dan']}, {'<en>', '<fi>'}, out_dir
    )

    def copy(side, lang):
        return os.path.join(out_dir, side, f'{lang}_tokenizer.json')

    assert new_paths == {
        ('src', 'deu'): (deu, copy('src', 'deu')),
        ('src', 'eng'): (eng, copy('src', 'eng')),
        ('tgt', 'eng'): (eng, copy('tgt', 'eng')),
        ('tgt', 'fin'): (deu, copy('tgt', 'fin')),
        ('tgt', 'dan'): (complete, complete),
    }
    assert opts.src_vocab == {'deu': copy('src', 'deu'), 'eng': copy('src', 'eng')}
    assert opts.tgt_vocab == {'eng': copy('tgt', 'eng'), 'fin': copy('tgt', 'fin'), 'dan': complete}
    assert '<en>' not in _read(deu)['model']['vocab']
    assert _read(copy('src', 'deu'))['model']['vocab']['<fi>'] == 3
    assert _read(copy('tgt', 'eng'))['model']['vocab']['b'] == 1
    # Languages sharing a source tokenizer get identical copies
    assert _read(copy('tgt', 'fin')) == _read(copy('src', 'deu'))
    assert sorted(os.listdir(out_dir)) == ['src', 'tgt']


def test_config_pointing_at_saved_copy_is_extended_in_place(tmp_path):
    original = str(tmp_path / 'shared' / 'tokenizer.json')
    original_bytes = _write_tokenizer(original, {'<unk>': 0, 'a': 1})
    out_dir = str(tmp_path / 'model' / 'tokenizers')

    # First training: copy written to out_dir/shared/tokenizer.json
    opts = SimpleNamespace(src_vocab={'en': original}, tgt_vocab={'fi': original})
    langs = {'src': ['en'], 'tgt': ['fi']}
    save_tokenizers_with_language_tokens(opts, langs, {'<en>', '<fi>'}, out_dir)
    copy = opts.src_vocab['en']
    assert copy == os.path.join(out_dir, 'src', 'en_tokenizer.json')

    # Resume with the config pointing at the copy, same languages: no error, nothing changes
    opts = SimpleNamespace(src_vocab={'en': copy}, tgt_vocab={'fi': copy})
    new_paths = save_tokenizers_with_language_tokens(
        opts, langs, {'<en>', '<fi>'}, out_dir, extend_existing=True
    )
    assert new_paths == {('src', 'en'): (copy, copy), ('tgt', 'fi'): (copy, copy)}
    assert opts.src_vocab == {'en': copy}

    # Resume with the copy and a new language '<da>' that sorts before the others:
    # existing ids are kept, the new token is appended
    opts = SimpleNamespace(src_vocab={'en': copy, 'da': copy}, tgt_vocab={'fi': copy})
    save_tokenizers_with_language_tokens(
        opts, {'src': ['en', 'da'], 'tgt': ['fi']}, {'<en>', '<fi>', '<da>'}, out_dir, extend_existing=True
    )
    vocab = _read(copy)['model']['vocab']
    assert vocab == {'<unk>': 0, 'a': 1, '<en>': 2, '<fi>': 3, '<da>': 4}
    with open(original, 'rb') as f:
        assert f.read() == original_bytes


def test_resume_with_original_path_extends_existing_copy(tmp_path):
    original = str(tmp_path / 'shared' / 'tokenizer.json')
    original_bytes = _write_tokenizer(original, {'<unk>': 0, 'a': 1})
    out_dir = str(tmp_path / 'model' / 'tokenizers')
    copy = os.path.join(out_dir, 'src', 'en_tokenizer.json')
    da_copy = os.path.join(out_dir, 'src', 'da_tokenizer.json')

    opts = SimpleNamespace(src_vocab={'en': original}, tgt_vocab={'fi': original})
    save_tokenizers_with_language_tokens(opts, {'src': ['en'], 'tgt': ['fi']}, {'<en>', '<fi>'}, out_dir)

    # Resume with the config still pointing at the original, adding '<da>'
    opts = SimpleNamespace(src_vocab={'en': original, 'da': original}, tgt_vocab={'fi': original})
    save_tokenizers_with_language_tokens(
        opts, {'src': ['en', 'da'], 'tgt': ['fi']}, {'<en>', '<fi>', '<da>'}, out_dir, extend_existing=True
    )
    assert opts.src_vocab == {'en': copy, 'da': da_copy}
    # Existing copy keeps its ids; the new language's copy is built fresh from the original
    assert _read(copy)['model']['vocab'] == {'<unk>': 0, 'a': 1, '<en>': 2, '<fi>': 3, '<da>': 4}
    assert _read(da_copy)['model']['vocab'] == {'<unk>': 0, 'a': 1, '<da>': 2, '<en>': 3, '<fi>': 4}
    with open(original, 'rb') as f:
        assert f.read() == original_bytes


def test_fresh_training_rebuilds_copy_from_original(tmp_path):
    original = str(tmp_path / 'shared' / 'tokenizer.json')
    _write_tokenizer(original, {'<unk>': 0, 'a': 1})
    out_dir = str(tmp_path / 'model' / 'tokenizers')
    copy = os.path.join(out_dir, 'src', 'en_tokenizer.json')
    # Stale copy left over from an earlier run in the same directory
    _write_tokenizer(copy, {'<unk>': 0, 'stale': 1, '<xx>': 2})

    opts = SimpleNamespace(src_vocab={'en': original}, tgt_vocab={'fi': original})
    save_tokenizers_with_language_tokens(opts, {'src': ['en'], 'tgt': ['fi']}, {'<en>', '<fi>'}, out_dir)
    assert _read(copy)['model']['vocab'] == {'<unk>': 0, 'a': 1, '<en>': 2, '<fi>': 3}
