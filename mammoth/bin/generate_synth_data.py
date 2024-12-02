import click
import numpy as np
import os
import yaml
from typing import Optional
from pathlib import Path

BRACKETS = [
    ('(', ')'),
    ('[', ']'),
    ('{', '}'),
    ('<', '>'),
    ('«', '»'),
    ('❲', '❳'),
]

TASKSEP_IDX = -1
PADDING_IDX = -2

TOKEN_MAP = {
    TASKSEP_IDX: '->',
    PADDING_IDX: '',
}


def multi_query_associative_recall(
    vocab_size: int,
    num_examples: int,
    seed: int,
    num_kv_pairs: int = 8,
    num_queries: int = 4,
):
    """
    Adapted from
    https://github.com/HazyResearch/zoology/blob/main/zoology/data/associative_recall.py
    """
    assert num_kv_pairs * 2 < vocab_size
    assert num_queries <= num_kv_pairs

    np.random.seed(seed)

    # two tokens for key and value
    context_size = num_kv_pairs * 2

    # create keys so that each key is present exactly once in each example
    key_vocab_size = vocab_size // 2
    key_choices = np.arange(1, key_vocab_size)
    value_choices = np.arange(key_vocab_size, vocab_size)

    keys_unshuffled = np.tile(key_choices, (num_examples, 1))
    keys = np.apply_along_axis(
        np.random.choice,
        1,
        keys_unshuffled,
        replace=False,
        size=num_kv_pairs
    )

    values_unshuffled = np.tile(value_choices, (num_examples, 1))
    values = np.apply_along_axis(
        np.random.choice,
        1,
        values_unshuffled,
        replace=False,
        size=num_kv_pairs
    )

    # create sequences
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # up to this point follows zoology

    # select a random shuffled index into keys, values
    potential_query_indices = np.tile(
        np.arange(0, num_kv_pairs),
        (num_examples, 1),
    )
    selected_query_indices = np.apply_along_axis(
        np.random.choice,
        1,
        potential_query_indices,
        replace=False,
        size=num_queries,
    )
    query_keys = np.take_along_axis(keys, selected_query_indices, axis=1)
    query_values = np.take_along_axis(values, selected_query_indices, axis=1)

    source = np.concatenate(
        [kvs, np.full((num_examples, 1), TASKSEP_IDX), query_keys],
        axis=1,
    )

    return source, query_values


def copy_source(
    vocab_size: int,
    num_examples: int,
    seq_len: int,
    seed: int,
    distractor_separator: Optional[int] = None,
):
    assert vocab_size > seq_len

    np.random.seed(seed)

    token_choices = np.arange(1, vocab_size)

    tokens_unshuffled = np.tile(token_choices, (num_examples, 1))
    tokens = np.apply_along_axis(
        np.random.choice,
        1,
        tokens_unshuffled,
        replace=False,
        size=seq_len
    )
    if distractor_separator is not None:
        source = np.concatenate(
            [
                tokens[:, :distractor_separator],
                np.full((num_examples, 1), TASKSEP_IDX),
                tokens[:, distractor_separator:],
            ],
            axis=1,
        )
    else:
        source = tokens

    return source, tokens


def reverse_source(*args, **kwargs):
    source, target = copy_source(*args, **kwargs)
    return source, target[:, ::-1]


def sort_source(*args, **kwargs):
    source, target = copy_source(*args, **kwargs)
    target = np.sort(target, axis=1)
    return source, target


def counting(
    vocab_size: int,
    num_examples: int,
    max_len: int,
    seed: int,
):
    np.random.seed(seed)

    token_choices = np.arange(1, vocab_size)
    count_choices = np.arange(1, max_len)
    selected_tokens = np.random.choice(token_choices, replace=True, size=num_examples)
    selected_counts = np.random.choice(count_choices, replace=True, size=num_examples)

    source = np.zeros((num_examples, max_len - 1))
    mask_indices = np.tile(count_choices, (num_examples, 1))
    mask = mask_indices <= selected_counts[:, np.newaxis]
    source += selected_tokens[:, np.newaxis] * mask
    source += np.full((num_examples, max_len - 1), PADDING_IDX) * ~mask

    target = np.concatenate([
        selected_tokens[:, np.newaxis],
        selected_counts[:, np.newaxis],
    ], axis=1)

    return source, target


def reverse_counting(*args, **kwargs):
    target, source = counting(*args, **kwargs)
    return source, target


def denumericalize(array):
    for i in range(array.shape[0]):
        tokens = [TOKEN_MAP.get(tok_i, str(int(tok_i))) for tok_i in array[i]]
        line = ' '.join(tokens).strip()
        yield line


def make_vocab(
    vocab_path,
    vocab_size,
    specials=None
):
    """
    vocab_path: Path to write out the vocabulary
    vocab_size: number of normal tokens
    specials:
        default None means "use all specials".
        To use no specials, pass an empty list.
    """
    if specials is None:
        vocab = ['<unk>', '<s>', '</s>']
        for token in TOKEN_MAP.values():
            if len(token) > 0:
                vocab.append(token)
        vocab.extend(range(vocab_size))
    with vocab_path.open('w') as fout:
        for item in vocab:
            print(f'{item}\t0', file=fout)


TASK_SPECS = {
    'multi_query_associative_recall_kv6_q2': {
        'func': multi_query_associative_recall,
        'func_args': {
            'num_kv_pairs': 6,
            'num_queries': 2,
        },
    },
    'multi_query_associative_recall_kv20_q4': {
        'func': multi_query_associative_recall,
        'func_args': {
            'num_kv_pairs': 20,
            'num_queries': 4,
        },
    },
    'multi_query_associative_recall_kv12_q8': {
        'func': multi_query_associative_recall,
        'func_args': {
            'num_kv_pairs': 12,
            'num_queries': 8,
        },
    },
    'copy_source': {
        'func': copy_source,
        'func_args': {
            'seq_len': 100,
        },
    },
    'distractor_separator_kv20_q4': {
        # The source of this task resembles MQAR.
        # Using this task prevents the separator symbol from being associated as a marker of MQAR.
        'func': copy_source,
        'func_args': {
            'seq_len': (2 * 20) + 4,
            'distractor_separator': -4,
        },
    },
    'distractor_separator_kv12_q8': {
        'func': copy_source,
        'func_args': {
            'seq_len': (2 * 12) + 8,
            'distractor_separator': -8,
        },
    },
    'reverse_source': {
        'func': reverse_source,
        'func_args': {
            'seq_len': 100,
        },
    },
    'sort_source': {
        'func': sort_source,
        'func_args': {
            'seq_len': 100,
        },
    },
    'counting': {
        'func': counting,
        'func_args': {
            'max_len': 100,
        },
    },
    'reverse_counting': {
        'func': reverse_counting,
        'func_args': {
            'max_len': 100,
        },
    },
}


def parse_config(config_path: Path):
    with config_path.open('r') as fin:
        config = yaml.safe_load(fin)
    src_path_template = config['config_config']['src_path']
    tgt_path_template = config['config_config']['tgt_path']
    valid_src_path_template = config['config_config']['valid_src_path']
    valid_tgt_path_template = config['config_config']['valid_tgt_path']
    vocabs = config['src_vocab']

    # vocab keys indicate which task param combinations you want to generate
    for key, vocab_path in vocabs.items():
        if key not in TASK_SPECS:
            raise Exception(f'Requested task {key} not in available tasks {TASK_SPECS.keys()}')
        task_specs = TASK_SPECS[key]
        paths = {
            'src_path': Path(src_path_template.format(src_lang=key, tgt_lang=key)),
            'tgt_path': Path(tgt_path_template.format(src_lang=key, tgt_lang=key)),
            'valid_src_path': Path(valid_src_path_template.format(src_lang=key, tgt_lang=key)),
            'valid_tgt_path': Path(valid_tgt_path_template.format(src_lang=key, tgt_lang=key)),
            'vocab_path': Path(vocab_path),
        }
        yield key, task_specs, paths


def ensure_parents_exist(paths):
    for path in paths:
        if path.parent.exists():
            if not path.parent.is_dir():
                raise Exception(f'The parent {path.parent} of path {path} exists, but is not a directory')
        else:
            os.makedirs(path.parent)


def generate_from_config(
    config_path: Path,
    vocab_size: int,
    num_examples_train: int,
    num_examples_test: int,
    start_seed: int,
    shared_vocab: Optional[Path] = None,
):
    if shared_vocab:
        ensure_parents_exist([shared_vocab])
        make_vocab(shared_vocab, vocab_size)
    for i, (key, task_specs, paths) in enumerate(parse_config(config_path)):
        print(f'Generating {key}...')
        ensure_parents_exist(paths.values())
        args = {
            'vocab_size': vocab_size,
            'num_examples': num_examples_train + num_examples_test,
            'seed': start_seed + i,
        }
        args.update(task_specs['func_args'])
        source, target = task_specs['func'](**args)
        if task_specs.get('format', 'numpy') == 'numpy':
            source_strs = denumericalize(source)
            target_strs = denumericalize(target)
        else:
            source_strs = source
            target_strs = target
        with open(paths['src_path'], 'w') as fout_train, \
             open(paths['valid_src_path'], 'w') as fout_test:
            for i, line in enumerate(source_strs):
                if i < num_examples_train:
                    print(line, file=fout_train)
                else:
                    print(line, file=fout_test)
        with open(paths['tgt_path'], 'w') as fout_train, \
             open(paths['valid_tgt_path'], 'w') as fout_test:
            for i, line in enumerate(target_strs):
                if i < num_examples_train:
                    print(line, file=fout_train)
                else:
                    print(line, file=fout_test)
        # TODO: make a vocab with custom specials, if not shared_vocab


@click.command(context_settings={'show_default': True})
@click.option('--config_path', type=Path, required=True)
@click.option('--vocab_size', type=int, default=300)
@click.option('--num_examples_train', type=int, default=1000000)
@click.option('--num_examples_test', type=int, default=100)
@click.option('--start_seed', type=int, default=1)
@click.option(
    '--shared_vocab',
    type=Path,
    default=None,
    help='if specified, outputs a shared vocab to this path. '
    'if not specified, task specific vocabs are created (TODO).'
)
def main(
    config_path: Path,
    vocab_size: int,
    num_examples_train: int,
    num_examples_test: int,
    start_seed: int,
    shared_vocab: Optional[Path] = None,
):
    generate_from_config(
        config_path=config_path,
        vocab_size=vocab_size,
        num_examples_train=num_examples_train,
        num_examples_test=num_examples_test,
        start_seed=start_seed,
        shared_vocab=shared_vocab,
    )


if __name__ == '__main__':
    main()


# other tasks: close brackets (with and without invalid inputs to detect), something else using separator?
# verification (from string?)
