import pytest
from itertools import product, count

from mammoth.inputters.dataloader import build_dataloader


class hashabledict(dict):
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


class MockStream():
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def __iter__(self):
        return iter(self.items)

    def collate_fn(self, items, line_idx):
        return items


@pytest.mark.parametrize(
    ('max_batch_size', 'lookahead_minibatches'),
    [
        (12, 4),
        (13, 4),
        (14, 4),
        (15, 4),
        (12, 5),
        (13, 5),
        (14, 5),
        (15, 5),
    ],
)
def test_simple_lookeahead_bucketing(max_batch_size, lookahead_minibatches):
    index_gen = count()
    stream = MockStream([
        hashabledict({
            'src': tuple([letter for _ in range(i)]),
            'tgt': tuple([letter for _ in range(j)]),
            'line_idx': next(index_gen)
        })
        for letter in 'xyz'
        for i, j in product(range(1, 11), range(1, 11))
    ])
    lab = build_dataloader(
        stream,
        batch_size=max_batch_size,
        batch_type='tokens',
        max_look_ahead_sentences=512,
        lookahead_minibatches=lookahead_minibatches,
        cycle=True,
        as_iter=False
    )
    examples_read = []
    batches = iter(lab)
    for _ in range(1000):
        batch = next(batches)
        print(batch)
        assert len(batch) > 0
        src_toks = sum(len(ex['src']) for ex in batch)
        tgt_toks = sum(len(ex['tgt']) for ex in batch)
        # check that the batch size is respected
        assert src_toks <= max_batch_size
        assert tgt_toks <= max_batch_size, str(batch)
        examples_read.extend(batch)
    # Check that the stream was cycled
    assert len(examples_read) > len(stream)


@pytest.mark.parametrize(
    'batch_size',
    [1, 5, 12, 2048],
)
def test_sentence_minibatcher(batch_size):
    index_gen = count()
    stream = MockStream([
        hashabledict({
            'src': tuple([letter for _ in range(i)]),
            'tgt': tuple([letter for _ in range(j)]),
            'line_idx': next(index_gen)
        })
        for letter in 'xyz'
        for i, j in product(range(1, 11), range(1, 11))
    ])
    lab = build_dataloader(
        stream,
        batch_size=batch_size,
        batch_type='sents',
        cycle=True,
        as_iter=False
    )
    examples_read = []
    batches = iter(lab)
    for _ in range(1000):
        batch = next(batches)
        print(batch)
        assert len(batch) == batch_size
        examples_read.extend(batch)
    # Check that the stream was cycled
    assert len(examples_read) > len(stream)
