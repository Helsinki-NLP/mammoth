from itertools import product

import unittest
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

    def collate_fn(self, items):
        return items


class TestLookAheadBucketing(unittest.TestCase):

    def test_all_read(self):
        max_batch_size = 12
        stream = MockStream([
            hashabledict({
                'src': tuple([letter for _ in range(i)]),
                'tgt': tuple([letter for _ in range(j)]),
            })
            for letter in 'xyz'
            for i, j in product(range(1, 11), range(1, 11))
        ])
        lab = build_dataloader(
            stream,
            batch_size=max_batch_size,
            batch_type='tokens',
            pool_size=4,
            n_buckets=4,
            cycle=True,
            as_iter=False
        )
        examples_read = []
        batches = iter(lab)
        for _ in range(1000):
            batch = next(batches)
            assert len(batch) > 0
            src_toks = sum(len(ex['src']) for ex in batch)
            tgt_toks = sum(len(ex['tgt']) for ex in batch)
            # check that the batch size is respected
            assert src_toks <= max_batch_size
            assert tgt_toks <= max_batch_size, str(batch)
            examples_read.extend(batch)
        # Check that the stream was cycled
        self.assertTrue(len(examples_read) > len(stream))
