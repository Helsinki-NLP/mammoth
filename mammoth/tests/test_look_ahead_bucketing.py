from itertools import product

import unittest
from mammoth.inputters.dataloader import (
    build_dataloader,
    LookAheadBucketing,
    InferenceBatcher,
)


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
        stream = MockStream([
            hashabledict({
                'src': tuple([letter for _ in range(i)]),
                'tgt': tuple([letter for _ in range(j)]),
            })
            for letter in 'xyz'
            for i, j in product(range(1, 11), range(1, 11))
        ])
        lab = build_dataloader(stream, 2, 'tokens', pool_size=4, n_buckets=4, cycle=True, as_iter=False)
        examples_read = []
        batches = iter(lab)
        while not (lab._is_exhausted and lab.is_empty()):
            examples_read.extend(next(batches))
        sorted_src_ref = sorted([ex['src'] for ex in stream.items])
        sorted_src_obs = sorted([ex['src'] for ex in examples_read])
        self.assertTrue(sorted_src_ref == sorted_src_obs)
        sorted_tgt_ref = sorted([ex['tgt'] for ex in stream.items])
        sorted_tgt_obs = sorted([ex['tgt'] for ex in examples_read])
        self.assertTrue(sorted_tgt_ref == sorted_tgt_obs)

    def test_reroutes(self):
        stream = MockStream([hashabledict({'src': '_', 'tgt': '_'})] * 10)
        lab = build_dataloader(stream, 2, 'tokens', 4, 2, cycle=True, as_iter=False)
        self.assertTrue(isinstance(lab, LookAheadBucketing))
        not_lab = build_dataloader(stream, 2, 'tokens', 4, 2, cycle=False, as_iter=False)
        self.assertTrue(isinstance(not_lab, InferenceBatcher))

    def test_always_continues(self):
        stream = MockStream([hashabledict({'src': '_', 'tgt': '_'})] * 10)
        was_exhausted = False
        stopped_exhaustion = False
        lab = build_dataloader(stream, 2, 'tokens', pool_size=4, n_buckets=4, cycle=True, as_iter=False)
        batches = iter(lab)
        all_items = []
        for _ in range(len(stream) * 3 // 2):
            all_items.extend(next(batches))
            was_exhausted = was_exhausted or lab._is_exhausted
            if was_exhausted:
                stopped_exhaustion = stopped_exhaustion or not lab._is_exhausted

        self.assertTrue(was_exhausted)
        self.assertTrue(stopped_exhaustion)
        self.assertTrue(len(all_items) > len(stream))
