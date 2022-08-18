"""Module that contain iterator used for dynamic data."""
# from collections import namedtuple
from itertools import cycle

# from torchtext.legacy.data import batch as torchtext_batch
# from onmt.inputters import str2sortkey, max_tok_len, OrderedIterator
# from onmt.inputters.corpus import build_corpora_iter, DatasetAdapter
# from onmt.inputters_mvp import build_dataloader
# from onmt.transforms import make_transforms
from onmt.utils.logging import logger


class MixingStrategy(object):
    """Mixing strategy that should be used in Data Iterator."""

    def __init__(self, iterables, weights):
        """Initilize neccessary attr."""
        self._valid_iterable(iterables, weights)
        self.iterables = iterables
        self.weights = weights

    def _valid_iterable(self, iterables, weights):
        iter_keys = iterables.keys()
        weight_keys = weights.keys()
        if iter_keys != weight_keys:
            raise ValueError(f"keys in {iterables} & {iterables} should be equal.")

    def __iter__(self):
        raise NotImplementedError


class SequentialMixer(MixingStrategy):
    """Generate data sequentially from `iterables` which is exhaustible."""

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in self._iter_datasets():
            iterable = self.iterables[ds_name]
            yield from iterable


class WeightedMixer(MixingStrategy):
    """A mixing strategy that mix data weightedly and iterate infinitely."""

    def __init__(self, iterables, weights):
        super().__init__(iterables, weights)
        self._iterators = {}
        self._counts = {}
        for ds_name in self.iterables.keys():
            self._reset_iter(ds_name)

    def _logging(self):
        """Report corpora loading statistics."""
        msgs = []
        for ds_name, ds_count in self._counts.items():
            msgs.append(f"\t\t\t* {ds_name}: {ds_count}")
        logger.info("Weighted corpora loaded so far:\n" + "\n".join(msgs))

    def _reset_iter(self, ds_name):
        self._iterators[ds_name] = iter(self.iterables[ds_name])
        self._counts[ds_name] = self._counts.get(ds_name, 0) + 1

    def _iter_datasets(self):
        for ds_name, ds_weight in self.weights.items():
            for _ in range(ds_weight):
                yield ds_name

    def __iter__(self):
        for ds_name in cycle(self._iter_datasets()):
            iterator = self._iterators[ds_name]
            try:
                item = next(iterator)
            except StopIteration:
                self._reset_iter(ds_name)
                iterator = self._iterators[ds_name]
                item = next(iterator)
            finally:
                yield item
