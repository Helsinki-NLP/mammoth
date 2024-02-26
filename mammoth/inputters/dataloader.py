import collections
import itertools
import math
import random

import torch

from mammoth.inputters.dataset import get_corpus
from mammoth.utils.logging import logger


def build_dataloader(dataset, batch_size, batch_type, pool_size=None, n_buckets=None, cycle=True, as_iter=True):
    """Convert an mammoth.inputters.ParallelCorpus into an infinite iterator of batches"""
    if not cycle:
        loader = InferenceBatcher(dataset, batch_size)
    else:
        if batch_type == 'sents':
            n_buckets = 1

            def bucket_fn(_):
                return 0

            def numel_fn(_):
                return 1

        elif batch_type == 'tokens':

            def bucket_fn(example_dict):
                """map example dict to bucket index"""
                # subtract two for bos/eos
                src_len = min(len(example_dict['src']), n_buckets) - 2
                if 'tgt' in example_dict:
                    tgt_len = min(len(example_dict['tgt']), n_buckets) - 2
                else:
                    tgt_len = src_len
                # maybe dump it in the last bucket if it's just too long
                return src_len, tgt_len

            def numel_fn(example_dict):
                """count tokens in example"""
                if 'tgt' in example_dict:
                    true_size = len(example_dict['src']) + len(example_dict['tgt'])
                else:
                    true_size = len(example_dict['src'])
                return true_size

        loader = LookAheadBucketing(dataset, pool_size, n_buckets, batch_size, bucket_fn, numel_fn)
    return iter(loader) if as_iter else loader


DatasetMetadata = collections.namedtuple('DatasetMetadata', 'src_lang tgt_lang encoder_id decoder_id corpus_id')


class InferenceBatcher():
    """Iterator for inference"""

    def __init__(self, dataset, batch_size):
        self.examples_stream = dataset
        self.collate_fn = dataset.collate_fn
        self.batch_size = batch_size

    def __iter__(self):
        accum = []
        for example in iter(self.examples_stream):
            accum.append(example)
            if len(accum) >= self.batch_size:
                yield self.collate_fn(accum)
                accum = []
        if accum:
            yield self.collate_fn(accum)


class LookAheadBucketing():
    def __init__(self, dataset, look_ahead_size, n_buckets, batch_size, bucket_fn, numel_fn):
        self.dataset = dataset
        # actual generator of examples
        self.examples_stream = iter([])
        # tracks whether the stream needs to be restarted
        self._is_exhausted = True
        self.n_buckets = n_buckets
        self._buckets = [
            [
                []
                for _ in range(n_buckets)
            ]
            for _ in range(n_buckets)
        ]
        self.look_ahead_size = look_ahead_size
        self.batch_size = batch_size
        self.bucket_fn = bucket_fn
        self.numel_fn = numel_fn
        self.collate_fn = dataset.collate_fn
        self._init()

    def _init(self):
        logger.info('LookAheadBucketing: initialization start')
        self.examples_stream = iter(self.dataset)
        for example in range(self.look_ahead_size):
            self.maybe_replenish()
            if self._is_exhausted:
                break
        assert not self.is_empty(), 'Dataset contains no usable example!'
        logger.info('LookAheadBucketing: initialization done')

    def maybe_replenish(self):
        """try to look up one more example to add to this reservoir."""
        try:
            example = next(self.examples_stream)
            s_idx, t_idx = self.bucket_fn(example)
            self._buckets[s_idx][t_idx].append(example)
            self._is_exhausted = False
        except StopIteration:
            self._is_exhausted = True

    def bucket_is_empty(self, s_idx: int, t_idx: int) -> bool:
        """check if this bucket is empty"""
        return len(self._buckets[s_idx][t_idx]) == 0

    def _choose_bucket(self):
        """pick a bucket at random"""
        buckets = [(s, t) for s in range(self.n_buckets) for t in range(self.n_buckets)]
        weights = [len(self._buckets[s][t]) for s in range(self.n_buckets) for t in range(self.n_buckets)]
        bucket_idx = random.choices(buckets, weights=weights, k=1)[0]
        return bucket_idx

    def _select_from_bucket(self, s_idx: int, t_idx: int) -> object:
        """randomly select an item from a bucket"""
        bucket = self._buckets[s_idx][t_idx]
        obj_idx = random.randrange(len(bucket))
        # swap to last to get O(1) deletion
        bucket[obj_idx], bucket[-1] = bucket[-1], bucket[obj_idx]
        return bucket.pop()

    def is_empty(self) -> bool:
        """check if all buckets are empty"""
        return all(len(bucket) == 0 for bucket in itertools.chain.from_iterable(self._buckets))

    def _spiralling(self, s_idx: int, t_idx: int):
        def _seq():
            # from https://math.stackexchange.com/questions/163080/on-a-two-dimensional-grid-is-there-a-formula-i-can-use-to-spiral-coordinates-in#answer-3448361  # noqa: E501
            for n in itertools.count(1):
                k = math.ceil((math.sqrt(n) - 1) / 2.0)
                t = 2 * k + 1
                m = t ** 2
                t = t - 1
                if n >= m - t:
                    yield k - (m - n), k
                else:
                    m = m - t
                    if n >= m - t:
                        yield -k, k - (m - n)
                    else:
                        m = m - t
                        if n >= m - t:
                            yield -k + (m - n), -k
                        else:
                            yield k, -k + (m - n - t)

        offsets = ((s_idx + x, t_idx + y) for x, y in _seq())
        # offsets = itertools.takewhile(
        #     # this far out is obviously too far out
        #     lambda tup: (tup[0] < self.n_buckets * 2 + 1) and (tup[1] < self.n_buckets * 2 + 1),
        #     offsets,
        # )
        offsets = filter(
            lambda tup: (0 <= tup[0] < self.n_buckets) and (0 <= tup[1] < self.n_buckets),
            offsets,
        )
        # maybe more brittle than the takewhile a few lines above
        offsets = itertools.islice(offsets, self.n_buckets ** 2)
        yield from offsets

    def __iter__(self):
        while True:
            # 1. maybe we've exhausted both the stream and the buckets:
            # if so, we restart the example stream
            if self.is_empty() and self._is_exhausted:
                self._init()
            accum, cur_batch_size = [], 0
            # 2. pick a length at random
            smallest_bucket_idx = self._choose_bucket()
            current_bucket_idx = smallest_bucket_idx
            # 3. build batch
            batch_is_complete = False
            # stop either when batch is built or when it can't be built
            while not (batch_is_complete or self.is_empty()):
                # maybe switch buckets
                current_bucket_idx = smallest_bucket_idx
                next_indices = self._spiralling(*current_bucket_idx)
                while self.bucket_is_empty(*current_bucket_idx):
                    current_bucket_idx = next(next_indices)
                # retrieve and process the example
                example = self._select_from_bucket(*current_bucket_idx)
                accum.append(example)
                numel = self.numel_fn(example)
                cur_batch_size += numel
                batch_is_complete = cur_batch_size >= self.batch_size

                # 4. try to replenish reservoir if possible
                # if not, this will also update self._is_exhausted
                self.maybe_replenish()

            yield self.collate_fn(accum)


class DynamicDatasetIter(object):
    """Yield batch from (multiple) plain text corpus.

    Args:
        corpora (dict[str, ParallelCorpus]): collections of corpora to iterate;
        corpora_info (dict[str, dict]): corpora infos correspond to corpora;
        transforms (dict[str, Transform]): transforms may be used by corpora;
        fields (dict[str, Field]): fields dict for convert corpora into Tensor;
        is_train (bool): True when generate data for training;
        batch_type (str): batching type to count on, choices=[tokens, sents];
        batch_size (int): numbers of examples in a batch;
        batch_size_multiple (int): make batch size multiply of this;
        data_type (str): input data type, currently only text;
        pool_size (int): accum this number of examples in a dynamic dataset;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate data files with this stride;
        offset (int): iterate data files with this offset.

    Attributes:
        dataset_adapter (DatasetAdapter): organize raw corpus to tensor adapt;
        mixer (MixingStrategy): the strategy to iterate corpora.
    """

    def __init__(
        self,
        task_queue_manager,
        opts,
        corpora_info,
        transforms_cls,
        vocabs_dict,
        is_train,
        batch_type,
        batch_size,
        batch_size_multiple,
        data_type="text",
        pool_size=2048,
        n_buckets=1024,
        skip_empty_level='warning',
    ):
        self.task_queue_manager = task_queue_manager
        self.opts = opts
        self.transforms_cls = transforms_cls
        self.vocabs_dict = vocabs_dict
        self.corpora_info = corpora_info
        self.is_train = is_train
        self.init_iterators = False

        self.batch_type = batch_type
        self.batch_size = batch_size
        self.batch_size_multiple = batch_size_multiple
        self.device = 'cpu'
        self.pool_size = pool_size
        self.n_buckets = n_buckets
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level

    @classmethod
    def from_opts(cls, task_queue_manager, transforms_cls, vocabs_dict, opts, is_train):
        """Initilize `DynamicDatasetIter` with options parsed from `opts`."""
        batch_size = opts.batch_size if is_train else opts.valid_batch_size
        if opts.batch_size_multiple is not None:
            batch_size_multiple = opts.batch_size_multiple
        else:
            batch_size_multiple = 8 if opts.model_dtype == "fp16" else 1
        return cls(
            task_queue_manager,
            opts,
            opts.tasks,
            transforms_cls,
            vocabs_dict,
            is_train,
            opts.batch_type,
            batch_size,
            batch_size_multiple,
            data_type=opts.data_type,
            pool_size=opts.pool_size,
            n_buckets=opts.n_buckets,
            skip_empty_level=opts.skip_empty_level,
        )

    def _init_datasets(self):
        self.dataset_iterators = dict()
        for task in self.task_queue_manager.get_my_tasks():
            src_vocab = self.vocabs_dict[('src', task.src_lang)]
            tgt_vocab = self.vocabs_dict[('tgt', task.tgt_lang)]
            # merged_fields = {'src': src_fields['src'], 'tgt': tgt_fields['tgt']}
            # logger.debug(f'merged_fields {merged_fields}')

            metadata = task.get_serializable_metadata()

            device = torch.device(
                self.task_queue_manager.device_context.local_rank
                if self.task_queue_manager.device_context.is_gpu()
                else 'cpu'
            )

            # Case 1: we are training, and the task must contain some path to training data
            # Case 2: we are validation (hence self.is_train := False), we need an iterator
            # if and only the task defines validation data, i.e. if the key `path_valid_src`
            # is defined
            if self.is_train or self.opts.tasks[task.corpus_id].get('path_valid_src', None) is not None:
                corpus = get_corpus(
                    self.opts, task, src_vocab, tgt_vocab, is_train=self.is_train
                ).to(device)

                # iterator over minibatches
                ordered_iter = build_dataloader(
                    corpus,
                    self.batch_size,
                    self.batch_type,
                    self.pool_size,
                    n_buckets=self.n_buckets,
                    cycle=self.is_train,
                    as_iter=self.is_train,
                )

                self.dataset_iterators[task.corpus_id] = (ordered_iter, metadata)

        self.init_iterators = True

    def __iter__(self):
        if self.init_iterators is False:
            self._init_datasets()

        if not self.is_train:
            # to be absolutely clear: all the validation data is read per validation loop
            all_val_data = [
                zip(ordered_iter, itertools.repeat(metadata), itertools.repeat(0))
                for ordered_iter, metadata in self.dataset_iterators.values()
            ]
            yield from itertools.chain.from_iterable(all_val_data)

        else:
            while True:
                batch_task_sample = self.task_queue_manager.sample_corpus_ids()
                my_task = batch_task_sample.tasks[self.task_queue_manager.global_rank]
                ordered_iter, metadata = self.dataset_iterators[my_task.corpus_id]
                for _ in self.task_queue_manager.accum_count:
                    batch = next(ordered_iter)
                    if batch_task_sample.training_step == 0:
                        # De-numericalize a few sentences for debugging
                        logger.warning(
                            f'src shape: {batch.src[0].shape} tgt shape: {batch.tgt.shape} '
                            f'batch size: {batch.batch_size}'
                        )
                        src_vocab = self.vocabs_dict[('src', metadata.src_lang)]
                        tgt_vocab = self.vocabs_dict[('tgt', metadata.tgt_lang)]
                        for sent_idx in range(3):
                            toks = [src_vocab.itos[tok_id.item()] for tok_id in batch.src[0][:, sent_idx, 0]]
                            logger.warning(f'{sent_idx} {metadata.src_lang} src: {" ".join(toks)}')
                            toks = [tgt_vocab.itos[tok_id.item()] for tok_id in batch.tgt[:, sent_idx, 0]]
                            logger.warning(f'{sent_idx} {metadata.tgt_lang} tgt: {" ".join(toks)}')
                    yield batch, metadata, batch_task_sample.training_step
