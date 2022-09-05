import bisect
import collections
import itertools
import random

import torch

from onmt.inputters_mvp.dataset import get_corpus
from onmt.utils.logging import logger


def infinite_iterator(iterable):
    return itertools.chain.from_iterable(itertools.repeat(iterable))


def build_dataloader(dataset, batch_size, batch_type, pool_size):
    """Convert an onmt.inputters_mvp.ParallelCorpus into an infinite iterator of batches"""
    examples_stream = infinite_iterator(dataset)
    if batch_type == 'sents':
        def counting_fn(_): return 1
    elif batch_type == 'tokens':
        def counting_fn(example_dict): return len(example_dict['src']) + len(example_dict['tgt'])
    collate_fn = dataset.collate_fn
    return iter(LookAheadBucketing(examples_stream, pool_size, batch_size, counting_fn, counting_fn, collate_fn))


DatasetMetadata = collections.namedtuple('DatasetMetadata', 'src_lang tgt_lang encoder_id decoder_id corpus_id')


class LookAheadBucketing():
    def __init__(self, examples_stream, look_ahead_size, batch_size, bucket_fn, numel_fn, collate_fn):
        self.examples_stream = examples_stream
        self._prefetched = collections.defaultdict(list)
        self._buckets = []
        self.look_ahead_size = look_ahead_size
        self.batch_size = batch_size
        self.bucket_fn = bucket_fn
        self.numel_fn = numel_fn
        self.collate_fn = collate_fn
        self._init()

    def _init(self):
        logger.info('LookAheadBucketing: initialization start')
        for example in itertools.islice(self.examples_stream, self.look_ahead_size):
            bucket = self.bucket_fn(example)
            self._prefetched[bucket].append(example)
        self._buckets = sorted(self._prefetched.keys())
        logger.info('LookAheadBucketing: initialization done')

    def maybe_replenish(self) -> bool:
        try:
            example = next(self.examples_stream)
            bucket = self.bucket_fn(example)
            creates_new_bucket = bucket not in self._prefetched
            self._prefetched[bucket].append(example)
            if creates_new_bucket:
                bisect.insort(self._buckets, bucket)
            return bucket if creates_new_bucket else None
        except StopIteration:
            return None

    def bucket_is_empty(self, bucket_idx) -> bool:
        bucket = self._buckets[bucket_idx]
        if bucket not in self._prefetched:
            return True
        elif len(self._prefetched[bucket]) == 0:
            del self._prefetched[bucket]
            return True
        else:
            return False

    def _choose_and_prepare_bucket(self, bucket_idx=None):
        """pick a bucket (at random unless specified) and prepare examples for iteration"""
        if bucket_idx is None:
            bucket_idx = random.choice(range(len(self._buckets)))
        # if bucket_idx >= len(self._buckets):
        #     import pdb; pdb.set_trace()
        # if len(self._prefetched[self._buckets[bucket_idx]]) == 0:
        #     import pdb; pdb.set_trace()
        random.shuffle(self._prefetched[self._buckets[bucket_idx]])
        return self._buckets[bucket_idx], bucket_idx

    def is_empty(self):
        return (len(self._prefetched) == 0) or (sum(map(len, self._prefetched.values())) == 0)

    def __iter__(self):
        while True:
            # 1. maybe we've exhausted the stream and the buckets
            if self.is_empty():
                break
            accum, cur_batch_size = [], 0
            # 2. pick a length at random
            bucket, bucket_idx = self._choose_and_prepare_bucket()
            # 3. build batch
            batch_is_complete = False
            while not batch_is_complete:
                # maybe switch buckets
                if self.bucket_is_empty(bucket_idx):
                    if self.is_empty():
                        logger.info('Reached end of stream')  # should not happen
                        if accum:
                            yield self.collate_fn(accum)
                        break
                    if self._buckets[bucket_idx] == self._buckets[-1]:
                        # this was the largest bucket, so we'll need to pick the next smallest instead
                        del self._buckets[bucket_idx]
                        bucket_idx -= 1
                    else:
                        # there was a larger bucket, by dropping the current bucket we're shifting the index by one
                        del self._buckets[bucket_idx]
                    new_bucket, new_bucket_idx = self._choose_and_prepare_bucket(bucket_idx=bucket_idx)
                    bucket, bucket_idx = new_bucket, new_bucket_idx
                example = self._prefetched[bucket].pop()
                accum.append(example)
                numel = self.numel_fn(example)
                cur_batch_size += numel
                batch_is_complete = cur_batch_size >= self.batch_size

                # 4. try to replenish reservoir if possible
                new_bucket = self.maybe_replenish()
                if (new_bucket is not None) and (new_bucket <= bucket):
                    assert self._buckets[bucket_idx] != bucket
                    bucket_idx += 1

            yield self.collate_fn(accum)
            if self.bucket_is_empty(bucket_idx):
                del self._buckets[bucket_idx]


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
        bucket_size (int): accum this number of examples in a dynamic dataset;
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
        bucket_size=2048,
        skip_empty_level='warning',
        stride=1,
        offset=0,
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
        self.bucket_size = bucket_size
        if stride <= 0:
            raise ValueError(f"Invalid argument for stride={stride}.")
        self.stride = stride
        self.offset = offset
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level

    @classmethod
    def from_opts(cls, task_queue_manager, transforms_cls, vocabs_dict, opts, is_train, stride=1, offset=0):
        """Initilize `DynamicDatasetIter` with options parsed from `opts`."""
        batch_size = opts.batch_size if is_train else opts.valid_batch_size
        if opts.batch_size_multiple is not None:
            batch_size_multiple = opts.batch_size_multiple
        else:
            batch_size_multiple = 8 if opts.model_dtype == "fp16" else 1
        return cls(
            task_queue_manager,
            opts,
            opts.data,
            transforms_cls,
            vocabs_dict,
            is_train,
            opts.batch_type,
            batch_size,
            batch_size_multiple,
            data_type=opts.data_type,
            bucket_size=opts.bucket_size,
            skip_empty_level=opts.skip_empty_level,
            stride=stride,
            offset=offset,
        )

    def _init_datasets(self):
        self.dataset_iterators = []
        for task in self.task_queue_manager.get_tasks():
            src_vocab = self.vocabs_dict[('src', task.src_lang)]
            tgt_vocab = self.vocabs_dict[('tgt', task.tgt_lang)]

            metadata = task.get_serializable_metadata()

            device = torch.device(self.task_queue_manager.local_rank)
            corpus = get_corpus(
                self.opts, task.corpus_id, src_vocab, tgt_vocab, is_train=self.is_train
            ).to(device)

            # iterator over minibatches
            ordered_iter = build_dataloader(
                corpus,
                self.batch_size,
                self.batch_type,
                self.bucket_size,
            )

            self.dataset_iterators.append((ordered_iter, metadata))

        self.init_iterators = True

    def __iter__(self):
        if self.init_iterators is False:
            self._init_datasets()

        # All minibatches with the same communication_batch_id should be trained on
        # before synching gradients between devices
        communication_batch_id = 0
        while True:
            # interleaves one minibatch from each language pair, in a round-robin fashion
            for ordered_iter, metadata in self.dataset_iterators:
                yield next(ordered_iter), metadata, communication_batch_id
            communication_batch_id += 1
