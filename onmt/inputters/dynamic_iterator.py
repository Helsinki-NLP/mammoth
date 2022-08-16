"""Module that contain iterator used for dynamic data."""
from collections import namedtuple
from itertools import cycle, chain, repeat

from torchtext.legacy.data import batch as torchtext_batch
from onmt.inputters import str2sortkey, max_tok_len, OrderedIterator
from onmt.inputters.corpus import build_corpora_iter, DatasetAdapter
from onmt.transforms import make_transforms
from onmt.utils.logging import logger


DatasetMetadata = namedtuple('DatasetMetadata', 'src_lang tgt_lang encoder_id decoder_id corpus_id')


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
        pool_factor (int): accum this number of batch before sorting;
        skip_empty_level (str): security level when encouter empty line;
        stride (int): iterate data files with this stride;
        offset (int): iterate data files with this offset.

    Attributes:
        batch_size_fn (function): functions to calculate batch_size;
        sort_key (function): functions define how to sort examples;
        dataset_adapter (DatasetAdapter): organize raw corpus to tensor adapt;
        mixer (MixingStrategy): the strategy to iterate corpora.
    """

    def __init__(
        self,
        scheduler,
        opts,
        corpora_info,
        transforms_cls,
        fields_dict,
        is_train,
        batch_type,
        batch_size,
        batch_size_multiple,
        data_type="text",
        bucket_size=2048,
        pool_factor=8192,
        skip_empty_level='warning',
        stride=1,
        offset=0,
    ):
        self.scheduler = scheduler
        self.opts = opts
        self.transforms_cls = transforms_cls
        self.fields_dict = fields_dict
        self.corpora_info = corpora_info
        self.is_train = is_train
        self.init_iterators = False
        self.batch_size = batch_size
        self.batch_size_fn = max_tok_len if batch_type == "tokens" else None
        self.batch_size_multiple = batch_size_multiple
        self.device = 'cpu'
        self.sort_key = str2sortkey[data_type]
        self.bucket_size = bucket_size
        self.pool_factor = pool_factor
        if stride <= 0:
            raise ValueError(f"Invalid argument for stride={stride}.")
        self.stride = stride
        self.offset = offset
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level

    @classmethod
    def from_opts(cls, scheduler, transforms_cls, fields_dict, opts, is_train, stride=1, offset=0):
        """Initilize `DynamicDatasetIter` with options parsed from `opts`."""
        batch_size = opts.batch_size if is_train else opts.valid_batch_size
        if opts.batch_size_multiple is not None:
            batch_size_multiple = opts.batch_size_multiple
        else:
            batch_size_multiple = 8 if opts.model_dtype == "fp16" else 1
        return cls(
            scheduler,
            opts,
            opts.data,
            transforms_cls,
            fields_dict,
            is_train,
            opts.batch_type,
            batch_size,
            batch_size_multiple,
            data_type=opts.data_type,
            bucket_size=opts.bucket_size,
            pool_factor=opts.pool_factor,
            skip_empty_level=opts.skip_empty_level,
            stride=stride,
            offset=offset,
        )

    def _init_datasets(self):
        self.dataset_iterators = []
        for tpl in self.scheduler.get_dataset_specs(self.fields_dict):
            (src_lang, tgt_lang, encoder_id, decoder_id, corpus_id, corpus, src_fields, tgt_fields) = tpl
            merged_fields = {'src': src_fields['src'], 'tgt': tgt_fields['tgt']}
            logger.debug(f'merged_fields {merged_fields}')

            metadata = DatasetMetadata(
                src_lang=src_lang, tgt_lang=tgt_lang, encoder_id=encoder_id, decoder_id=decoder_id, corpus_id=corpus_id
            )

            logger.debug(f'self.transforms_cls {self.transforms_cls}')
            if self.transforms_cls:
                transforms = make_transforms(self.opts, self.transforms_cls, merged_fields)
            else:
                print('No transforms defined')
                transforms = []

            raw_iter = build_corpora_iter(
                corpus_id,
                corpus,
                transforms,
                self.corpora_info[corpus_id],
                skip_empty_level=self.skip_empty_level,
                stride=self.stride,
                offset=self.offset,
            )

            # We repeat the raw_iter object (an instance of ParallelCorpusIterator), rather
            # than cycling and cacheing through its __iter__ function. This avoids loading
            # the full corpus in memory.
            infinite_iter = chain.from_iterable(repeat(raw_iter))

            # iterator over lists of strings
            # each list is a bucket, not a minibatch
            bucketed_iter = self._bucketing(infinite_iter)

            # iterator over single-bucket torchtext datasets
            # transforms are applied here
            dataset_adapter = DatasetAdapter(merged_fields, self.is_train)
            transformed_iter = dataset_adapter.wrap(bucketed_iter)

            # iterator over minibatches
            ordered_iter = self._wrap_in_ordered_iterator(transformed_iter)

            self.dataset_iterators.append((ordered_iter, metadata))

        self.init_iterators = True

    def _bucketing(self, iterable):
        buckets = torchtext_batch(iterable, batch_size=self.bucket_size, batch_size_fn=None)
        yield from buckets

    def _wrap_in_ordered_iterator(self, transformed_iter):
        for bucket_dataset in transformed_iter:
            train_iter = OrderedIterator(
                bucket_dataset,
                self.batch_size,
                pool_factor=self.pool_factor,
                batch_size_fn=self.batch_size_fn,
                batch_size_multiple=self.batch_size_multiple,
                device=self.device,
                train=self.is_train,
                sort=False,
                sort_within_batch=True,
                sort_key=self.sort_key,
                repeat=False,
            )
            yield from train_iter

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
