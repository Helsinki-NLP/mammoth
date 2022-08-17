import collections
import itertools
import warnings

import torch
from torch.utils.data import BatchSampler, SequentialSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence

from onmt.constants import DefaultTokens


def infinite_iterator(iterable):
    return itertools.chain.from_iterable(itertools.repeat(iterable))


# for compliance with previous code
Batch = collections.namedtuple('Batch', 'src tgt')


class TokenBatchSampler(BatchSampler):
    """
    A Sampler that batches data so that tokens in examples total at or below batch size.
    """
    def __init__(self, source_iterable, batch_size):
        self.source = source_iterable
        self.batch_size = batch_size

    # FIXME: unsure whether necessary
    def __len__(self):
        return float('inf')

    def __iter__(self):
        accum, cur_batch_size = [], 0
        for example in self.source:
            length = len(example['src']) + len(example['tgt'])
            if length > self.batch_size:
                warnings.warn(f'Example {example} larger than requested batch size, dropping it.')
            else:
                if cur_batch_size + length > self.batch_size:
                    yield accum
                    accum, cur_batch_size = [], 0
                cur_batch_size += length
                # FIXME: iterable-style dataset quirk: only considers the number of calls to __next__.
                # For this impl to work, we need to guarantee that the iteration in the sampler is the same as in the
                # dataset; so this is a fairly brittle solution for now.
                accum.append(examples)
        if accum:
            yield accum

def build_dataloader(dataset, batch_size, batch_type):
    """Convert an onmt.inputters_mvp.ParallelCorpus into an infinite iterator of batches"""

    if batch_type == 'tokens':
        batch_sampler = TokenBatchSampler(dataset, batch_size)
    else:
        batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    src_pad_idx = dataset.vocabs['src'][DefaultTokens.PAD]
    tgt_pad_idx = dataset.vocabs['tgt'][DefaultTokens.PAD]
    device = dataset.device

    # FIXME: some RNN archs require sorting src's by length
    def collate_fn(examples):
        src_lengths = torch.tensor([ex['src'].numel() for ex in examples], device=device)
        src = (pad_sequence([ex['src'] for ex in examples], pad_value=src_pad_idx), src_lengths)
        tgt = pad_sequence([ex['src'] for ex in examples], pad_value=src_pad_idx)
        return Batch(src, tgt)

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)
    return infinite_iterator(dataloader)


DatasetMetadata = collections.namedtuple('DatasetMetadata', 'src_lang tgt_lang encoder_id decoder_id corpus_id')

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
        vocabs_dict,
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
        self.vocabs_dict = vocabs_dict
        self.corpora_info = corpora_info
        self.is_train = is_train
        self.init_iterators = False
        self.batch_size = batch_size
        self.batch_type = batch_type
        # self.batch_size_fn = max_tok_len if batch_type == "tokens" else None
        self.batch_size_multiple = batch_size_multiple
        self.device = 'cpu'
        # self.sort_key = str2sortkey[data_type]
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
    def from_opts(cls, scheduler, transforms_cls, vocabs_dict, opts, is_train, stride=1, offset=0):
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
            vocabs_dict,
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
        for tpl in self.scheduler.get_dataset_specs():
            (src_lang, tgt_lang, encoder_id, decoder_id, corpus_id, corpus) = tpl
            # , src_fields, tgt_fields) = tpl
            # merged_fields = {'src': src_fields['src'], 'tgt': tgt_fields['tgt']}
            # logger.debug(f'merged_fields {merged_fields}')

            metadata = DatasetMetadata(
                src_lang=src_lang, tgt_lang=tgt_lang, encoder_id=encoder_id, decoder_id=decoder_id, corpus_id=corpus_id
            )

            # logger.debug(f'self.transforms_cls {self.transforms_cls}')
            # if self.transforms_cls:
            #     transforms = make_transforms(self.opts, self.transforms_cls, merged_fields)
            # else:
            #     print('No transforms defined')
            #     transforms = []

            ordered_iter = build_dataloader(corpus, self.batch_size, self.batch_type)

            self.dataset_iterators.append((ordered_iter, metadata))

        self.init_iterators = True

    # def _bucketing(self, iterable):
    #     buckets = torchtext_batch(iterable, batch_size=self.bucket_size, batch_size_fn=None)
    #     yield from buckets
    #
    # def _wrap_in_ordered_iterator(self, transformed_iter):
    #     for bucket_dataset in transformed_iter:
    #         train_iter = OrderedIterator(
    #             bucket_dataset,.to(torch.device(local_rank)
    #             self.batch_size,
    #             pool_factor=self.pool_factor,
    #             batch_size_fn=self.batch_size_fn,
    #             batch_size_multiple=self.batch_size_multiple,
    #             device=self.device,
    #             train=self.is_train,
    #             sort=False,
    #             sort_within_batch=True,
    #             sort_key=self.sort_key,
    #             repeat=False,
    #         )
    #         yield from train_iter

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
