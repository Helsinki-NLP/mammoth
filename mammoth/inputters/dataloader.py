import collections
import itertools
import math

import torch

from mammoth.inputters.dataset import get_corpus
from mammoth.utils.logging import logger


def build_dataloader(dataset, batch_size, batch_type, pool_size=None, n_buckets=None, cycle=True, as_iter=True):
    """Convert an mammoth.inputters.ParallelCorpus into an infinite iterator of batches"""
    if not cycle:
        loader = InferenceBatcher(dataset, batch_size)
    else:
        if batch_type == 'sents':
            raise NotImplementedError()
        elif batch_type == 'tokens':
            loader = SimpleLookAheadBucketing(
                dataset=dataset,
                max_look_ahead_size=pool_size,
                n_buckets=n_buckets,
                batch_size=batch_size,
                score_fn=SimpleLookAheadBucketing.max_of_lens,
            )
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


class ScoredInfiniteExamples():
    def __init__(self, dataset, score_fn):
        self.score_fn = score_fn if score_fn else self.max_of_lens
        self.dataset = dataset
        self._it = iter(self.dataset)
        self._prev = next(self._it)
        self._score = self.score_fn(self._prev)

    def peek_at_score(self):
        return self._score

    def next(self):
        try:
            example = next(self._it)
        except StopIteration:
            self._it = iter(self.dataset)
            example = next(self._it)
        score_out, example_out = self._score, self._prev
        self._score = self.score_fn(example)
        self._prev = example
        return score_out, example_out


class SimpleLookAheadBucketing():
    """
    Arguments:
        dataset: mammoth.inputters.ParallelCorpus
        max_look_ahead_size:
            The maximum number of sentence pairs to read before yielding minibatches.
            Limits the time spent looping if there is a corpus with unexpectedly short sentences.
        n_buckets:
            The number of minibatches that will be yielded once bucketing is complete.
            Recommended value: same as accum_count, or at least a multiple of it.
            Setting n_buckets == accum_count means that each accumulated batch uses up the whole buffer.
            All tasks stay in sync concerning the length sorting: each task begins with the smallest
            minibatch and ends with the largest just before accumulation ends.
        batch_size:
            The maximum size of each minibatch in tokens.
            Note that the maximum batch size can not be guaranteed if the data contains examples
            that exceed the limit on their own. Use filtertoolong to avoid such examples.
        score_fn:
            Compute the size estimate (single integer) for sorting examples.
    """
    def __init__(self, dataset, max_look_ahead_size, n_buckets, batch_size, score_fn=None):
        score_fn = score_fn if score_fn else self.max_of_lens
        self._sie = ScoredInfiniteExamples(dataset, score_fn)
        self.max_look_ahead_size = max_look_ahead_size
        self.batch_size = batch_size
        self.n_buckets = n_buckets
        self.multi_batch_size = n_buckets * batch_size
        self.collate_fn = dataset.collate_fn

    @staticmethod
    def max_of_lens(example_dict) -> int:
        if 'tgt' in example_dict:
            score = max(len(example_dict['src']), len(example_dict['tgt']))
        else:
            score = len(example_dict['src'])
        return score

    def __iter__(self):
        while True:
            multibatch = []
            max_score = 0
            for i in range(self.max_look_ahead_size):
                score = self._sie.peek_at_score()
                # Decide whether to add it or not
                if len(multibatch) <= self.n_buckets:
                    # Always add at least one example per minibatch
                    still_fits = True
                else:
                    still_fits = (max(max_score, score) * (len(multibatch) + 1)) < (self.multi_batch_size)
                if still_fits:
                    score, example = self._sie.next()
                    multibatch.append((score, example))
                    max_score = max(max_score, score)
                else:
                    break
            # Sort by score to reduce padding
            multibatch = list(sorted(multibatch, key=lambda x: x[0]))
            # Split into minibatches and yield
            examples_per_batch = math.ceil(len(multibatch) / self.n_buckets)
            multibatch_it = iter(multibatch)
            for _ in range(self.n_buckets):
                yield self.collate_fn(
                    [
                        example_dict for _, example_dict
                        in itertools.islice(multibatch_it, examples_per_batch)
                    ]
                )


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
                for _ in range(self.task_queue_manager.accum_count):
                    batch = next(ordered_iter)
                    if batch_task_sample.training_step == 0 and self.opts.verbose:
                        # De-numericalize a few sentences for debugging
                        logger.warning(
                            f'src shape: {batch.src[0].shape} tgt shape: {batch.tgt.shape} '
                            f'batch size: {batch.batch_size}'
                        )
                        src_vocab = self.vocabs_dict[('src', metadata.src_lang)]
                        tgt_vocab = self.vocabs_dict[('tgt', metadata.tgt_lang)]
                        for sent_idx in range(min(3, batch.src[0].shape[2])):
                            toks = [src_vocab.itos[tok_id.item()] for tok_id in batch.src[0][:, sent_idx, 0]]
                            logger.warning(f'{sent_idx} {metadata.src_lang} src: {" ".join(toks)}')
                            toks = [tgt_vocab.itos[tok_id.item()] for tok_id in batch.tgt[:, sent_idx, 0]]
                            logger.warning(f'{sent_idx} {metadata.tgt_lang} tgt: {" ".join(toks)}')
                    yield batch, metadata, batch_task_sample.training_step
