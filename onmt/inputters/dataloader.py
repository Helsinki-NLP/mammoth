import collections
import itertools
import random

import torch

from onmt.inputters.dataset import get_corpus
from onmt.utils.logging import logger


def infinite_iterator(iterable):
    return itertools.chain.from_iterable(itertools.repeat(iterable))


def build_dataloader(
    dataset,
    batch_size,
    batch_type,
    pool_size=None,
    n_buckets=None,
    cycle=True,
    as_iter=True,
    data_state=None,
):
    """Convert an onmt.inputters.ParallelCorpus into an infinite iterator of batches"""
    if not cycle:
        loader = InferenceBatcher(dataset, batch_size)
    else:
        examples_stream = infinite_iterator(dataset)
        if batch_type == 'sents':
            n_buckets = 1

            def bucket_fn(_):
                return 0

            def numel_fn(_):
                return 1

        elif batch_type == 'tokens':
            def bucket_fn(example_dict):
                if 'tgt' in example_dict:
                    # subtract four for bos/eos on both sides
                    true_size = len(example_dict['src']) + len(example_dict['tgt']) - 4
                else:
                    true_size = len(example_dict['src']) + 2
                # maybe dump it in the last bucket if it's just too long
                return min(n_buckets - 1, true_size)

            def numel_fn(example_dict):
                if 'tgt' in example_dict:
                    true_size = len(example_dict['src']) + len(example_dict['tgt'])
                else:
                    true_size = len(example_dict['src'])
                return true_size

        collate_fn = dataset.collate_fn
        loader = LookAheadBucketing(
            examples_stream, pool_size, n_buckets, batch_size, bucket_fn, numel_fn, collate_fn, data_state
            )
    return (iter(loader), loader) if as_iter else loader


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
    def __init__(
        self,
        examples_stream,
        look_ahead_size,
        n_buckets,
        batch_size,
        bucket_fn,
        numel_fn,
        collate_fn,
        data_state=None,
    ):
        self.examples_stream = examples_stream
        self.look_ahead_size = look_ahead_size
        self.batch_size = batch_size
        self.bucket_fn = bucket_fn
        self.numel_fn = numel_fn
        self.collate_fn = collate_fn
        if data_state is None or (isinstance(data_state,dict) and data_state['buckets'] is None):
            self.current_file_index = None
            self._buckets = [[] for _ in range(n_buckets)]
            self._lens = [0 for _ in range(n_buckets)]
            self._init()
        else:
            self.current_file_index = data_state['indices']
            logger.info('LookAheadBucketing: relying on pre-computed buckets, no initialization')
            self._buckets = data_state['buckets']
            self._lens = list(map(len, data_state['buckets']))


    def _init(self):
        logger.info('LookAheadBucketing: initialization start')
        for example in itertools.islice(self.examples_stream, self.look_ahead_size):
            bucket_idx = self.bucket_fn(example)
            self._buckets[bucket_idx].append(example)
            self._lens[bucket_idx] += 1
        logger.info('LookAheadBucketing: initialization done')

    def maybe_replenish(self) -> bool:
        """look up one more example to add to this reservoir."""
        try:
            example = next(self.examples_stream)
            self.current_file_index = example['idx']
            bucket_idx = self.bucket_fn(example)
            creates_new_bucket = self._lens[bucket_idx] == 0
            self._buckets[bucket_idx].append(example)
            self._lens[bucket_idx] += 1
            return creates_new_bucket
        except StopIteration:
            return None

    def bucket_is_empty(self, bucket_idx) -> bool:
        return self._lens[bucket_idx] == 0

    def _choose_and_prepare_bucket(self, bucket_idx=None):
        """pick a bucket (at random unless specified) and prepare examples for iteration"""
        if bucket_idx is None:
            bucket_idx = random.choices(range(len(self._buckets)), weights=self._lens, k=1)[0]
        # if bucket_idx >= len(self._buckets):
        #     import pdb; pdb.set_trace()
        # if len(self._prefetched[self._buckets[bucket_idx]]) == 0:
        #     import pdb; pdb.set_trace()
        random.shuffle(self._buckets[bucket_idx])
        return bucket_idx

    def is_empty(self):
        return all(size == 0 for size in self._lens)

    def get_data_state(self):
        return {'indices': self.current_file_index, 'buckets': self._buckets}

    def __iter__(self):
        while True:
            # 1. maybe we've exhausted the stream and the buckets
            if self.is_empty():
                break
            accum, cur_batch_size = [], 0
            # 2. pick a length at random
            smallest_bucket_idx = self._choose_and_prepare_bucket()
            current_bucket_idx = smallest_bucket_idx
            # 3. build batch
            batch_is_complete = False
            while not batch_is_complete:
                # maybe switch buckets
                if self.bucket_is_empty(current_bucket_idx):
                    if self.is_empty():
                        logger.info('Reached end of stream')  # should not happen
                        if accum:
                            yield self.collate_fn(accum)
                        break
                    if not any(self._lens[current_bucket_idx:]):
                        # this was the largest bucket, so we'll need to pick the next smallest instead
                        smallest_bucket_idx = next(
                            bucket_idx
                            for bucket_idx in range(smallest_bucket_idx, -1, -1)
                            if self._lens[bucket_idx] != 0
                        )
                        current_bucket_idx = smallest_bucket_idx
                    else:
                        # there was a larger bucket, shift the index by one
                        current_bucket_idx = next(
                            bucket_idx
                            for bucket_idx in range(current_bucket_idx, len(self._buckets) + 1)
                            if self._lens[bucket_idx] != 0
                        )
                    _ = self._choose_and_prepare_bucket(bucket_idx=current_bucket_idx)
                # retrieve and process the example
                example = self._buckets[current_bucket_idx].pop()
                self._lens[current_bucket_idx] -= 1
                accum.append(example)
                numel = self.numel_fn(example)
                cur_batch_size += numel
                batch_is_complete = cur_batch_size >= self.batch_size

                # 4. try to replenish reservoir if possible
                self.maybe_replenish()
                # if (new_bucket is not None) and (new_bucket <= bucket):
                #     assert self._buckets[bucket_idx] != bucket
                #     bucket_idx += 1 
            yield self.collate_fn(accum)
            # if self.bucket_is_empty(bucket_idx):
            #     del self._buckets[bucket_idx]


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
        n_buckets=1024,
        skip_empty_level='warning',
        data_state=None,
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
        self.n_buckets = n_buckets
        if skip_empty_level not in ['silent', 'warning', 'error']:
            raise ValueError(f"Invalid argument skip_empty_level={skip_empty_level}")
        self.skip_empty_level = skip_empty_level
        self.data_state = data_state

    @classmethod
    def from_opts(
        cls,
        task_queue_manager,
        transforms_cls,
        vocabs_dict,
        opts,
        is_train,
        data_state=None,
    ):
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
            n_buckets=opts.n_buckets,
            skip_empty_level=opts.skip_empty_level,
            data_state=data_state,
        )

    def _init_datasets(self):
        self.dataset_iterators = dict()
        for task in self.task_queue_manager.get_tasks():
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
            # Recover current file index from the data state
            corpus_datastate = None
            if self.data_state: 
                corpus_datastate = self.data_state.get(task.corpus_id, {'indices': 0, 'buckets': None})
                logger.info(
                    f'RESUME TRAINING: task {task.corpus_id} to resume form example num. {corpus_datastate["indices"]} in corpus'
                    )
            # Case 1: we are training, and the task must contain some path to training data
            # Case 2: we are validation (hence self.is_train := False), we need an iterator
            # if and only the task defines validation data, i.e. if the key `path_valid_src`
            # is defined
            if self.is_train or self.opts.data[task.corpus_id].get('path_valid_src', None) is not None:
                corpus = get_corpus(
                    self.opts, task, src_vocab, tgt_vocab, is_train=self.is_train, data_state=corpus_datastate,
                ).to(device)

                # iterator over minibatches
                ordered_iter, lab = build_dataloader(
                    corpus,
                    self.batch_size,
                    self.batch_type,
                    self.bucket_size,
                    n_buckets=self.n_buckets,
                    cycle=self.is_train,
                    as_iter=self.is_train,
                    data_state=corpus_datastate,
                )

                self.dataset_iterators[task.corpus_id] = (ordered_iter, lab, metadata)

        self.init_iterators = True


    def __iter__(self):
        if self.init_iterators is False:
            self._init_datasets()

        if not self.is_train:
            # to be absolutely clear: all the validation data is read per validation loop
            all_val_data = [
                zip(ordered_iter, itertools.repeat(metadata), itertools.repeat(0))
                for ordered_iter, lab, metadata in self.dataset_iterators.values()
            ]
            yield from itertools.chain.from_iterable(all_val_data)

        else:
            # All minibatches with the same communication_batch_id should be trained on
            # before synching gradients between devices
            communication_batch_id = 0
            total_num_batches = 0
            while True:
                for corpus_id in self.task_queue_manager.sample_corpus_ids(communication_batch_id):
                    ordered_iter, _, metadata = self.dataset_iterators[corpus_id]
                    batch = next(ordered_iter)
                    if communication_batch_id == 0:
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
                    total_num_batches += 1
                    # FIXME: make the following compatible with len(accum_count) > 1
                    final_step = self.opts.train_steps * self.opts.accum_count[0]
                    real_step = self.opts.save_checkpoint_steps * self.opts.accum_count[0]
                    if (total_num_batches % real_step == 0) or (total_num_batches == final_step):
                        data_states = dict()
                        for taskname, (_, lab, _) in self.dataset_iterators.items():
                            data_states[taskname] = lab.get_data_state()
                    else:
                        data_states = None
                    yield batch, metadata, communication_batch_id, data_states

                communication_batch_id += 1
                if communication_batch_id % 1000 == 0:
                    total = sum(self.task_queue_manager.sampled_task_counts.values())
                    logger.info(f'Task sampling distribution: (total {total})')
                    for task, count in self.task_queue_manager.sampled_task_counts.most_common():
                        logger.info(f'Task: {task}\tcount: {count}\t{100 * count / total} %')
                

