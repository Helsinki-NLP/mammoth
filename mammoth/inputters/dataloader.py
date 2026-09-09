import collections
import itertools
import math

import torch

from mammoth.inputters.dataset import get_corpus
from mammoth.utils.logging import logger


def task_needs_validation_dataset(is_train, path_valid_src):
    """Whether a task needs a dataset iterator built for the current mode.

    At validation time a task is included whenever it defines validation
    data, regardless of its training `weight` — weight=0 (eval-only) tasks
    must still be validated even though they're never sampled for training.
    """
    return is_train or path_valid_src is not None


def build_dataloader(
    dataset,
    batch_size,
    batch_type,
    max_look_ahead_sentences=None,
    lookahead_minibatches=None,
    cycle=True,
    as_iter=True
):
    """Convert an mammoth.inputters.ParallelCorpus into an infinite iterator of batches"""
    if not cycle:
        loader = InferenceBatcher(dataset, batch_size)
    else:
        if batch_type == 'sents':
            loader = SentenceMinibatcher(
                dataset=dataset,
                batch_size=batch_size,
            )
        elif batch_type == 'tokens':
            loader = SimpleLookAheadBucketing(
                dataset=dataset,
                max_look_ahead_sentences=max_look_ahead_sentences,
                lookahead_minibatches=lookahead_minibatches,
                batch_size=batch_size,
                score_fn=SimpleLookAheadBucketing.max_of_lens,
            )
    return iter(loader) if as_iter else loader


DatasetMetadata = collections.namedtuple(
    'DatasetMetadata',
    ['src_lang', 'tgt_lang', 'encoder_id', 'decoder_id', 'corpus_id']
)


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
                # line idx == 0 during inference
                yield self.collate_fn(accum, 0)
                accum = []
        if accum:
            yield self.collate_fn(accum, 0)


class ScoredInfiniteExamples():
    def __init__(self, dataset, score_fn):
        self.score_fn = score_fn if score_fn else self.max_of_lens
        self.dataset = dataset
        self._it = iter(self.dataset)
        self._prev = next(self._it)
        self._score = self.score_fn(self._prev)
        self._current_line_idx = self._prev['line_idx']

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
        self._current_line_idx = self._prev['line_idx']
        return score_out, example_out


class SimpleLookAheadBucketing():
    """
    Arguments:
        dataset: mammoth.inputters.ParallelCorpus
        max_look_ahead_sentences:
            The maximum number of sentence pairs to read before yielding minibatches.
            Limits the time spent looping if there is a corpus with unexpectedly short sentences.
        lookahead_minibatches:
            The number of minibatches that will be yielded once bucketing is complete.
            Recommended value: same as accum_count, or at least a multiple of it.
            Setting lookahead_minibatches == accum_count means that each accumulated batch uses up the whole buffer.
            All tasks stay in sync concerning the length sorting: each task begins with the smallest
            minibatch and ends with the largest just before accumulation ends.
        batch_size:
            The maximum size of each minibatch in tokens.
            Note that the maximum batch size can not be guaranteed if the data contains examples
            that exceed the limit on their own. Use filtertoolong to avoid such examples.
        score_fn:
            Compute the size estimate (single integer) for sorting examples.
    """
    def __init__(self, dataset, max_look_ahead_sentences, lookahead_minibatches, batch_size, score_fn=None):
        score_fn = score_fn if score_fn else self.max_of_lens
        self._sie = ScoredInfiniteExamples(dataset, score_fn)
        self.max_look_ahead_sentences = max_look_ahead_sentences
        self.batch_size = batch_size
        self.lookahead_minibatches = lookahead_minibatches
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
            maxi_batch = []
            max_score = 0
            for i in range(self.max_look_ahead_sentences):
                score = self._sie.peek_at_score()
                # Decide whether to add it or not
                if len(maxi_batch) < self.lookahead_minibatches:
                    # Always add at least one example per minibatch
                    still_fits = True
                else:
                    estimated_minibatch_size = math.ceil((len(maxi_batch) + 1) / self.lookahead_minibatches)
                    still_fits = (max(max_score, score) * estimated_minibatch_size) < (self.batch_size)
                if still_fits:
                    score, example = self._sie.next()
                    maxi_batch.append((score, example))
                    max_score = max(max_score, score)
                else:
                    break
            # Sort by score to reduce padding
            maxi_batch = list(sorted(maxi_batch, key=lambda x: x[0]))
            # Split into minibatches and yield
            floor_examples_per_batch = math.floor(len(maxi_batch) / self.lookahead_minibatches)
            examples_per_batch = [floor_examples_per_batch] * self.lookahead_minibatches
            for i in range(len(maxi_batch) % self.lookahead_minibatches):
                examples_per_batch[i] += 1
            assert all(epb > 0 for epb in examples_per_batch)
            assert sum(examples_per_batch) == len(maxi_batch)
            maxi_batch_it = iter(maxi_batch)
            for epb in examples_per_batch:
                yield self.collate_fn(
                    [
                        example_dict for _, example_dict
                        in itertools.islice(maxi_batch_it, epb)
                    ],
                    self._sie._current_line_idx,
                )


class SentenceMinibatcher():
    """
    Arguments:
        dataset: mammoth.inputters.ParallelCorpus
        batch_size:
            The maximum size of each minibatch in sentence.
    """
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.collate_fn = dataset.collate_fn
        self._sie = ScoredInfiniteExamples(dataset, score_fn=lambda x: 1)

    def __iter__(self):
        while True:
            minibatch = []
            for _ in range(self.batch_size):
                _, example = self._sie.next()
                minibatch.append(example)
            yield self.collate_fn(minibatch, self._sie._current_line_idx)


def preprocess_indexed_datasets_parallel_nodes(
    task_queue_manager,
    opts,
    current_node_rank,
    total_nodes
):
    """
    Preprocess indexed datasets using all nodes' CPUs in parallel.

    Each node is assigned different datasets using consistent hashing:
    - Node i processes datasets where hash(dataset_path) % total_nodes == i
    - No coordination needed - each node independently knows its work
    - Files appear on shared filesystem (Lustre on LUMI)

    This is CPU-only work - no GPUs involved. Each dataset uses all
    available CPU cores on the assigned node for parallel tokenization.

    Args:
        task_queue_manager: TaskQueueManager instance
        opts: Training options
        current_node_rank: This node's rank (0 to total_nodes-1)
        total_nodes: Total number of nodes

    Returns:
        None
    """
    import time
    import multiprocessing
    from mammoth.inputters.dataset import _get_indexed_path, _run_pretokenization
    from mammoth.inputters.indexed_dataset import exists as indexed_dataset_exists

    # Collect all datasets that need preprocessing
    datasets_to_preprocess = []

    for task in task_queue_manager.get_all_tasks():
        corpus_opts = opts.tasks[task.corpus_id]
        data_type = corpus_opts.get('data_type', 'text')

        if data_type != 'indexed':
            continue  # Skip non-indexed datasets

        # Check both training and validation paths
        for is_train in [True, False]:
            if is_train:
                src_path = corpus_opts.get("path_src")
                tgt_path = corpus_opts.get("path_tgt")
            else:
                src_path = corpus_opts.get("path_valid_src")
                tgt_path = corpus_opts.get("path_valid_tgt")

            if src_path:
                indexed_path = _get_indexed_path(src_path)
                if not indexed_dataset_exists(indexed_path):
                    datasets_to_preprocess.append({
                        'text_path': src_path,
                        'indexed_path': indexed_path,
                        'is_src': True,
                        'task': task,
                    })

            if tgt_path:
                indexed_path = _get_indexed_path(tgt_path)
                if not indexed_dataset_exists(indexed_path):
                    datasets_to_preprocess.append({
                        'text_path': tgt_path,
                        'indexed_path': indexed_path,
                        'is_src': False,
                        'task': task,
                    })

    # Deduplicate based on indexed_path
    unique_datasets = {}
    for dataset_info in datasets_to_preprocess:
        path = dataset_info['indexed_path']
        if path not in unique_datasets:
            unique_datasets[path] = dataset_info

    total_datasets = len(unique_datasets)

    if total_datasets == 0:
        logger.info(f"[Node {current_node_rank}] No indexed datasets need preprocessing - all files exist")
        return

    # Assign datasets to this node using consistent hashing
    # Note: Use deterministic hash (MD5) instead of Python's hash() which is randomized per-process
    import hashlib

    my_datasets = {}
    for indexed_path, dataset_info in unique_datasets.items():
        # Consistent hashing: assign dataset to node based on path hash
        # Use MD5 hash for deterministic assignment across all nodes
        path_hash = int(hashlib.md5(indexed_path.encode('utf-8')).hexdigest(), 16)
        assigned_node = path_hash % total_nodes
        if assigned_node == current_node_rank:
            my_datasets[indexed_path] = dataset_info

    num_my_datasets = len(my_datasets)

    # Get number of CPU cores to use (use all available cores)
    num_workers = max(1, multiprocessing.cpu_count())

    logger.info("=" * 80)
    logger.info(f"[Node {current_node_rank}/{total_nodes}] Parallel CPU-only pretokenization")
    logger.info(f"Total datasets needing preprocessing: {total_datasets}")
    logger.info(f"Datasets assigned to this node: {num_my_datasets}")
    logger.info(f"Using {num_workers} CPU workers per dataset")
    logger.info(f"Note: This is CPU work, not using GPUs")

    # Log which datasets are assigned to this node
    if num_my_datasets > 0:
        logger.info(f"[Node {current_node_rank}] Assigned datasets:")
        for i, indexed_path in enumerate(my_datasets.keys(), 1):
            logger.info(f"  {i}. {indexed_path}")

    logger.info("=" * 80)

    if num_my_datasets == 0:
        logger.info(f"[Node {current_node_rank}] No datasets assigned to this node, waiting for others...")
        # Still need to wait for other nodes to finish
        _wait_for_all_datasets(unique_datasets, current_node_rank)
        return

    # Process datasets assigned to this node
    processed = 0
    for indexed_path, dataset_info in my_datasets.items():
        processed += 1
        logger.info(
            f"[Node {current_node_rank}] Processing {processed}/{num_my_datasets}: "
            f"{dataset_info['text_path']}"
        )

        try:
            _run_pretokenization(
                opts,
                dataset_info['task'],
                dataset_info['text_path'],
                dataset_info['indexed_path'],
                dataset_info['is_src']
            )
            logger.info(
                f"[Node {current_node_rank}] Completed {processed}/{num_my_datasets}: "
                f"{indexed_path}"
            )
        except Exception as e:
            logger.error(
                f"[Node {current_node_rank}] Failed to preprocess dataset {processed}: "
                f"{dataset_info['text_path']}: {e}"
            )
            raise

    logger.info("=" * 80)
    logger.info(
        f"[Node {current_node_rank}] Finished preprocessing {num_my_datasets} datasets. "
        f"Waiting for other nodes..."
    )
    logger.info("=" * 80)

    # Wait for all datasets to be ready (other nodes may still be processing)
    _wait_for_all_datasets(unique_datasets, current_node_rank)

    logger.info("=" * 80)
    logger.info(f"[Node {current_node_rank}] All {total_datasets} datasets ready!")
    logger.info("=" * 80)


def _wait_for_all_datasets(all_datasets, node_rank):
    """
    Wait for all datasets to be preprocessed by polling filesystem.

    Args:
        all_datasets: Dict of all datasets that need preprocessing
        node_rank: Current node rank for logging
    """
    import time
    from mammoth.inputters.indexed_dataset import exists as indexed_dataset_exists

    max_wait_seconds = 3600  # 1 hour timeout
    check_interval = 5  # Check every 5 seconds
    elapsed = 0

    missing_datasets = list(all_datasets.keys())

    while missing_datasets:
        # Check which datasets still don't exist
        still_missing = []
        for indexed_path in missing_datasets:
            if not indexed_dataset_exists(indexed_path):
                still_missing.append(indexed_path)

        if not still_missing:
            # All datasets ready!
            break

        missing_datasets = still_missing

        if elapsed >= max_wait_seconds:
            raise TimeoutError(
                f"[Node {node_rank}] Timeout waiting for preprocessing to complete.\n"
                f"Still missing {len(missing_datasets)} datasets after {max_wait_seconds}s:\n"
                f"{missing_datasets[:5]}..."  # Show first 5
            )

        if elapsed % 30 == 0 and elapsed > 0:  # Log every 30 seconds
            logger.info(
                f"[Node {node_rank}] Waiting for {len(missing_datasets)} datasets "
                f"(elapsed: {elapsed}s)..."
            )
            # Log which specific datasets are missing
            for path in missing_datasets[:5]:  # Show first 5
                logger.info(f"[Node {node_rank}]   Missing: {path}")

        time.sleep(check_interval)
        elapsed += check_interval


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
        max_look_ahead_sentences (int): accum this number of examples in a dynamic dataset;
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
        max_look_ahead_sentences=2048,
        lookahead_minibatches=4,
        line_idx_restore=None,
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
        self.max_look_ahead_sentences = max_look_ahead_sentences
        self.lookahead_minibatches = lookahead_minibatches
        self.line_idx_restore = dict() if line_idx_restore is None else line_idx_restore

    @classmethod
    def from_opts(cls, task_queue_manager, transforms_cls, vocabs_dict, opts, is_train, line_idx_restore):
        """Initilize `DynamicDatasetIter` with options parsed from `opts`."""
        batch_size = opts.batch_size if is_train else opts.valid_batch_size
        if opts.batch_size_multiple is not None:
            batch_size_multiple = opts.batch_size_multiple
        else:
            batch_size_multiple = 8 if opts.model_dtype in ["fp16", "bf16"] else 1
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
            max_look_ahead_sentences=opts.max_look_ahead_sentences,
            lookahead_minibatches=opts.lookahead_minibatches,
            line_idx_restore=line_idx_restore,
        )

    def _init_datasets(self):
        # Note: Preprocessing is now done centrally in train.py before producer is spawned
        # No need to do it here anymore
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
            path_valid_src = self.opts.tasks[task.corpus_id].get('path_valid_src', None)
            if task_needs_validation_dataset(self.is_train, path_valid_src):
                corpus = get_corpus(
                    self.opts,
                    task,
                    src_vocab,
                    tgt_vocab,
                    is_train=self.is_train,
                    line_idx_restore=self.line_idx_restore.get(task.corpus_id, None),
                    device_rank=self.task_queue_manager.device_context.global_rank,
                ).to(device)

                # iterator over minibatches
                ordered_iter = build_dataloader(
                    corpus,
                    self.batch_size,
                    self.batch_type,
                    self.max_look_ahead_sentences,
                    lookahead_minibatches=self.lookahead_minibatches,
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
                        # FIXME should be debug, not warn
                        logger.warning(
                            f'src shape: {batch.src.tensor.shape} tgt shape: {batch.tgt.tensor.shape} '
                            f'batch size: {batch.batch_size}'
                        )
                        src_vocab = self.vocabs_dict[('src', metadata.src_lang)]
                        tgt_vocab = self.vocabs_dict[('tgt', metadata.tgt_lang)]
                        for sent_idx in range(min(3, batch.src.tensor.shape[2])):
                            toks = [src_vocab.itos[tok_id.item()] for tok_id in batch.src.tensor[:, sent_idx, 0]]
                            logger.warning(f'{sent_idx} {metadata.src_lang} src: {" ".join(toks)}')
                            toks = [tgt_vocab.itos[tok_id.item()] for tok_id in batch.tgt.tensor[:, sent_idx, 0]]
                            logger.warning(f'{sent_idx} {metadata.tgt_lang} tgt: {" ".join(toks)}')
                    yield batch, metadata, batch_task_sample.training_step
