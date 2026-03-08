import collections
import gzip
import itertools
from dataclasses import dataclass
from functools import partial
from io import IOBase

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from mammoth.constants import DefaultTokens
from mammoth.transforms import TransformPipe, get_transforms_cls, make_transforms
from mammoth.utils.logging import logger
from mammoth.inputters.vocab import Vocab
from mammoth.inputters.indexed_corpus import IndexedCorpus


TensorWithMask = collections.namedtuple('TensorWithMask', ['tensor', 'mask'])


@dataclass
class Batch():
    src: TensorWithMask
    tgt: TensorWithMask
    labels: Tensor
    batch_size: int
    line_idx: int

    def to(self, device, non_blocking=False):
        """
        Transfer batch to device.

        Args:
            device: Target device (cpu or cuda)
            non_blocking: If True, use async transfer (requires pinned memory)
        """
        self.src = TensorWithMask(
            self.src.tensor.to(device, non_blocking=non_blocking),
            self.src.mask.to(device, non_blocking=non_blocking)
        )
        if self.tgt is not None:
            self.tgt = TensorWithMask(
                self.tgt.tensor.to(device, non_blocking=non_blocking),
                self.tgt.mask.to(device, non_blocking=non_blocking)
            )
        if self.labels is not None:
            self.labels = self.labels.to(device, non_blocking=non_blocking)
        return self

    def pin_memory(self):
        """
        Pin all tensors in this batch to enable faster CPU->GPU transfers.
        Should be called on CPU tensors before transfer to GPU.
        """
        self.src = TensorWithMask(
            self.src.tensor.pin_memory(),
            self.src.mask.pin_memory()
        )
        if self.tgt is not None:
            self.tgt = TensorWithMask(
                self.tgt.tensor.pin_memory(),
                self.tgt.mask.pin_memory()
            )
        if self.labels is not None:
            self.labels = self.labels.pin_memory()
        return self


def read_examples_from_files(
    src_path,
    tgt_path,
    tokenize_fn=str.split,
    transforms_fn=lambda x: x,
    stride=None,
    offset=None,
    is_train=False,
    verbose_dataloader=False,
    device_rank=0,
):
    """Helper function to read examples"""

    # match starting line with offset
    # we need step, because _make_example_dict is applied after slicing the stride
    line_idx_generator = itertools.count(
        offset if offset is not None else 0,
        step=stride if stride is not None else 1
    )

    def _make_example_dict(packed):
        """Helper function to convert lines to dicts"""
        src_str, tgt_str = packed
        line_idx = next(line_idx_generator)

        # Log first 5 lines of this training session for dataset continuation testing
        # Only log during training, not validation (validation always resets to line 1)
        # By default, only master device (rank 0) logs; set verbose_dataloader=True for all devices
        start_line = offset if offset is not None else 0
        if is_train and line_idx < start_line + 5 and (verbose_dataloader or device_rank == 0):
            logger.info(
                f"[DataLoader Rank {device_rank}] Line {line_idx + 1}: "
                f"SRC={src_str.strip()[:100]} "
                f"TGT={tgt_str.strip()[:100] if tgt_str else 'None'}"
            )

        return {
            'src': tokenize_fn(src_str, side='src'),
            'tgt': tokenize_fn(tgt_str, side='tgt') if tgt_str is not None else None,
            # 'align': None,
            'line_idx': line_idx,
        }

    if isinstance(src_path, IOBase):
        src_fh = src_path
    elif src_path.endswith('.gz'):
        src_fh = gzip.open(src_path, 'rt')
    else:
        src_fh = open(src_path, 'rt')
    if tgt_path is None:
        tgt_fh = itertools.repeat(None)
    elif isinstance(tgt_path, IOBase):
        tgt_fh = src_path
    elif tgt_path.endswith('.gz'):
        tgt_fh = gzip.open(tgt_path, 'rt')
    else:
        tgt_fh = open(tgt_path, 'rt')

    examples = zip(src_fh, tgt_fh)
    if stride is not None and offset is not None:
        # Start by skipping offset examples. After that return every stride:th example.
        examples = itertools.islice(examples, offset, None, stride)
    elif offset is not None:
        # No stride, but we need to skip to the offset position for dataset continuation
        examples = itertools.islice(examples, offset, None)
    examples = map(_make_example_dict, examples)
    examples = map(transforms_fn, examples)
    examples = filter(None, examples)  # filtertoolong replaces invalid examples with None
    yield from examples

    src_fh.close()
    if tgt_path is not None:
        tgt_fh.close()


class ParallelCorpus(IterableDataset):
    """Torch-style dataset"""

    def __init__(
        self,
        src_file,
        tgt_file,
        src_vocab,
        tgt_vocab,
        transforms,
        device='cpu',
        stride=None,
        offset=None,
        is_train=False,
        task=None,
        max_length=None,
        line_idx_restore=None,
        model_max_seq_len=None,
        verbose_dataloader=False,
        device_rank=0,
    ):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.transforms = transforms
        self.device = device
        self.vocabs = {
            'src': src_vocab,
            'tgt': tgt_vocab,
        }
        self.stride = stride
        self.offset = offset
        self.is_train = is_train
        self.corpus_id = task.corpus_id
        self.task_prefix_token = task.task_prefix_token if task is not None else None
        self.max_length = max_length # for padding
        self.model_max_seq_len = model_max_seq_len
        self._line_idx_restore = line_idx_restore
        self.verbose_dataloader = verbose_dataloader
        self.device_rank = device_rank

    def _tokenize(self, string, side='src'):
        """
        Split string into tokens.

        If using HFTokenizerVocab, applies subword tokenization directly.
        Otherwise, performs simple whitespace splitting for word-level tokens.
        """
        vocab = self.vocabs[side]
        from mammoth.inputters.vocab import HFTokenizerVocab

        if isinstance(vocab, HFTokenizerVocab):
            # Use HF tokenizer to get subword tokens
            return vocab.tokenize(string, is_train=self.is_train)
        else:
            # Traditional word-level tokenization (whitespace split)
            return string.split()

    def _maybe_numericalize(self, key, value):
        """Convert list of strings into list of indices"""
        if key not in ('src', 'tgt'):
            return value
        tokens, side = value, key
        vocab = self.vocabs[side]
        bos = vocab[DefaultTokens.BOS]
        eos = vocab[DefaultTokens.EOS]
        unk = vocab[DefaultTokens.UNK]

        # Check if using HuggingFace tokenizer
        from mammoth.inputters.vocab import HFTokenizerVocab
        if isinstance(vocab, HFTokenizerVocab):
            # For HF tokenizers, tokens are already tokenized subwords (e.g., ['▁Hello', '▁world'])
            # Look up each token's ID directly instead of re-encoding
            # (Re-encoding would treat '▁' as literal text and produce wrong tokenization)
            token_ids = []
            for token in tokens:
                token_id = vocab.tokenizer.token_to_id(token)
                if token_id is None:
                    # Token not in vocabulary, use UNK
                    token_ids.append(unk)
                else:
                    token_ids.append(token_id)

            # Strip HF-added special tokens that conflict with MAMMOTH's BOS/EOS
            # Different HF models use different special tokens:
            # - BERT/ModernBERT: [CLS], [SEP]
            # We strip common ones to avoid double-wrapping
            special_tokens_to_strip = set()

            # Collect special token IDs (if they exist in tokenizer)
            for token_str in ['[CLS]', '[SEP]', '<s>', '</s>', '<bos>', '<eos>']:
                token_id = vocab.tokenizer.token_to_id(token_str)
                if token_id is not None:
                    special_tokens_to_strip.add(token_id)

            # Track which tokens were actually stripped
            actually_stripped = []

            # Strip from beginning
            while token_ids and token_ids[0] in special_tokens_to_strip:
                actually_stripped.append(token_ids[0])
                token_ids = token_ids[1:]

            # Strip from end
            while token_ids and token_ids[-1] in special_tokens_to_strip:
                actually_stripped.append(token_ids[-1])
                token_ids = token_ids[:-1]

            # Log a few examples for debugging
            # import random
            # if random.random() < 0.0001:  # Log ~0.1% of examples
            #     logger.info(f'HF Tokenizer {side} direct lookup example:')
            #     logger.info(f'  Input tokens: {tokens[:10]}...')
            #     logger.info(f'  Token IDs: {token_ids[:10]}...')
            #     if actually_stripped:
            #         logger.info(f'  Stripped special tokens: {actually_stripped}')

            # Resolve task prefix token ID for tgt side
            task_prefix_ids = []
            if side == 'tgt' and self.task_prefix_token is not None:
                task_token_id = vocab.tokenizer.token_to_id(self.task_prefix_token)
                if task_token_id is None:
                    logger.warning(
                        f"task_prefix_token '{self.task_prefix_token}' not found in tgt vocabulary. "
                        "Task conditioning will not work. Add it as a special token."
                    )
                else:
                    task_prefix_ids = [task_token_id]

            # BART-specific: decoder sequences start with </s> (EOS) then <s> (BOS)
            # For BART decoder: [</s>, <s>, tokens..., </s>]
            # For others: [<s>, tokens..., </s>]
            if side == 'tgt' and hasattr(vocab, 'decoder_start_with_eos') and vocab.decoder_start_with_eos:
                indices = torch.tensor([eos, bos, *task_prefix_ids, *token_ids, eos], device='cpu')
            else:
                indices = torch.tensor([bos, *task_prefix_ids, *token_ids, eos], device='cpu')

            # Debug: Catch sequences that will exceed positional embedding limit
            # final_length = len(indices)
            # if self.model_max_seq_len is not None:
            #     if final_length > self.model_max_seq_len:
            #         logger.error(
            #             f"❌ SEQUENCE TOO LONG AFTER NUMERICALIZATION! {side.upper()}\n"
            #             f"   Token count: {len(tokens)}\n"
            #             f"   Token IDs: {len(token_ids)}\n"
            #             f"   Final tensor length (with special tokens): {final_length}\n"
            #             f"   Exceeds limit by: {final_length - self.model_max_seq_len} tokens\n"
            #             f"   First 30 tokens: {tokens[:30]}\n"
            #             f"   Last 30 tokens: {tokens[-30:]}\n"
            #         )
            #     elif final_length > self.model_max_seq_len - 5:
            #         logger.warning(
            #             f"⚠️  Sequence near limit after numericalization. {side}: "
            #             f"tokens={len(tokens)} → token_ids={len(token_ids)} → final={final_length}"
            #         )
        else:
            # Traditional vocab: lookup tokens individually
            task_prefix_ids = []
            if side == 'tgt' and self.task_prefix_token is not None:
                task_token_id = vocab.stoi.get(self.task_prefix_token, None)
                if task_token_id is None:
                    logger.warning(
                        f"task_prefix_token '{self.task_prefix_token}' not found in tgt vocabulary. "
                        "Task conditioning will not work. Add it as a special token."
                    )
                else:
                    task_prefix_ids = [task_token_id]
            indices = torch.tensor([
                bos,
                *task_prefix_ids,
                *(vocab.stoi.get(token, unk) for token in tokens),
                eos,
            ], device='cpu')
        return indices

    def _pad_sequence(self, tensors: list, padding_value: int = 0):
        padded = None
        if self.max_length is not None:
            padded = torch.full((self.max_length, len(tensors)), padding_value, device='cpu')
            for idx, tensor in enumerate(tensors):
                if tensor.numel() > self.max_length:
                    tensor = tensor[:self.max_length]
                padded[:tensor.numel(), idx] = tensor
        else:
            padded = pad_sequence(tensors, padding_value=padding_value)
        return padded.unsqueeze(-1)

    def to(self, device):
        self.device = device
        return self

    def __iter__(self):
        """Read file, produce batches of examples"""

        def _cast(example_dict):
            return {
                k: self._maybe_numericalize(k, v)
                for k, v in example_dict.items()
                if v is not None
            }

        # ensure we only restore the first time the corpus is restored
        if self._line_idx_restore is not None:
            logger.warning(f'restoring {self.corpus_id} to line: {self._line_idx_restore}')
            if self.stride is not None:
                # sanity check
                assert (self._line_idx_restore - self.offset) % self.stride == 0, \
                    f'Stride {self.stride} is inconsistent with data restoration index {self._line_idx_restore} ' \
                    'and original offset {self.offset}. ({self.corpus_id})'
            offset = self._line_idx_restore
            self._line_idx_restore = None
        else:
            offset = self.offset

        examples = read_examples_from_files(
            self.src_file,
            self.tgt_file,
            tokenize_fn=self._tokenize,
            transforms_fn=(
                partial(
                    self.transforms.apply,
                    is_train=self.is_train,
                    corpus_name=self.corpus_id,
                )
                if self.transforms is not None else lambda x: x
            ),
            stride=self.stride,
            offset=offset,
            is_train=self.is_train,
            verbose_dataloader=self.verbose_dataloader,
            device_rank=self.device_rank,
        )
        examples = map(_cast, examples)
        yield from examples

    def collate_fn(self, examples, line_idx):
        has_tgt = 'tgt' in examples[0].keys()
        src_padding_idx = self.vocabs['src'][DefaultTokens.PAD]
        tgt_padding_idx = self.vocabs['tgt'][DefaultTokens.PAD]
        src = self._pad_sequence([ex['src'] for ex in examples], padding_value=src_padding_idx)
        src_mask = src[:, :, 0].ne(src_padding_idx)
        if has_tgt:
            tgt = self._pad_sequence([ex['tgt'] for ex in examples], padding_value=tgt_padding_idx)
            tgt_mask = tgt[:, :, 0].ne(tgt_padding_idx)
            if 'labels' not in examples[0].keys():
                labels = tgt
            else:
                labels = self._pad_sequence([ex['labels'] for ex in examples], padding_value=tgt_padding_idx)
            tgt_with_mask = TensorWithMask(tgt, tgt_mask)
        else:
            tgt_with_mask = None
            labels = None
        batch = Batch(TensorWithMask(src, src_mask), tgt_with_mask, labels, len(examples), line_idx)
        return batch


def _get_indexed_path(text_path):
    """
    Convert text file path to indexed dataset path prefix.

    Examples:
        /data/train.txt -> /data/train
        /data/train.txt.gz -> /data/train
        /data/train -> /data/train

    Args:
        text_path: Path to text file

    Returns:
        Path prefix for indexed files (without .bin/.idx extension)
    """
    path = str(text_path)
    # Remove common extensions
    for ext in ['.txt.gz', '.txt', '.gz']:
        if path.endswith(ext):
            path = path[:-len(ext)]
            break
    return path


def _check_and_create_indexed(opts, task, text_path, is_src, device_rank):
    """
    Check if indexed files exist, and create them if needed (single-process only).

    In distributed mode, preprocessing should already be done by
    _preprocess_all_indexed_datasets() in dataloader.py before this is called.

    In single-process mode, this function will preprocess on-the-fly.

    Args:
        opts: Configuration options
        task: TaskSpecs object
        text_path: Path to text file
        is_src: True if source side, False if target side
        device_rank: Global rank of current device

    Returns:
        Path prefix for indexed files
    """
    import os
    from mammoth.inputters.indexed_dataset import exists as indexed_dataset_exists
    from mammoth.utils.logging import logger

    # Get indexed path prefix
    indexed_path = _get_indexed_path(text_path)

    # Check if indexed files already exist
    if indexed_dataset_exists(indexed_path):
        return indexed_path

    # Check if running in distributed mode
    try:
        import torch.distributed as dist
        is_distributed = dist.is_initialized()
    except (ImportError, AttributeError):
        is_distributed = False

    if is_distributed:
        # In distributed mode, preprocessing should already be done
        # If we reach here, something went wrong
        raise RuntimeError(
            f"[Rank {device_rank}] Indexed files not found: {indexed_path}\n"
            f"Expected files: {indexed_path}.bin / {indexed_path}.idx\n"
            f"These files should have been created during the preprocessing phase.\n"
            f"This indicates a bug in the preprocessing logic."
        )

    # Single-process mode: preprocess on-the-fly
    if not os.path.exists(text_path):
        raise FileNotFoundError(
            f"Neither indexed files nor source text file found:\n"
            f"  Text file: {text_path}\n"
            f"  Indexed files: {indexed_path}.bin / {indexed_path}.idx\n"
            f"Please provide either pre-tokenized indexed files or a text file for preprocessing."
        )

    logger.info(f"Starting pretokenization: {text_path}")
    _run_pretokenization(opts, task, text_path, indexed_path, is_src)

    # Verify files were created successfully
    if not indexed_dataset_exists(indexed_path):
        raise RuntimeError(
            f"Pretokenization failed: indexed files not found at {indexed_path}\n"
            f"Expected files: {indexed_path}.bin and {indexed_path}.idx"
        )

    logger.info(f"Using indexed files: {indexed_path}")
    return indexed_path


def _run_pretokenization(opts, task, text_path, output_prefix, is_src):
    """
    Run pretokenization for a text file.

    Args:
        opts: Configuration options
        task: TaskSpecs object
        text_path: Path to input text file
        output_prefix: Path prefix for output files
        is_src: True if source side, False if target side
    """
    import os
    import multiprocessing
    from mammoth.inputters.indexed_dataset import preprocess_text_to_indexed
    from mammoth.utils.logging import logger

    # Determine language and vocab
    lang = task.src_lang if is_src else task.tgt_lang
    vocab_dict = opts.src_vocab if is_src else opts.tgt_vocab

    if lang not in vocab_dict:
        raise ValueError(
            f"Vocabulary not found for language '{lang}'\n"
            f"Available vocabularies: {list(vocab_dict.keys())}"
        )

    vocab_path = vocab_dict[lang]

    # Validate vocab file exists
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocabulary file not found: {vocab_path}\n"
            f"Please ensure the vocabulary file exists for language '{lang}'"
        )

    # Determine worker count
    workers = max(1, multiprocessing.cpu_count() // 2)

    # Get decoder_start_with_eos setting
    decoder_start_with_eos = getattr(opts, 'decoder_start_with_eos', False)

    # Run preprocessing
    logger.info("Pretokenization settings:")
    logger.info(f"  Input: {text_path}")
    logger.info(f"  Output: {output_prefix}")
    logger.info(f"  Vocab: {vocab_path}")
    logger.info(f"  Language: {lang}")
    logger.info(f"  Workers: {workers}")

    success = preprocess_text_to_indexed(
        input_path=text_path,
        output_prefix=output_prefix,
        vocab_path=vocab_path,
        lang=lang,
        decoder_start_with_eos=decoder_start_with_eos,
        workers=workers,
    )

    if not success:
        raise RuntimeError(f"Pretokenization failed for {text_path}")


def get_corpus(
    opts,
    task,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    is_train: bool = False,
    line_idx_restore: int = None,
    device_rank: int = 0,
):
    """build an iterable Dataset object"""
    # Auto-add task_prefix_token to target vocab if configured but missing.
    # Rank 0 is responsible for writing the updated vocab back to disk so that
    # inference (which loads from disk) sees the same tokens the model was trained with.
    # Other ranks only update their in-memory copy; they will read the correct vocab
    # from disk on the next run once rank 0 has saved it.
    if task.task_prefix_token is not None:
        from mammoth.inputters.vocab import HFTokenizerVocab
        if isinstance(tgt_vocab, HFTokenizerVocab):
            new_id = tgt_vocab.add_special_token(task.task_prefix_token)
            if device_rank == 0 and tgt_vocab.path is not None:
                tgt_vocab.tokenizer.save(tgt_vocab.path)
                logger.info(
                    f"Saved updated tokenizer with '{task.task_prefix_token}' (id={new_id}) "
                    f"to {tgt_vocab.path}"
                )
        elif task.task_prefix_token not in tgt_vocab.stoi:
            tgt_vocab.add_token(task.task_prefix_token, is_special=True)
            logger.info(
                f"Added special token '{task.task_prefix_token}' to tgt vocab for task '{task.corpus_id}'"
            )
            if device_rank == 0 and tgt_vocab.path is not None:
                import codecs
                with codecs.open(tgt_vocab.path, 'a', 'utf-8') as f:
                    f.write(f'\n{task.task_prefix_token}')
                logger.info(
                    f"Appended '{task.task_prefix_token}' to vocab file {tgt_vocab.path}"
                )

    # get transform classes to infer special tokens
    # FIXME ensure TQM properly initializes transform with global if necessary
    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}
    corpus_opts = opts.tasks[task.corpus_id]
    transforms_to_apply = corpus_opts.get('transforms', None)
    transforms_to_apply = transforms_to_apply or opts.transforms
    transforms_to_apply = transforms_to_apply or []
    transforms_cls = make_transforms(
        opts,
        get_transforms_cls(transforms_to_apply),
        vocabs,
        task=task,
    )
    transforms_to_apply = [transforms_cls[trf_name] for trf_name in transforms_to_apply]

    max_length = None
    model_max_seq_len = opts.max_length
    if opts.pad_to_max_length:
        assert opts.max_length is not None and opts.max_length > 0, 'Please provide a --max_length'
        max_length = opts.max_length

    # Check if using indexed (pre-tokenized) dataset
    data_type = corpus_opts.get('data_type', 'text')

    if data_type == 'indexed':
        # Use indexed dataset with automatic pretokenization
        src_path = corpus_opts["path_src"] if is_train else corpus_opts["path_valid_src"]
        tgt_path = corpus_opts.get("path_tgt") if is_train else corpus_opts.get("path_valid_tgt")

        # Auto-detect and create indexed files if needed
        src_indexed_path = _check_and_create_indexed(
            opts, task, src_path, is_src=True, device_rank=device_rank
        )
        tgt_indexed_path = None
        if tgt_path:
            tgt_indexed_path = _check_and_create_indexed(
                opts, task, tgt_path, is_src=False, device_rank=device_rank
            )

        # Warn if transforms are specified for indexed datasets
        if transforms_to_apply:
            logger.warning(
                f"⚠️  Transforms specified for indexed dataset '{task.corpus_id}': {[t.__class__.__name__ for t in transforms_to_apply]}\n"
                f"   Indexed datasets contain pre-tokenized token IDs and do NOT support transforms during training.\n"
                f"   Transforms are IGNORED for indexed datasets.\n"
                f"   To apply transforms, preprocess your data with the desired transforms first."
            )

        logger.info(f"Using indexed dataset: {src_indexed_path}")
        dataset = IndexedCorpus(
            src_indexed_path,
            tgt_indexed_path,
            src_vocab,
            tgt_vocab,
            TransformPipe(opts, transforms_to_apply),
            stride=corpus_opts.get('stride', None),
            offset=corpus_opts.get('offset', None),
            is_train=is_train,
            task=task,
            max_length=max_length,
            line_idx_restore=line_idx_restore,
            model_max_seq_len=model_max_seq_len,
            verbose_dataloader=getattr(opts, 'verbose_dataloader', False),
            device_rank=device_rank,
        )
    else:
        # Use traditional text dataset
        dataset = ParallelCorpus(
            corpus_opts["path_src"] if is_train else corpus_opts["path_valid_src"],
            corpus_opts["path_tgt"] if is_train else corpus_opts["path_valid_tgt"],
            src_vocab,
            tgt_vocab,
            TransformPipe(opts, transforms_to_apply),
            stride=corpus_opts.get('stride', None),
            offset=corpus_opts.get('offset', None),
            is_train=is_train,
            task=task,
            max_length=max_length,
            line_idx_restore=line_idx_restore,
            model_max_seq_len=model_max_seq_len,
            verbose_dataloader=getattr(opts, 'verbose_dataloader', False),
            device_rank=device_rank,
        )

    return dataset


def build_sub_vocab(examples):
    """Build vocab counts on (strided) subpart of the data."""
    sub_counter_src = collections.Counter()
    sub_counter_tgt = collections.Counter()
    for i, item in enumerate(examples):
        src, tgt = item['src'], item['tgt']
        sub_counter_src.update(src)
        sub_counter_tgt.update(tgt)
    return sub_counter_src, sub_counter_tgt


def init_pool(queues):
    """Add the queues as attribute of the pooled function."""
    build_sub_vocab.queues = queues
