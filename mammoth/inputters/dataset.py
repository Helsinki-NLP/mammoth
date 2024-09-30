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


TensorWithMask = collections.namedtuple('TensorWithMask', ['tensor', 'mask'])


@dataclass
class Batch():
    src: TensorWithMask
    tgt: TensorWithMask
    labels: Tensor
    batch_size: int
    line_idx: int

    def to(self, device):
        self.src = TensorWithMask(self.src.tensor.to(device), self.src.mask.to(device))
        if self.tgt is not None:
            self.tgt = TensorWithMask(self.tgt.tensor.to(device), self.tgt.mask.to(device))
        if self.labels is not None:
            self.labels = self.labels.to(device)
        return self


def read_examples_from_files(
    src_path,
    tgt_path,
    tokenize_fn=str.split,
    transforms_fn=lambda x: x,
    stride=None,
    offset=None,
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
        return {
            'src': tokenize_fn(src_str, side='src'),
            'tgt': tokenize_fn(tgt_str, side='tgt') if tgt_str is not None else None,
            # 'align': None,
            'line_idx': next(line_idx_generator)
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
        self.max_length = max_length
        self._line_idx_restore = line_idx_restore

    # FIXME: most likely redundant with mammoth.transforms.tokenize
    def _tokenize(self, string, side='src'):
        """Split string, accompanied by a drumroll"""
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
        indices = torch.tensor([
            bos,
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
            logger.info(f'restoring {self.corpus_id} to line: {self._line_idx_restore}')
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


def get_corpus(
    opts,
    task,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    is_train: bool = False,
    line_idx_restore: int = None,
):
    """build an iterable Dataset object"""
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
    if opts.pad_to_max_length:
        assert opts.max_length is not None and opts.max_length > 0, 'Please provide a --max_length'
        max_length = opts.max_length
    # build Dataset proper
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
