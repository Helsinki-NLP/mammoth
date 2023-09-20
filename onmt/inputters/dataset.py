import collections
from dataclasses import dataclass
import itertools
from functools import partial
import gzip

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from onmt.constants import DefaultTokens
from onmt.transforms import TransformPipe, get_transforms_cls, make_transforms
from onmt.utils.logging import logger
from onmt.inputters.vocab import Vocab


@dataclass
class Batch():
    src: tuple  # of torch Tensors
    tgt: torch.Tensor
    batch_size: int
    idx:torch.Tensor

    def to(self, device):
        self.src = (self.src[0].to(device), self.src[1].to(device))
        if self.tgt is not None:
            self.tgt = self.tgt.to(device)
        self.idx = self.idx
        return self

def read_examples_from_files(
    src_path,
    tgt_path,
    tokenize_fn=str.split,
    transforms_fn=lambda x: x,
    stride=None,
    offset=None,
    start_index=0,
):
    """Helper function to read examples"""
    idaux =  itertools.count(start_index)
    def _make_example_dict(packed):
        """Helper function to convert lines to dicts"""
        src_str, tgt_str = packed
        return {
            'src': tokenize_fn(src_str, side='src'),
            'tgt': tokenize_fn(tgt_str, side='tgt') if tgt_str is not None else None,
            'idx': next(idaux)
            # 'align': None,
        }

    if src_path.endswith('.gz'):
        src_fh = gzip.open(src_path, 'rt')
    else:
        src_fh = open(src_path, 'rt')
    if tgt_path is None:
        tgt_fh = itertools.repeat(None)
    elif tgt_path.endswith('.gz'):
        tgt_fh = gzip.open(tgt_path, 'rt')
    else:
        tgt_fh = open(tgt_path, 'rt')

    examples = zip(src_fh, tgt_fh)
    if start_index > 0:
        # ignore 1st start_index examples when we restart training
        examples = itertools.islice(examples, start_index + 1, None)

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
        current_file_index=None,
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
        self.current_file_index = current_file_index 

    # FIXME: most likely redundant with onmt.transforms.tokenize
    def _tokenize(self, string, side='src'):
        """Split string, accompanied by a drumroll"""
        return string.split()

    def _numericalize(self, tokens, side='src'):
        """Convert list of strings into list of indices"""
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

    def to(self, device):
        self.device = device
        return self

    def __iter__(self):
        """Read file, produce batches of examples"""
        start_index = self.current_file_index
        self.current_file_index = None
        def _cast(example_dict):
            return {
                k: self._numericalize(v, side=k) if k != 'idx' else v
                for k, v in example_dict.items()
                if v is not None
            }

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
            offset=self.offset,
            start_index=start_index,
        )
        examples = map(_cast, examples)
        yield from examples

    # FIXME: some RNN archs require sorting src's by length
    def collate_fn(self, examples):
        has_tgt = 'tgt' in examples[0].keys()
        src_padidx = self.vocabs['src'][DefaultTokens.PAD]
        tgt_padidx = self.vocabs['tgt'][DefaultTokens.PAD]
        src_lengths = torch.tensor([ex['src'].numel() for ex in examples], device='cpu')
        src = (pad_sequence([ex['src'] for ex in examples], padding_value=src_padidx).unsqueeze(-1), src_lengths)
        tgt = pad_sequence([ex['tgt'] for ex in examples], padding_value=tgt_padidx).unsqueeze(-1) if has_tgt else None
        idx = torch.tensor([ex['idx'] for ex in examples]).max()
        batch = Batch(src, tgt, len(examples), idx)
        return batch


def get_corpus(opts, task, src_vocab: Vocab, tgt_vocab: Vocab, is_train: bool = False, current_file_index=None):
    """build an iterable Dataset object"""
    # get transform classes to infer special tokens
    # FIXME ensure TQM properly initializes transform with global if necessary
    corpus_opts = opts.data[task.corpus_id]
    transforms_cls = get_transforms_cls(corpus_opts.get('transforms', opts.transforms))

    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}
    # build Dataset proper
    dataset = ParallelCorpus(
        corpus_opts["path_src"] if is_train else corpus_opts["path_valid_src"],
        corpus_opts["path_tgt"] if is_train else corpus_opts["path_valid_tgt"],
        src_vocab,
        tgt_vocab,
        TransformPipe(opts, make_transforms(opts, transforms_cls, vocabs, task=task).values()),
        stride=corpus_opts.get('stride', None),
        offset=corpus_opts.get('offset', None),
        is_train=is_train,
        task=task,
        current_file_index=current_file_index,
    )
    return dataset


def build_sub_vocab(examples, n_sample):
    """Build vocab counts on (strided) subpart of the data."""
    sub_counter_src = collections.Counter()
    sub_counter_tgt = collections.Counter()
    for i, item in enumerate(examples):
        src, tgt = item['src'], item['tgt']
        # for feat_name, feat_line in maybe_example["src"].items():
        #     if feat_name not in ["src", "src_original"]:
        #         sub_counter_src_feats[feat_name].update(feat_line.split(' '))
        #         if opts.dump_samples:
        #             src_line_pretty = append_features_to_example(src_line_pretty, feat_line)
        sub_counter_src.update(src)
        sub_counter_tgt.update(tgt)
        if n_sample > 0 and (i + 1) >= n_sample:
            break
    return sub_counter_src, sub_counter_tgt


def init_pool(queues):
    """Add the queues as attribute of the pooled function."""
    build_sub_vocab.queues = queues


def build_vocab_counts(opts, corpus_id, transforms, n_sample=3):
    """Build vocabulary counts from data."""

    from functools import partial
    import multiprocessing as mp

    if n_sample == -1:
        logger.info(f"n_sample={n_sample}: Build vocab on full datasets.")
    elif n_sample > 0:
        logger.info(f"Build vocab on {n_sample} transformed examples/corpus.")
    else:
        raise ValueError(f"n_sample should > 0 or == -1, get {n_sample}.")

    # FIXME
    assert not opts.dump_samples, 'Not implemented'

    corpora = {
        corpus_id: read_examples_from_files(
                opts.data[corpus_id]["path_src"],
                opts.data[corpus_id]["path_tgt"],
                # FIXME this is likely not working
                transforms_fn=TransformPipe(transforms).apply if transforms else lambda x: x,
            )
        }
    counter_src = collections.Counter()
    counter_tgt = collections.Counter()

    queues = {
        c_name: [mp.Queue(opts.vocab_sample_queue_size) for i in range(opts.num_threads)] for c_name in corpora.keys()
    }
    # sample_path = os.path.join(os.path.dirname(opts.save_data), CorpusName.SAMPLE)
    with mp.Pool(opts.num_threads, init_pool, [queues]) as p:
        func = partial(build_sub_vocab, corpora[corpus_id], n_sample, opts.num_threads)
        for sub_counter_src, sub_counter_tgt in p.imap(func, range(0, opts.num_threads)):
            counter_src.update(sub_counter_src)
            counter_tgt.update(sub_counter_tgt)

    return counter_src, counter_tgt
