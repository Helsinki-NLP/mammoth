import collections
import itertools
import warnings

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset

from onmt.constants import DefaultTokens
from onmt.transforms import TransformPipe, get_transforms_cls, make_transforms_with_vocabs
from onmt.inputters_mvp.vocab import get_vocab
from onmt.utils.logging import logger


# for compliance with previous code
Batch = collections.namedtuple('Batch', 'src tgt')


class ParallelCorpus(IterableDataset):
    """Torch-style dataset """
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, transforms, batch_size, batch_type, device='cpu'):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.transforms = transforms
        self.device = device
        self.vocabs = {
            'src': src_vocab,
            'tgt': tgt_vocab,
        }
        self.batch_size = batch_size
        self.batch_type = batch_type

    #FIXME: most likely redundant with onmt.transforms.tokenize
    def _tokenize(self, string, side='src'):
        """Convert string into list of indices"""
        vocab = self.vocabs[side]
        unk = DefaultTokens.UNK
        tokens = [
            DefaultTokens.BOS,
            *(
                word if word in vocab.stoi else DefaultTokens.UNK
                for word in string.split()
            ),
            DefaultTokens.EOS,
        ]
        return tokens

    def _numericalize(self, tokens, side='src'):
        vocab = self.vocabs[side]
        indices = torch.tensor(list(map(vocab.__getitem__, tokens)), device=self.device)
        return indices


    def to(self, device):
        self.device = device
        return self

    def __iter__(self):
        """Read file, produce batches of examples"""
        def _make_example_dict(packed):
            src_str, tgt_str = packed
            return {
                'src': self._tokenize(src_str),
                'tgt': self._tokenize(tgt_str),
                # 'align': None,
            }
        def _cast(example_dict):
            return {
                k: self._numericalize(v, side=k)
                for k, v in example_dict.items()
                # if v is not None
            }
        with open(self.src_file) as src_fh, open(self.tgt_file) as tgt_fh:
            examples = zip(src_fh, tgt_fh)
            examples = map(_make_example_dict, examples)
            examples = map(self.transforms.apply, examples)
            examples = filter(None, examples)  # filtertoolong replaces invalid examples with None
            examples = map(_cast, examples)
            accum, cur_batch_size = [], 0
            for example in examples:
                length = 1 if self.batch_type == 'sents' else (len(example['src']) + len(example['tgt']))
                if length > self.batch_size:
                    warnings.warn(f'Example {example} larger than requested batch size, dropping it.')
                else:
                    if cur_batch_size + length > self.batch_size:
                        yield self.collate_fn(accum)
                        accum, cur_batch_size = [], 0
                    cur_batch_size += length
                    # FIXME: iterable-style dataset quirk: only considers the number of calls to __next__.
                    # For this impl to work, we need to guarantee that the iteration in the sampler is the same as in the
                    # dataset; so this is a fairly brittle solution for now.
                    accum.append(example)
            if accum:
                yield self.collate_fn(accum)


    # FIXME: some RNN archs require sorting src's by length
    def collate_fn(self, examples):
        src_pad_idx = self.vocabs['src'][DefaultTokens.PAD]
        tgt_pad_idx = self.vocabs['tgt'][DefaultTokens.PAD]
        src_lengths = torch.tensor([ex['src'].numel() for ex in examples], device=self.device)
        src = (pad_sequence([ex['src'] for ex in examples], padding_value=src_pad_idx), src_lengths)
        tgt = pad_sequence([ex['src'] for ex in examples], padding_value=src_pad_idx)
        batch = Batch(src, tgt)
        import pdb; pdb.set_trace()
        return batch


def get_corpus(opts, corpus_id: str, src_lang: str, tgt_lang: str, is_train: bool = False):
    """build an iterable Dataset object"""

    # 1. get transform classes to infer special tokens
    transforms_cls = get_transforms_cls(opts.data[corpus_id].get('transforms', opts.transforms))

    # 2. build src/tgt vocabs
    if transforms_cls:
        logger.info(f'Transforms: {transforms_cls}')
        src_specials, tgt_specials = zip(*(cls.get_specials() for cls in transforms_cls))
        src_specials = sorted(itertools.chain.from_iterable(src_specials))
        tgt_specials = sorted(itertools.chain.from_iterable(tgt_specials))
    else:
        logger.info('No transforms found')
        src_specials = tgt_specials = [DefaultTokens.BOS, DefaultTokens.EOS, DefaultTokens.UNK, DefaultTokens.PAD, DefaultTokens.MASK]
    src_vocab_size = opts.src_vocab_size or None
    tgt_vocab_size = opts.tgt_vocab_size or None
    src_vocab = get_vocab(opts.src_vocab[src_lang], src_lang, src_vocab_size)
    tgt_vocab = get_vocab(opts.tgt_vocab[tgt_lang], tgt_lang, tgt_vocab_size)
    if opts.share_vocab:
        assert src_vocab_size == tgt_vocab_size
        src_vocab = tgt_vocab = Vocab.merge(src_vocab, tgt_vocab, size=src_vocab_size)
    vocabs = {'src': src_vocab, 'tgt': tgt_vocab}

    # 3. build Dataset proper
    dataset = ParallelCorpus(
        opts.data[corpus_id]["path_src"],
        opts.data[corpus_id]["path_tgt"],
        src_vocab,
        tgt_vocab,
        TransformPipe(opts, make_transforms_with_vocabs(opts, transforms_cls, vocabs)),
        opts.batch_size,
        opts.batch_type,
    )
    return dataset
