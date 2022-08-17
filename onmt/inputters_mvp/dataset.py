import torch
from torch.utils.data import IterableDataset

from onmt.constants import DefaultTokens
from onmt.transforms import TransformPipe, get_transforms_cls, make_transforms_with_vocabs
from onmt.inputters_mvp.vocab import get_vocab
from onmt.utils.logging import logger

class ParallelCorpus(IterableDataset):
    """Torch-style dataset """
    def __init__(self, src_file, tgt_file, src_vocab, tgt_vocab, transforms, device='cpu'):
        self.src_file = src_file
        self.tgt_file = tgt_file
        self.transforms = transforms
        self.device = device
        self.vocabs = {
            'src': src_vocab,
            'tgt': tgt_vocab,
        }

    #FIXME: most likely redundant with onmt.transforms.tokenize
    def _tokenize(self, string, side='src'):
        """Convert string into list of indices"""
        vocab = self.vocabs[side]
        unk = DefaultTokens.UNK
        tokens = [
            DefaultTokens.BOS,
            *(
                word if word in vocab else DefaultTokens.UNK
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
        """Read file, produce examples"""
        def _make_example_dict(src_str, tgt_str):
            return {
                'src': self._tokenize(src_str),
                'tgt': self._tokenize(tgt_str),
                # 'align': None,
            }
        def _cast(example_dict):
            return {
                k: self._numericalize(v, side=k)
                for k, v in example_dict
                # if v is not None
            }
        with open(self.src_file) as src_fh, open(self.tgt_file) as tgt_fh:
            examples = zip(src_fh, tgt_fh)
            examples = map(_make_example_dict, examples)
            examples = map(self.transforms, examples)
            examples = filter(None, examples)  # filtertoolong replaces invalid examples with None
            examples = map(_cast, examples)
            yield from examples


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
    )
    return dataset
