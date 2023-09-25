"""The point of this package is to provide:
    - vocab loading (cf. vocab.py)
    - token-counts based batch sampler (cf. dataloader.py)
    - on the fly pad, bos, eos, unk handling (cf. dataset.py)
    - handling of transforms (cf. dataset.py)
    - multiple parallel corpora, in accordance with TaskDistributor (cf. distributed.py)
"""

from mammoth.inputters.dataloader import build_dataloader, DynamicDatasetIter
from mammoth.inputters.dataset import get_corpus, build_vocab_counts, ParallelCorpus
from mammoth.inputters.vocab import get_vocab, DEFAULT_SPECIALS


__all__ = [
    'build_dataloader',
    'DynamicDatasetIter',
    'get_corpus',
    'get_vocab',
    'DEFAULT_SPECIALS',
    'build_vocab_counts',
    'ParallelCorpus'
]
