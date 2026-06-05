"""
Shared config for golden fixture generation and parity tests.
Both generate.py and test_golden.py import from here to ensure
they use identical opts, inputs, and random seed.
"""
import torch

from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS

# Fixed seed for both model init and input generation
SEED = 42

# Model configuration matching the active training config
MODEL_DIM = 512   # must be divisible by HEADS
HEADS = 16
FF_MULT = 4.0
ROTARY_POS_EMB = True
POST_EMB_NORM = True
ATTN_DROPOUT = 0.1
FF_DROPOUT = 0.1
ENC_LAYERS = [2]
DEC_LAYERS = [2]

# Test batch dimensions
BATCH = 3
SRC_LEN = 10
TGT_LEN = 8   # decoder_input will be TGT_LEN-1 tokens

# Minimal bilingual vocab (src lang 'a', tgt lang 'b')
SRC_ITEMS = [f'w{i}' for i in range(40)]
TGT_ITEMS = [f't{i}' for i in range(50)]

VOCABS = {
    ('src', 'a'): Vocab(None, items=SRC_ITEMS, tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('tgt', 'b'): Vocab(None, items=TGT_ITEMS, tag='dummy', specials=list(DEFAULT_SPECIALS)),
}


def make_inputs(device='cpu'):
    """Return (src, decoder_input, src_mask, tgt_labels) with fixed seed."""
    rng = torch.Generator(device=device)
    rng.manual_seed(SEED)

    src_vocab_size = len(VOCABS[('src', 'a')])
    tgt_vocab_size = len(VOCABS[('tgt', 'b')])

    src = torch.randint(0, src_vocab_size, (BATCH, SRC_LEN), generator=rng, device=device)
    # src_mask: True = valid token (non-padding)
    src_mask = torch.ones(BATCH, SRC_LEN, dtype=torch.bool, device=device)
    # decoder input: BOS + TGT_LEN-1 tokens; labels: the TGT_LEN-1 target tokens
    tgt = torch.randint(0, tgt_vocab_size, (BATCH, TGT_LEN), generator=rng, device=device)
    decoder_input = tgt[:, :-1]   # (batch, TGT_LEN-1)
    tgt_labels = tgt[:, 1:]       # (batch, TGT_LEN-1)

    return src, decoder_input, src_mask, tgt_labels
