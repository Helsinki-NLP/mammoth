"""ff_mult resolution is shared by the model builder and the FLOP counter."""
from argparse import Namespace

import pytest

from mammoth.distributed.components import Side
from mammoth.model_builder import resolve_ff_mult
from mammoth.utils.flops import compute_transformer_flops


@pytest.mark.parametrize('opts, side, expected', [
    # unset -> activation-based default
    (Namespace(), Side.encoder, (2.67, 'swiglu', False)),
    (Namespace(ff_activation='gelu'), Side.decoder, (4.0, 'gelu', False)),
    (Namespace(ff_activation='geglu'), Side.decoder, (2.67, 'geglu', False)),
    # shared value applies to both sides
    (Namespace(ff_mult=3.0), Side.decoder, (3.0, 'swiglu', True)),
    # side-prefixed values override the shared ones
    (Namespace(ff_mult=3.0, enc_ff_mult=2.0), Side.encoder, (2.0, 'swiglu', True)),
    (Namespace(ff_mult=3.0, enc_ff_mult=2.0), Side.decoder, (3.0, 'swiglu', True)),
    (Namespace(ff_activation='swiglu', dec_ff_activation='gelu'), Side.decoder, (4.0, 'gelu', False)),
    (Namespace(ff_activation='swiglu', dec_ff_activation='gelu'), Side.encoder, (2.67, 'swiglu', False)),
    # explicit None side opt falls through to the shared one
    (Namespace(ff_mult=3.0, dec_ff_mult=None), Side.decoder, (3.0, 'swiglu', True)),
])
def test_resolve_ff_mult(opts, side, expected):
    assert resolve_ff_mult(opts, side) == expected


def test_flops_uses_per_side_ff_mult():
    common = dict(n_src_tokens=10, n_tgt_tokens=7, src_seq_len=10, tgt_seq_len=7,
                  model_dim=8, enc_layers=2, dec_layers=3, vocab_size=0)
    base = compute_transformer_flops(**common, enc_ff_mult=2.0, dec_ff_mult=2.0)
    bigger_enc = compute_transformer_flops(**common, enc_ff_mult=3.0, dec_ff_mult=2.0)
    bigger_dec = compute_transformer_flops(**common, enc_ff_mult=2.0, dec_ff_mult=3.0)
    # MLP term: 4 * dim * (dim * ff_mult) per token per layer, then x6
    per_unit_mult = 6 * 4 * 8 * 8
    assert bigger_enc - base == 10 * 2 * per_unit_mult
    assert bigger_dec - base == 7 * 3 * per_unit_mult
