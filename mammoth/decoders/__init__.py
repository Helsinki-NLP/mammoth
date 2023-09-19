"""Module defining decoders."""
from mammoth.decoders.decoder import DecoderBase
from mammoth.decoders.transformer_decoder import TransformerDecoder


str2dec = {
    "transformer": TransformerDecoder,
}

__all__ = [
    "DecoderBase",
    "TransformerDecoder",
    "str2dec",
]
