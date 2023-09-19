"""Module defining encoders."""
from mammoth.encoders.encoder import EncoderBase
from mammoth.encoders.transformer_encoder import TransformerEncoder
from mammoth.encoders.mean_encoder import MeanEncoder


str2enc = {
    "transformer": TransformerEncoder,
    "mean": MeanEncoder,
}

__all__ = ["EncoderBase", "TransformerEncoder", "MeanEncoder", "str2enc"]
