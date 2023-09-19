"""Module defining encoders."""
from mammoth.encoders.encoder import EncoderBase
from mammoth.encoders.transformer import TransformerEncoder
from mammoth.encoders.ggnn_encoder import GGNNEncoder
from mammoth.encoders.rnn_encoder import RNNEncoder
from mammoth.encoders.cnn_encoder import CNNEncoder
from mammoth.encoders.mean_encoder import MeanEncoder


str2enc = {
    "ggnn": GGNNEncoder,
    "rnn": RNNEncoder,
    "brnn": RNNEncoder,
    "cnn": CNNEncoder,
    "transformer": TransformerEncoder,
    "mean": MeanEncoder,
}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder", "MeanEncoder", "str2enc"]
