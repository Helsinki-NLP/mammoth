"""Module defining decoders."""
from mammoth.decoders.decoder import DecoderBase, InputFeedRNNDecoder, StdRNNDecoder
from mammoth.decoders.transformer import TransformerDecoder
from mammoth.decoders.cnn_decoder import CNNDecoder


str2dec = {
    "rnn": StdRNNDecoder,
    "ifrnn": InputFeedRNNDecoder,
    "cnn": CNNDecoder,
    "transformer": TransformerDecoder,
}

__all__ = [
    "DecoderBase",
    "TransformerDecoder",
    "StdRNNDecoder",
    "CNNDecoder",
    "InputFeedRNNDecoder",
    "str2dec",
]
