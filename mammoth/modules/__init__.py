"""Components libary"""
from mammoth.modules.util_class import Elementwise
from mammoth.modules.multi_headed_attn import MultiHeadedAttention
from mammoth.modules.embeddings import Embeddings, PositionalEncoding
# from mammoth.modules.weight_norm import WeightNormConv2d
from mammoth.modules.average_attn import AverageAttention
from mammoth.modules.attention_bridge import AttentionBridge

from mammoth.modules.encoder import EncoderBase
from mammoth.modules.transformer_encoder import TransformerEncoder
from mammoth.modules.mean_encoder import MeanEncoder

from mammoth.modules.decoder import DecoderBase
from mammoth.modules.transformer_decoder import TransformerDecoder


str2enc = {
    "transformer": TransformerEncoder,
    "mean": MeanEncoder,
}

str2dec = {
    "transformer": TransformerDecoder,
}


__all__ = [
    "DecoderBase",
    "TransformerDecoder",
    "str2dec",
    "EncoderBase",
    "TransformerEncoder",
    "MeanEncoder",
    "str2enc",
    "Elementwise",
    "MultiHeadedAttention",
    "Embeddings",
    "PositionalEncoding",
    "AverageAttention",
    "AttentionBridge",
]
