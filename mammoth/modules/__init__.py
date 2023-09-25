"""  Attention and normalization modules  """
from mammoth.modules.util_class import Elementwise
from mammoth.modules.gate import context_gate_factory, ContextGate
from mammoth.modules.global_attention import GlobalAttention
from mammoth.modules.conv_multi_step_attention import ConvMultiStepAttention
from mammoth.modules.copy_generator import (
    CopyGenerator,
    CopyGeneratorLoss,
    CopyGeneratorLossCompute,
    CopyGeneratorLMLossCompute,
)
from mammoth.modules.multi_headed_attn import MultiHeadedAttention
from mammoth.modules.embeddings import Embeddings, PositionalEncoding
from mammoth.modules.weight_norm import WeightNormConv2d
from mammoth.modules.average_attn import AverageAttention
from mammoth.modules.stable_embeddings import StableEmbedding
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
    "context_gate_factory",
    "ContextGate",
    "GlobalAttention",
    "ConvMultiStepAttention",
    "CopyGenerator",
    "CopyGeneratorLoss",
    "CopyGeneratorLossCompute",
    "MultiHeadedAttention",
    "Embeddings",
    "PositionalEncoding",
    "WeightNormConv2d",
    "AverageAttention",
    "CopyGeneratorLMLossCompute",
    "StableEmbedding",
    "AttentionBridge",
]
