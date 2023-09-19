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

__all__ = [
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
]
