from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .cache import LayerCache
from .ffn import FeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        norm_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(dim, heads, attn_dropout=attn_dropout)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, ff_mult, ff_dropout)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        rotary: Optional[tuple[Tensor, Tensor]] = None,
        cache: Optional[LayerCache] = None,
    ) -> Tensor:
        h, cache = self.self_attn(self.norm1(x), mask=mask, rotary=rotary, cache=cache)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        norm_bias: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(dim, heads, attn_dropout=attn_dropout, causal=True)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True)
        self.cross_attn = MultiHeadAttention(dim, heads, attn_dropout=attn_dropout, is_cross_attn=True)
        self.norm3 = nn.RMSNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, ff_mult, ff_dropout)

    def forward(
        self,
        x: Tensor,
        context: Tensor,
        context_mask: Optional[Tensor] = None,
        rotary: Optional[tuple[Tensor, Tensor]] = None,
        cache: Optional[LayerCache] = None,
    ) -> Tensor:
        h, cache = self.self_attn(self.norm1(x), rotary=rotary, cache=cache)
        x = x + h
        h, _ = self.cross_attn(self.norm2(x), context=context, mask=context_mask, cache=cache)
        x = x + h
        x = x + self.ff(self.norm3(x))
        return x
