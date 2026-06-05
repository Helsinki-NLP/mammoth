from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .rotary import apply_rotary
from .cache import LayerCache


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: Optional[int] = None,
        attn_dropout: float = 0.0,
        causal: bool = False,
        is_cross_attn: bool = False,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head if dim_head is not None else dim // heads
        self.attn_dropout = attn_dropout
        self.causal = causal
        self.is_cross_attn = is_cross_attn

        inner = heads * self.dim_head
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(dim, inner, bias=False)
        self.to_v = nn.Linear(dim, inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

    def _split_heads(self, x: Tensor) -> Tensor:
        # (batch, seq, heads * dim_head) -> (batch, heads, seq, dim_head)
        b, s, _ = x.shape
        return x.view(b, s, self.heads, self.dim_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        # (batch, heads, seq, dim_head) -> (batch, seq, heads * dim_head)
        b, h, s, d = x.shape
        return x.transpose(1, 2).contiguous().view(b, s, h * d)

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        rotary: Optional[tuple[Tensor, Tensor]] = None,
        cache: Optional[LayerCache] = None,
    ) -> tuple[Tensor, Optional[LayerCache]]:
        q = self._split_heads(self.to_q(x))

        if self.is_cross_attn:
            # Compute or reuse cross-attention K, V from encoder output
            if cache is not None and cache.cross_k is not None:
                k, v = cache.cross_k, cache.cross_v
            else:
                src = context if context is not None else x
                k = self._split_heads(self.to_k(src))
                v = self._split_heads(self.to_v(src))
                if cache is not None:
                    cache.cross_k = k
                    cache.cross_v = v
        else:
            k = self._split_heads(self.to_k(x))
            v = self._split_heads(self.to_v(x))

            if rotary is not None:
                cos, sin = rotary
                q, k = apply_rotary(q, k, cos, sin)

            if cache is not None:
                if cache.self_k is not None:
                    k = torch.cat([cache.self_k, k], dim=2)
                    v = torch.cat([cache.self_v, v], dim=2)
                cache.self_k = k
                cache.self_v = v

        use_causal = self.causal and cache is None
        dropout_p = self.attn_dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
        )

        out = self._merge_heads(out)
        return self.to_out(out), cache
