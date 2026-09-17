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
        context_dim: Optional[int] = None,
        kv_heads: Optional[int] = None,
        qk_norm: bool = False,
        qk_norm_eps: float = 1e-6,
        attn_scale: Optional[float] = None,
    ):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads if kv_heads is not None else heads
        self.dim_head = dim_head if dim_head is not None else dim // heads
        self.attn_dropout = attn_dropout
        self.causal = causal
        self.is_cross_attn = is_cross_attn
        self.attn_scale = attn_scale

        inner = heads * self.dim_head
        kv_inner = self.kv_heads * self.dim_head
        # For cross-attn, K/V are projected from the encoder's output, which
        # may have a different feature dim than this block's own `dim`
        # (e.g. Gemma3-hosting decoder blocks vs. a smaller fake encoder).
        kv_input_dim = context_dim if context_dim is not None else dim
        self.to_q = nn.Linear(dim, inner, bias=False)
        self.to_k = nn.Linear(kv_input_dim, kv_inner, bias=False)
        self.to_v = nn.Linear(kv_input_dim, kv_inner, bias=False)
        self.to_out = nn.Linear(inner, dim, bias=False)

        self.qk_norm = qk_norm
        if qk_norm:
            self.q_norm = nn.RMSNorm(self.dim_head, eps=qk_norm_eps, elementwise_affine=True)
            self.k_norm = nn.RMSNorm(self.dim_head, eps=qk_norm_eps, elementwise_affine=True)

    def _split_heads(self, x: Tensor, heads: Optional[int] = None) -> Tensor:
        # (batch, seq, heads * dim_head) -> (batch, heads, seq, dim_head)
        b, s, _ = x.shape
        heads = heads if heads is not None else self.heads
        return x.view(b, s, heads, self.dim_head).transpose(1, 2)

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
                k = self._split_heads(self.to_k(src), heads=self.kv_heads)
                v = self._split_heads(self.to_v(src), heads=self.kv_heads)
                if cache is not None:
                    cache.cross_k = k
                    cache.cross_v = v
        else:
            k = self._split_heads(self.to_k(x), heads=self.kv_heads)
            v = self._split_heads(self.to_v(x), heads=self.kv_heads)

            if self.qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            if rotary is not None:
                cos, sin = rotary
                q, k = apply_rotary(q, k, cos, sin)

            if cache is not None:
                if cache.self_k is not None:
                    k = torch.cat([cache.self_k, k], dim=2)
                    v = torch.cat([cache.self_v, v], dim=2)
                cache.self_k = k
                cache.self_v = v

        use_causal = self.causal and cache is None and mask is None
        dropout_p = self.attn_dropout if self.training else 0.0

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
            is_causal=use_causal,
            scale=self.attn_scale,
            enable_gqa=(self.kv_heads != self.heads),
        )

        out = self._merge_heads(out)
        return self.to_out(out), cache
