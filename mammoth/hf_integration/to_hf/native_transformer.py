"""Vendored Mammoth native transformer modules — no mammoth package required.

This file is a self-contained copy of mammoth/modules/transformer/*.py with all
relative imports inlined. It is copied into HF artifacts by convert_mammoth_to_hf.py
so that modeling_mammoth.py can load without the mammoth package installed.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

@dataclass
class LayerCache:
    self_k:  Optional[Tensor] = None
    self_v:  Optional[Tensor] = None
    cross_k: Optional[Tensor] = None
    cross_v: Optional[Tensor] = None


class KVCache:
    def __init__(self, num_layers: int):
        self.layers: list[LayerCache] = [LayerCache() for _ in range(num_layers)]

    def reorder_beams(self, beam_indices: Tensor) -> None:
        for layer in self.layers:
            if layer.self_k is not None:
                layer.self_k = layer.self_k[beam_indices]
                layer.self_v = layer.self_v[beam_indices]
            if layer.cross_k is not None:
                layer.cross_k = layer.cross_k[beam_indices]
                layer.cross_v = layer.cross_v[beam_indices]


# ---------------------------------------------------------------------------
# rotary
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def __init__(self, dim_head: int, base: int = 10000):
        super().__init__()
        self.dim_head = dim_head
        self.base = base
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim_head, 2, device=device).float() / self.dim_head)
        )
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
        self._seq_len_cached = seq_len

    def forward(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        if seq_len > self._seq_len_cached:
            self._build_cache(seq_len, device)
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ---------------------------------------------------------------------------
# ffn
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: float = 2.67,
        ff_dropout: float = 0.0,
        activation: Literal["swiglu", "gelu"] = "swiglu",
    ):
        super().__init__()
        inner = int(dim * ff_mult)
        self.activation = activation
        if activation == "swiglu":
            self.w1 = nn.Linear(dim, inner, bias=False)
            self.w2 = nn.Linear(inner, dim, bias=False)
            self.w3 = nn.Linear(dim, inner, bias=False)
        else:
            self.w1 = nn.Linear(dim, inner, bias=False)
            self.w2 = nn.Linear(inner, dim, bias=False)
            self.w3 = None
        self.dropout = nn.Dropout(ff_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        return self.dropout(self.w2(F.gelu(self.w1(x))))


# ---------------------------------------------------------------------------
# attention
# ---------------------------------------------------------------------------

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
        b, s, _ = x.shape
        return x.view(b, s, self.heads, self.dim_head).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
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
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=use_causal)
        out = self._merge_heads(out)
        return self.to_out(out), cache


# ---------------------------------------------------------------------------
# block
# ---------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        ff_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        norm_bias: bool = False,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(dim, heads, attn_dropout=attn_dropout)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, ff_mult, ff_dropout, activation)

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
        activation: str = "swiglu",
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(dim, heads, attn_dropout=attn_dropout, causal=True)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=True)
        self.cross_attn = MultiHeadAttention(dim, heads, attn_dropout=attn_dropout, is_cross_attn=True)
        self.norm3 = nn.RMSNorm(dim, elementwise_affine=True)
        self.ff = FeedForward(dim, ff_mult, ff_dropout, activation)

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


# ---------------------------------------------------------------------------
# stack
# ---------------------------------------------------------------------------

class TransformerStack(nn.Module):
    def __init__(
        self,
        blocks: nn.ModuleList,
        final_norm: Optional[nn.Module],
        dim: int,
        layer_stack_index: int,
        xcoder_id: str,
    ):
        super().__init__()
        self.blocks = blocks
        self.final_norm = final_norm
        self.dim = dim
        self.layer_stack_index = layer_stack_index
        self.xcoder_id = xcoder_id

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary: Optional[tuple[Tensor, Tensor]] = None,
        cache: Optional[KVCache] = None,
    ) -> tuple[Tensor, Optional[KVCache]]:
        for i, block in enumerate(self.blocks):
            layer_cache = cache.layers[i] if cache is not None else None
            if isinstance(block, DecoderBlock):
                x = block(x, context=context, context_mask=context_mask,
                          rotary=rotary, cache=layer_cache)
            else:
                x = block(x, mask=mask, rotary=rotary, cache=layer_cache)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x, cache

    @property
    def depth(self) -> int:
        return len(self.blocks)

    def get_sub_modules(self) -> dict:
        return dict(self._modules)
