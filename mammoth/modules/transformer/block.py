from typing import Optional

import torch.nn as nn
from torch import Tensor

from .attention import MultiHeadAttention
from .cache import LayerCache
from .ffn import FeedForward
from .masking import build_sliding_window_causal_mask
from .rotary import RotaryEmbedding


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: Optional[int] = None,
        ff_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        norm_bias: bool = False,
        norm_eps: Optional[float] = None,
        activation: str = "swiglu",
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(dim, heads, dim_head=dim_head, attn_dropout=attn_dropout)
        self.norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
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
    """
    Pre-norm decoder block: self-attn -> cross-attn -> ff, each with a
    residual connection.

    Optional Gemma3-style extensions (see CLAUDE.md "Gemma3-270M -> Mammoth
    Conversion" for the architecture this mirrors, borrowed directly from
    `transformers.models.gemma3.modeling_gemma3.Gemma3DecoderLayer`):

    - `sandwich_norm`: adds a post-branch RMSNorm applied to the self-attn
      and ff outputs *before* the residual add (Gemma3's
      post_attention_layernorm / post_feedforward_layernorm). Cross-attn is
      Mammoth-only scaffolding (Gemma3 has no cross-attention) and is never
      sandwich-normed.
    - `sliding_window`: builds a windowed-causal mask internally (causal AND
      `query_pos - key_pos < window`) and feeds it to self_attn, accounting
      for any cached prefix length.
    - `rotary_emb`: gives the block its own RotaryEmbedding (used for
      Gemma3's dual RoPE, where local/sliding layers use a different theta
      than global/full-attention layers) instead of relying on the rotary
      tuple computed once per stack and shared across all blocks.
    - `kv_heads` / `qk_norm` / `attn_scale`: forwarded to the self-attn
      MultiHeadAttention (see attention.py).
    - `context_dim`: lets cross-attn's K/V project from a different feature
      size than this block's own `dim` (e.g. encoder dim != decoder dim),
      instead of requiring them to match.
    - `use_cross_attn`: when False, cross_attn/norm2 are never built at all
      (not just zero-initialized) and forward() never touches `context` --
      a TRUE decoder-only block, architecturally identical to a plain LM
      decoder layer (e.g. Gemma3DecoderLayer, which has no cross-attention).
      This is the "option 1" path from CLAUDE.md "Gemma3-270M -> Mammoth
      Conversion", used by convert_gemma3_decoder_only.py, as opposed to the
      "fake encoder" path (convert_gemma3_native.py) which keeps
      use_cross_attn=True (the default) and zeros cross_attn.to_out instead.
    - `norm_eps`: RMSNorm epsilon for all norms in this block (including
      self-attn's qk_norm). Gemma3 uses a fixed 1e-6 everywhere; the default
      `None` keeps torch's dtype-based default for non-Gemma3 models. This
      matters more than it looks: activation rows can have near-zero
      variance, and torch's default eps (~1.19e-7 for fp32) vs Gemma3's 1e-6
      differ enough to blow up the normalized output on those rows once
      multiplied by a large-magnitude norm weight.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: Optional[int] = None,
        ff_mult: float = 4.0,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        norm_bias: bool = False,
        norm_eps: Optional[float] = None,
        activation: str = "swiglu",
        kv_heads: Optional[int] = None,
        qk_norm: bool = False,
        attn_scale: Optional[float] = None,
        sandwich_norm: bool = False,
        sliding_window: Optional[int] = None,
        rotary_emb: Optional[RotaryEmbedding] = None,
        context_dim: Optional[int] = None,
        use_cross_attn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.self_attn = MultiHeadAttention(
            dim, heads, dim_head=dim_head, attn_dropout=attn_dropout, causal=True,
            kv_heads=kv_heads, qk_norm=qk_norm, qk_norm_eps=norm_eps if norm_eps is not None else 1e-6,
            attn_scale=attn_scale,
        )
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.norm2 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
            self.cross_attn = MultiHeadAttention(
                dim, heads, dim_head=dim_head, attn_dropout=attn_dropout, is_cross_attn=True,
                context_dim=context_dim,
            )
        else:
            self.norm2 = None
            self.cross_attn = None
        self.norm3 = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
        self.ff = FeedForward(dim, ff_mult, ff_dropout, activation)

        self.sandwich_norm = sandwich_norm
        if sandwich_norm:
            self.norm1_post = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)
            self.norm3_post = nn.RMSNorm(dim, eps=norm_eps, elementwise_affine=True)

        self.sliding_window = sliding_window
        self.rotary_emb = rotary_emb

    def forward(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary: Optional[tuple[Tensor, Tensor]] = None,
        cache: Optional[LayerCache] = None,
    ) -> Tensor:
        if self.rotary_emb is not None:
            offset = 0
            if cache is not None and cache.self_k is not None:
                offset = cache.self_k.size(2)
            rotary = self.rotary_emb(x.size(1), x.device, offset=offset)

        self_attn_mask = None
        if self.sliding_window is not None:
            q_len = x.size(1)
            kv_len = q_len
            if cache is not None and cache.self_k is not None:
                kv_len += cache.self_k.size(2)
            full_mask = build_sliding_window_causal_mask(kv_len, self.sliding_window, x.device)
            self_attn_mask = full_mask[-q_len:, :][None, None, :, :]

        h, cache = self.self_attn(self.norm1(x), mask=self_attn_mask, rotary=rotary, cache=cache)
        if self.sandwich_norm:
            h = self.norm1_post(h)
        x = x + h

        if self.use_cross_attn:
            h, _ = self.cross_attn(self.norm2(x), context=context, mask=context_mask, cache=cache)
            x = x + h

        h = self.ff(self.norm3(x))
        if self.sandwich_norm:
            h = self.norm3_post(h)
        x = x + h
        return x
