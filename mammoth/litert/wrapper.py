"""
LiteRT-compatible wrapper modules for Mammoth encoder-decoder.

Three signatures for a .tflite multi-signature model:
  encode  →  MammothLiteRTEncoder
  prefill →  MammothLiteRTPrefill   (decoder first pass, computes all KV)
  decode  →  MammothLiteRTDecode    (subsequent steps, reads cross-KV, updates self-KV)

Design constraints (torch.export / LiteRT):
  - No Python control flow on tensor values
  - No in-place mutations; KV cache is returned as new tensors
  - Static shapes: enc_max_len and dec_max_len fixed at construction time
  - Rotary cache pre-built at __init__ to avoid dynamic cache growth in forward
  - KV tensors passed as flat positional args (no dataclasses / dicts)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import (
    MammothForConditionalGeneration,
    MammothEncoder,
    MammothDecoder,
)
from mammoth.modules.transformer.rotary import apply_rotary


# ──────────────────────────────────────────────────────────────────────────────
# HLFB-wrapped RMSNorm for GPU / NPU delegate fusion
# ──────────────────────────────────────────────────────────────────────────────

class _HLFBRMSNorm(nn.Module):
    """
    Drop-in replacement for nn.RMSNorm that wraps the computation in a
    StableHLO composite boundary (odml.rms_norm).  This lets the LiteRT
    converter emit a fused custom op that GPU and Qualcomm NPU delegates
    can accelerate.

    Only used when --patch-norms is passed to convert.py / convert_multi.py.
    Must be applied *before* torch.export so the composite appears in the FX
    graph.
    """

    def __init__(self, eps: float | None, weight: Tensor):
        super().__init__()
        self._eps = eps if eps is not None else 1e-6
        self._weight = weight
        self._attr = {"epsilon": float(self._eps)}

    def forward(self, x: Tensor) -> Tensor:
        from litert_torch.backend.composite import StableHLOCompositeBuilder
        composite = StableHLOCompositeBuilder("odml.rms_norm", self._attr)
        x, w = composite.mark_inputs(x, self._weight)
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self._eps) * w
        y = composite.mark_outputs(y)
        return y


def patch_rms_norms(module: nn.Module) -> None:
    """
    Recursively replace every nn.RMSNorm in *module* with _HLFBRMSNorm.

    Call on each wrapper (encoder, prefill, decode) before torch.export when
    targeting GPU or NPU delegates.  The replacement shares the original
    weight tensor — no parameter duplication.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, _HLFBRMSNorm(child.eps, child.weight))
        else:
            patch_rms_norms(child)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_rotary(rotary_emb, max_len: int) -> tuple[Tensor, Tensor]:
    """Return (cos, sin) for positions 0..max_len-1."""
    rotary_emb._build_cache(max_len, device="cpu")
    cos = rotary_emb._cos_cached[:max_len].clone()
    sin = rotary_emb._sin_cached[:max_len].clone()
    return cos, sin


def _collect_dec_blocks(decoder: MammothDecoder):
    """Yield (block, stack) pairs over all decoder blocks in order."""
    for stack in decoder.stacks:
        for block in stack.blocks:
            yield block, stack


def _n_dec_layers(decoder: MammothDecoder) -> int:
    return sum(stack.depth for stack in decoder.stacks)


# ──────────────────────────────────────────────────────────────────────────────
# MammothLiteRTEncoder
# ──────────────────────────────────────────────────────────────────────────────

class MammothLiteRTEncoder(nn.Module):
    """
    Encoder wrapper: input_ids + float pad_mask → (encoder_hidden_states,).

    pad_mask: float32, shape (1, enc_max_len), 0 for valid tokens, -inf for padding.
    Equivalent to the bool attention_mask used by MammothEncoder but expressed as
    an additive SDPA mask so torch.export sees a static float input rather than a
    bool input with dynamic True/False values.
    """

    def __init__(self, hf_model: MammothForConditionalGeneration, enc_max_len: int):
        super().__init__()
        enc: MammothEncoder = hf_model.encoder

        # Register encoder sub-modules as children so torch.export can see params.
        # Both hf_model.encoder and this wrapper point to the same nn.Module objects
        # — no parameter duplication.
        self.token_emb = enc.token_emb
        self.post_emb_norm = enc.post_emb_norm
        self.stacks = enc.stacks

        if enc.rotary_emb is not None:
            cos, sin = _build_rotary(enc.rotary_emb, enc_max_len)
            self.register_buffer("cos", cos)  # (enc_max_len, dim_head)
            self.register_buffer("sin", sin)
            self._has_rotary = True
        else:
            self._has_rotary = False

        self._enc_max_len = enc_max_len

    def forward(self, input_ids: Tensor, pad_mask: Tensor) -> tuple[Tensor, ...]:
        """
        Args:
            input_ids : (1, enc_max_len) int64
            pad_mask  : (1, enc_max_len) float32, 0=valid, -inf=pad
        Returns:
            tuple of one tensor: encoder_hidden_states (1, enc_max_len, model_dim)
        """
        h = self.post_emb_norm(self.token_emb(input_ids))
        rotary = (self.cos, self.sin) if self._has_rotary else None
        mask = pad_mask[:, None, None, :]  # (1, 1, 1, enc_max_len) — broadcast to all heads/rows
        for stack in self.stacks:
            h, _ = stack(h, mask=mask, rotary=rotary, cache=None)
        return (h,)


# ──────────────────────────────────────────────────────────────────────────────
# MammothLiteRTPrefill
# ──────────────────────────────────────────────────────────────────────────────

class MammothLiteRTPrefill(nn.Module):
    """
    Decoder prefill: processes the full prefix at once (teacher-forcing style).

    Bypasses DecoderBlock.forward() to manually collect per-layer K/V tensors
    that are returned as flat positional outputs for the decode wrapper to consume.

    Input  (encoder_hidden_states, input_ids, causal_mask, cross_mask, *zero_kv)
    Output (logits, self_k_0, self_v_0, …, self_k_{N-1}, self_v_{N-1},
                    cross_k_0, cross_v_0, …, cross_k_{N-1}, cross_v_{N-1})
    """

    def __init__(
        self,
        hf_model: MammothForConditionalGeneration,
        enc_max_len: int,
        dec_max_len: int,
    ):
        super().__init__()
        dec: MammothDecoder = hf_model.decoder

        self.token_emb = dec.token_emb
        self.post_emb_norm = dec.post_emb_norm
        self.stacks = dec.stacks
        self.to_logits = dec.to_logits

        if dec.rotary_emb is not None:
            cos, sin = _build_rotary(dec.rotary_emb, dec_max_len)
            self.register_buffer("cos", cos)  # (dec_max_len, dim_head)
            self.register_buffer("sin", sin)
            self._has_rotary = True
        else:
            self._has_rotary = False

        self._dec_max_len = dec_max_len
        self._enc_max_len = enc_max_len
        self._n_layers = _n_dec_layers(dec)

    def forward(
        self,
        encoder_hidden_states: Tensor,  # (1, enc_max_len, model_dim)
        input_ids: Tensor,              # (1, dec_max_len) int64
        causal_mask: Tensor,            # (1, 1, dec_max_len, dec_max_len) float
        cross_mask: Tensor,             # (1, 1, dec_max_len, enc_max_len) float
        *zero_kv: Tensor,               # 4N placeholder tensors (shapes fix the graph)
    ) -> tuple[Tensor, ...]:
        h = self.post_emb_norm(self.token_emb(input_ids))

        # Rotary for full prefix (positions 0..dec_max_len-1, offset=0)
        rotary = (self.cos, self.sin) if self._has_rotary else None

        self_ks: list[Tensor] = []
        self_vs: list[Tensor] = []
        cross_ks: list[Tensor] = []
        cross_vs: list[Tensor] = []

        for stack in self.stacks:
            for block in stack.blocks:
                # ── Self-attention ──────────────────────────────────────────
                x_sa = block.norm1(h)
                q = block.self_attn._split_heads(block.self_attn.to_q(x_sa))
                k = block.self_attn._split_heads(block.self_attn.to_k(x_sa))
                v = block.self_attn._split_heads(block.self_attn.to_v(x_sa))
                if rotary is not None:
                    q, k = apply_rotary(q, k, self.cos, self.sin)
                sa_out = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=causal_mask, is_causal=False
                )
                h = h + block.self_attn.to_out(block.self_attn._merge_heads(sa_out))

                self_ks.append(k)   # (1, heads, dec_max_len, dim_head)
                self_vs.append(v)

                # ── Cross-attention ─────────────────────────────────────────
                x_ca = block.norm2(h)
                q_c = block.cross_attn._split_heads(block.cross_attn.to_q(x_ca))
                k_c = block.cross_attn._split_heads(block.cross_attn.to_k(encoder_hidden_states))
                v_c = block.cross_attn._split_heads(block.cross_attn.to_v(encoder_hidden_states))
                ca_out = F.scaled_dot_product_attention(
                    q_c, k_c, v_c, attn_mask=cross_mask, is_causal=False
                )
                h = h + block.cross_attn.to_out(block.cross_attn._merge_heads(ca_out))

                cross_ks.append(k_c)  # (1, heads, enc_max_len, dim_head)
                cross_vs.append(v_c)

                # ── Feed-forward ────────────────────────────────────────────
                h = h + block.ff(block.norm3(h))

            if stack.final_norm is not None:
                h = stack.final_norm(h)

        logits = self.to_logits(h)  # (1, dec_max_len, vocab)

        # Interleave K/V pairs as required by the flat convention:
        # self_k_0, self_v_0, …, cross_k_0, cross_v_0, …
        self_kv_flat = [t for pair in zip(self_ks, self_vs) for t in pair]
        cross_kv_flat = [t for pair in zip(cross_ks, cross_vs) for t in pair]
        return (logits, *self_kv_flat, *cross_kv_flat)


# ──────────────────────────────────────────────────────────────────────────────
# MammothLiteRTDecode
# ──────────────────────────────────────────────────────────────────────────────

class MammothLiteRTDecode(nn.Module):
    """
    Single-step decoder: processes one new token and updates the self-KV cache.

    Cross-KV is read from the prefill cache (no cross-projection at decode time).
    Self-KV is updated at position step_index using a static-shape scatter pattern
    compatible with torch.export (torch.where over a position mask).

    Input  (encoder_hidden_states, input_ids, causal_mask, cross_mask, step_index,
            self_k_0, self_v_0, …, cross_k_0, cross_v_0, …)
    Output (logits, self_k_0_updated, self_v_0_updated, …)
    """

    def __init__(
        self,
        hf_model: MammothForConditionalGeneration,
        enc_max_len: int,
        dec_max_len: int,
    ):
        super().__init__()
        dec: MammothDecoder = hf_model.decoder

        self.token_emb = dec.token_emb
        self.post_emb_norm = dec.post_emb_norm
        self.stacks = dec.stacks
        self.to_logits = dec.to_logits

        if dec.rotary_emb is not None:
            cos, sin = _build_rotary(dec.rotary_emb, dec_max_len)
            self.register_buffer("cos", cos)  # (dec_max_len, dim_head)
            self.register_buffer("sin", sin)
            self._has_rotary = True
        else:
            self._has_rotary = False

        self._dec_max_len = dec_max_len
        self._enc_max_len = enc_max_len
        self._n_layers = _n_dec_layers(dec)

    def forward(
        self,
        encoder_hidden_states: Tensor,  # (1, enc_max_len, model_dim)
        input_ids: Tensor,              # (1, 1) int64
        causal_mask: Tensor,            # (1, 1, 1, dec_max_len) float, 0/-inf
        cross_mask: Tensor,             # (1, 1, 1, enc_max_len) float, 0/-inf
        step_index: Tensor,             # () scalar int64 — current decode position
        *kv_flat: Tensor,               # self_k_0, self_v_0, …, cross_k_0, cross_v_0, …
    ) -> tuple[Tensor, ...]:
        n = self._n_layers
        # Split flat KV into self and cross components
        # kv_flat layout: [self_k_0, self_v_0, …, self_k_{n-1}, self_v_{n-1},
        #                   cross_k_0, cross_v_0, …, cross_k_{n-1}, cross_v_{n-1}]
        self_ks_in = [kv_flat[2 * i]     for i in range(n)]
        self_vs_in = [kv_flat[2 * i + 1] for i in range(n)]
        cross_ks   = [kv_flat[2 * n + 2 * i]     for i in range(n)]
        cross_vs   = [kv_flat[2 * n + 2 * i + 1] for i in range(n)]

        h = self.post_emb_norm(self.token_emb(input_ids))  # (1, 1, model_dim)

        # Rotary for single token at step_index (dynamic position)
        if self._has_rotary:
            idx = step_index.unsqueeze(0)  # (1,)
            cos_step = torch.index_select(self.cos, 0, idx)  # (1, dim_head)
            sin_step = torch.index_select(self.sin, 0, idx)
            rotary = (cos_step, sin_step)
        else:
            rotary = None

        # Position mask for updating self-KV at step_index (static shape, data-dep values)
        pos = torch.arange(self._dec_max_len, device=step_index.device)
        step_mask = (pos == step_index).view(1, 1, self._dec_max_len, 1)  # (1,1,L,1)

        updated_self_ks: list[Tensor] = []
        updated_self_vs: list[Tensor] = []

        layer_idx = 0
        for stack in self.stacks:
            for block in stack.blocks:
                self_k_cache = self_ks_in[layer_idx]   # (1, heads, dec_max_len, dim_head)
                self_v_cache = self_vs_in[layer_idx]
                cross_k = cross_ks[layer_idx]           # (1, heads, enc_max_len, dim_head)
                cross_v = cross_vs[layer_idx]

                # ── Self-attention ──────────────────────────────────────────
                x_sa = block.norm1(h)
                q = block.self_attn._split_heads(block.self_attn.to_q(x_sa))
                k_new = block.self_attn._split_heads(block.self_attn.to_k(x_sa))
                v_new = block.self_attn._split_heads(block.self_attn.to_v(x_sa))
                if rotary is not None:
                    q, k_new = apply_rotary(q, k_new, cos_step, sin_step)

                # Write new K/V at step_index using static-shape scatter
                k_updated = torch.where(step_mask, k_new.expand_as(self_k_cache), self_k_cache)
                v_updated = torch.where(step_mask, v_new.expand_as(self_v_cache), self_v_cache)

                # Attend Q (single token) to full K,V buffer; causal_mask zeroes future slots
                sa_out = F.scaled_dot_product_attention(
                    q, k_updated, v_updated, attn_mask=causal_mask, is_causal=False
                )
                h = h + block.self_attn.to_out(block.self_attn._merge_heads(sa_out))

                updated_self_ks.append(k_updated)
                updated_self_vs.append(v_updated)

                # ── Cross-attention (reuse prefill cache) ───────────────────
                x_ca = block.norm2(h)
                q_c = block.cross_attn._split_heads(block.cross_attn.to_q(x_ca))
                ca_out = F.scaled_dot_product_attention(
                    q_c, cross_k, cross_v, attn_mask=cross_mask, is_causal=False
                )
                h = h + block.cross_attn.to_out(block.cross_attn._merge_heads(ca_out))

                # ── Feed-forward ────────────────────────────────────────────
                h = h + block.ff(block.norm3(h))

                layer_idx += 1

            if stack.final_norm is not None:
                h = stack.final_norm(h)

        logits = self.to_logits(h)  # (1, 1, vocab)

        # Return logits + updated self-KV (cross-KV is unchanged)
        self_kv_flat = [t for pair in zip(updated_self_ks, updated_self_vs) for t in pair]
        return (logits, *self_kv_flat)


# ──────────────────────────────────────────────────────────────────────────────
# Sample input factories
# ──────────────────────────────────────────────────────────────────────────────

def make_encoder_sample_inputs(
    enc_max_len: int,
    src_vocab_size: int = 32000,
) -> tuple[Tensor, Tensor]:
    """Return (input_ids, pad_mask) for the encoder wrapper."""
    input_ids = torch.zeros(1, enc_max_len, dtype=torch.long)
    pad_mask = torch.zeros(1, enc_max_len)  # all valid, float32
    return (input_ids, pad_mask)


def make_prefill_sample_inputs(
    enc_out: tuple[Tensor, ...],
    enc_max_len: int,
    dec_max_len: int,
    config: MammothConfig,
) -> tuple[Tensor, ...]:
    """
    Return the full tuple expected by MammothLiteRTPrefill.forward().

    The zero_kv placeholder tensors fix the graph shapes for export even though
    prefill ignores their values (it recomputes all KV from scratch).
    """
    encoder_hidden_states = enc_out[0]
    input_ids = torch.zeros(1, dec_max_len, dtype=torch.long)

    # Lower-triangular causal mask: 0 on/below diagonal, -inf above
    causal_mask = torch.full((1, 1, dec_max_len, dec_max_len), float("-inf"))
    causal_mask = torch.triu(causal_mask, diagonal=1)

    # Cross mask: all-zero (attend to all encoder positions, no padding)
    cross_mask = torch.zeros(1, 1, dec_max_len, enc_max_len)

    n_layers = sum(config.dec_layers)
    heads = config.heads
    dim_head = config.model_dim // heads

    zero_self_kv = [
        torch.zeros(1, heads, dec_max_len, dim_head)
        for _ in range(2 * n_layers)
    ]
    zero_cross_kv = [
        torch.zeros(1, heads, enc_max_len, dim_head)
        for _ in range(2 * n_layers)
    ]

    return (
        encoder_hidden_states,
        input_ids,
        causal_mask,
        cross_mask,
        *zero_self_kv,
        *zero_cross_kv,
    )


def make_decode_sample_inputs(
    enc_out: tuple[Tensor, ...],
    prefill_out: tuple[Tensor, ...],
    enc_max_len: int,
    dec_max_len: int,
    config: MammothConfig,
    step_index: int = 0,
    input_ids: Tensor | None = None,
) -> tuple[Tensor, ...]:
    """
    Return the full tuple expected by MammothLiteRTDecode.forward().

    Uses KV tensors from prefill_out as the starting cache.
    """
    encoder_hidden_states = enc_out[0]
    if input_ids is None:
        input_ids = torch.zeros(1, 1, dtype=torch.long)

    # Single-row causal mask: 0 for positions 0..step_index, -inf for the rest
    row = torch.full((1, 1, 1, dec_max_len), float("-inf"))
    row[..., : step_index + 1] = 0.0

    cross_mask = torch.zeros(1, 1, 1, enc_max_len)
    step_idx_tensor = torch.tensor(step_index, dtype=torch.long)

    n_layers = sum(config.dec_layers)
    # Extract self-KV and cross-KV from prefill output
    # prefill_out = (logits, self_k_0, self_v_0, …, cross_k_0, cross_v_0, …)
    self_kv = list(prefill_out[1 : 1 + 2 * n_layers])
    cross_kv = list(prefill_out[1 + 2 * n_layers : 1 + 4 * n_layers])

    return (
        encoder_hidden_states,
        input_ids,
        row,
        cross_mask,
        step_idx_tensor,
        *self_kv,
        *cross_kv,
    )
