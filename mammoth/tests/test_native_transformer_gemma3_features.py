"""
Unit tests for native-PyTorch-backend features needed to correctly host
Gemma3 (see CLAUDE.md "Gemma3-270M -> Mammoth Conversion" section for the
full gap analysis). Written before the corresponding implementation.
"""
import math

import pytest
import torch
import torch.nn.functional as F

from mammoth.modules.transformer.attention import MultiHeadAttention
from mammoth.modules.transformer.ffn import FeedForward
from mammoth.modules.transformer.masking import build_sliding_window_causal_mask
from mammoth.modules.transformer.block import DecoderBlock
from mammoth.modules.transformer.rotary import RotaryEmbedding
from mammoth.modules.transformer.wrapper import NativeTransformerWrapper


torch.manual_seed(0)


# ---------------------------------------------------------------------------
# masking.py: sliding-window causal mask
# ---------------------------------------------------------------------------
class TestSlidingWindowCausalMask:
    def test_shape_and_dtype(self):
        mask = build_sliding_window_causal_mask(seq_len=6, window=3, device=torch.device('cpu'))
        assert mask.shape == (6, 6)
        assert mask.dtype == torch.bool

    def test_causal_component(self):
        # window large enough to be a no-op -> pure causal mask
        mask = build_sliding_window_causal_mask(seq_len=4, window=100, device=torch.device('cpu'))
        expected = torch.tril(torch.ones(4, 4, dtype=torch.bool))
        assert torch.equal(mask, expected)

    def test_window_component(self):
        # window=3: position i can see positions (i-2 .. i), inclusive, and never future
        mask = build_sliding_window_causal_mask(seq_len=6, window=3, device=torch.device('cpu'))
        # row 5 (0-indexed) should allow keys 3,4,5 only
        assert mask[5].tolist() == [False, False, False, True, True, True]
        # row 0 should only allow key 0
        assert mask[0].tolist() == [True, False, False, False, False, False]
        # row 2 (window not yet full) should allow keys 0,1,2
        assert mask[2].tolist() == [True, True, True, False, False, False]


# ---------------------------------------------------------------------------
# ffn.py: GeGLU (gated GELU-tanh) activation
# ---------------------------------------------------------------------------
class TestFeedForwardGeGLU:
    def test_output_shape(self):
        ff = FeedForward(dim=16, ff_mult=2.0, activation="geglu")
        x = torch.randn(2, 5, 16)
        out = ff(x)
        assert out.shape == (2, 5, 16)

    def test_matches_manual_formula(self):
        ff = FeedForward(dim=8, ff_mult=2.0, activation="geglu")
        x = torch.randn(3, 8)
        with torch.no_grad():
            out = ff(x)
            gate = F.gelu(ff.w1(x), approximate="tanh")
            up = ff.w3(x)
            expected = ff.w2(gate * up)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_geglu_has_three_matrices(self):
        ff = FeedForward(dim=8, ff_mult=2.0, activation="geglu")
        assert ff.w1 is not None and ff.w2 is not None and ff.w3 is not None


# ---------------------------------------------------------------------------
# attention.py: GQA/MQA, QK-norm, custom attention scale
# ---------------------------------------------------------------------------
class TestMultiHeadAttentionGQA:
    def test_mqa_output_shape(self):
        attn = MultiHeadAttention(dim=32, heads=4, dim_head=8, kv_heads=1, causal=True)
        x = torch.randn(2, 6, 32)
        out, _ = attn(x)
        assert out.shape == (2, 6, 32)

    def test_kv_projections_use_kv_heads(self):
        attn = MultiHeadAttention(dim=32, heads=4, dim_head=8, kv_heads=1)
        assert attn.to_k.out_features == 1 * 8
        assert attn.to_v.out_features == 1 * 8
        assert attn.to_q.out_features == 4 * 8

    def test_default_kv_heads_equals_heads(self):
        attn = MultiHeadAttention(dim=32, heads=4, dim_head=8)
        assert attn.to_k.out_features == 4 * 8


class TestMultiHeadAttentionQKNorm:
    def test_qk_norm_modules_exist(self):
        attn = MultiHeadAttention(dim=32, heads=4, dim_head=8, qk_norm=True)
        assert hasattr(attn, 'q_norm')
        assert hasattr(attn, 'k_norm')

    def test_qk_norm_changes_output_vs_no_norm(self):
        torch.manual_seed(1)
        attn_plain = MultiHeadAttention(dim=32, heads=4, dim_head=8, qk_norm=False)
        attn_norm = MultiHeadAttention(dim=32, heads=4, dim_head=8, qk_norm=True)
        # copy weights so only qk_norm differs
        attn_norm.to_q.load_state_dict(attn_plain.to_q.state_dict())
        attn_norm.to_k.load_state_dict(attn_plain.to_k.state_dict())
        attn_norm.to_v.load_state_dict(attn_plain.to_v.state_dict())
        attn_norm.to_out.load_state_dict(attn_plain.to_out.state_dict())

        x = torch.randn(2, 5, 32)
        out_plain, _ = attn_plain(x)
        out_norm, _ = attn_norm(x)
        assert not torch.allclose(out_plain, out_norm)

    def test_qk_norm_applied_before_rotary_is_unit_rms(self):
        # With RMSNorm(elementwise_affine weight initialized to 1), the
        # per-head RMS of q/k after normalization should be ~1.
        attn = MultiHeadAttention(dim=32, heads=4, dim_head=8, qk_norm=True)
        x = torch.randn(2, 5, 32)
        q = attn._split_heads(attn.to_q(x))
        q_normed = attn.q_norm(q)
        rms = q_normed.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


class TestMultiHeadAttentionCustomScale:
    def test_custom_scale_changes_output(self):
        torch.manual_seed(2)
        attn_default = MultiHeadAttention(dim=16, heads=2, dim_head=8)
        attn_scaled = MultiHeadAttention(dim=16, heads=2, dim_head=8, attn_scale=0.01)
        attn_scaled.load_state_dict(attn_default.state_dict())

        x = torch.randn(2, 4, 16)
        out_default, _ = attn_default(x)
        out_scaled, _ = attn_scaled(x)
        assert not torch.allclose(out_default, out_scaled)

    def test_default_scale_is_none_equivalent_to_1_over_sqrt_dim_head(self):
        attn = MultiHeadAttention(dim=16, heads=2, dim_head=8, attn_scale=1.0 / math.sqrt(8))
        attn_default = MultiHeadAttention(dim=16, heads=2, dim_head=8)
        attn.load_state_dict(attn_default.state_dict())
        x = torch.randn(2, 4, 16)
        out_a, _ = attn(x)
        out_b, _ = attn_default(x)
        assert torch.allclose(out_a, out_b, atol=1e-5)


# ---------------------------------------------------------------------------
# block.py: sandwich norm + sliding window + per-block rotary theta
# ---------------------------------------------------------------------------
class TestDecoderBlockNormEps:
    def test_default_norm_eps_is_none_backward_compatible(self):
        block = DecoderBlock(dim=8, heads=2, ff_mult=2.0)
        assert block.norm1.eps is None
        assert not hasattr(block.self_attn, 'q_norm')

    def test_norm_eps_propagates_to_all_norms_and_qk_norm(self):
        # A fixed eps (Gemma3 uses 1e-6) must reach every RMSNorm in the
        # block, including self-attn's qk_norm -- torch's default eps
        # (~1.19e-7 for fp32) differs enough from 1e-6 to produce very
        # different outputs on near-zero-variance activation rows, especially
        # when combined with a large-magnitude norm weight (this is exactly
        # what caused a silent ~59-magnitude divergence converting Gemma3's
        # layer 5 before this parameter existed).
        block = DecoderBlock(dim=8, heads=2, ff_mult=2.0, sandwich_norm=True, qk_norm=True, norm_eps=1e-6)
        assert block.norm1.eps == 1e-6
        assert block.norm2.eps == 1e-6
        assert block.norm3.eps == 1e-6
        assert block.norm1_post.eps == 1e-6
        assert block.norm3_post.eps == 1e-6
        assert block.self_attn.q_norm.eps == 1e-6
        assert block.self_attn.k_norm.eps == 1e-6

    def test_eps_difference_diverges_on_near_zero_variance_row(self):
        # Isolate the actual failure mode at the RMSNorm level: torch's
        # dtype-default eps vs Gemma3's fixed 1e-6 diverge sharply once
        # mean(x**2) is itself tiny and the norm weight is large.
        dim = 8
        x = torch.full((1, 1, dim), 1e-4)
        norm_default_eps = torch.nn.RMSNorm(dim, elementwise_affine=True)
        norm_gemma_eps = torch.nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        with torch.no_grad():
            norm_default_eps.weight.fill_(50.0)
            norm_gemma_eps.weight.fill_(50.0)
        out_default = norm_default_eps(x)
        out_gemma = norm_gemma_eps(x)
        assert not torch.allclose(out_default, out_gemma, rtol=0.1)


class TestDecoderBlockSandwichNorm:
    def test_sandwich_norm_has_four_norms(self):
        block = DecoderBlock(dim=16, heads=2, ff_mult=2.0, sandwich_norm=True)
        # 2 norms around self-attn (pre+post) + 2 around ff (pre+post) + cross-attn's own pre-norm
        assert hasattr(block, 'norm1_post')
        assert hasattr(block, 'norm3_post')  # ff post-branch norm (norm3 = pre-ff norm)

    def test_sandwich_norm_forward_matches_manual(self):
        torch.manual_seed(3)
        block = DecoderBlock(dim=8, heads=2, ff_mult=2.0, sandwich_norm=True)
        block.eval()
        x = torch.randn(1, 3, 8)
        context = torch.randn(1, 3, 8)

        with torch.no_grad():
            out = block(x, context=context)

            residual = x
            h, _ = block.self_attn(block.norm1(x), rotary=None, cache=None)
            h = block.norm1_post(h)
            x2 = residual + h

            residual = x2
            h, _ = block.cross_attn(block.norm2(x2), context=context, cache=None)
            x2 = residual + h

            residual = x2
            h = block.ff(block.norm3(x2))
            h = block.norm3_post(h)
            expected = residual + h

        assert torch.allclose(out, expected, atol=1e-6)

    def test_no_sandwich_norm_is_backward_compatible(self):
        # default behaviour (sandwich_norm=False) must match the pre-existing
        # 3-norm DecoderBlock forward pass exactly.
        torch.manual_seed(4)
        block = DecoderBlock(dim=8, heads=2, ff_mult=2.0)
        assert not hasattr(block, 'norm1_post')
        x = torch.randn(1, 3, 8)
        context = torch.randn(1, 3, 8)
        out = block(x, context=context)
        assert out.shape == x.shape


class TestDecoderBlockSlidingWindow:
    def test_sliding_window_restricts_attention(self):
        torch.manual_seed(5)
        dim, heads, seq = 8, 2, 6
        block_full = DecoderBlock(dim=dim, heads=heads, ff_mult=2.0)
        block_window = DecoderBlock(dim=dim, heads=heads, ff_mult=2.0, sliding_window=2)
        block_window.load_state_dict(block_full.state_dict())
        block_full.eval()
        block_window.eval()

        x = torch.randn(1, seq, dim)
        context = torch.randn(1, seq, dim)
        out_full = block_full(x, context=context)
        out_window = block_window(x, context=context)
        # With a small window, outputs must differ from full causal attention
        # once sequence length exceeds the window.
        assert not torch.allclose(out_full, out_window)

    def test_sliding_window_first_positions_unaffected(self):
        # first `window` positions see the same keys under both regimes,
        # so their outputs should match exactly for position 0 (self-only).
        torch.manual_seed(6)
        dim, heads = 8, 2
        block_full = DecoderBlock(dim=dim, heads=heads, ff_mult=2.0)
        block_window = DecoderBlock(dim=dim, heads=heads, ff_mult=2.0, sliding_window=2)
        block_window.load_state_dict(block_full.state_dict())
        block_full.eval()
        block_window.eval()

        x = torch.randn(1, 1, dim)  # single token: window vs full identical
        context = torch.randn(1, 1, dim)
        out_full = block_full(x, context=context)
        out_window = block_window(x, context=context)
        assert torch.allclose(out_full, out_window, atol=1e-6)


class TestDecoderBlockOwnRotary:
    def test_block_with_own_rotary_ignores_passed_rotary(self):
        torch.manual_seed(7)
        dim, heads, dim_head = 8, 2, 4
        own_rotary = RotaryEmbedding(dim_head, base=10)
        block = DecoderBlock(dim=dim, heads=heads, ff_mult=2.0, rotary_emb=own_rotary)
        block.eval()
        x = torch.randn(1, 4, dim)
        context = torch.randn(1, 4, dim)

        # passing a mismatched external rotary tuple should have no effect
        bogus_rotary = (torch.zeros(4, dim_head), torch.zeros(4, dim_head))
        out_with_bogus = block(x, context=context, rotary=bogus_rotary)
        out_without = block(x, context=context, rotary=None)
        assert torch.allclose(out_with_bogus, out_without, atol=1e-6)


# ---------------------------------------------------------------------------
# wrapper.py: scaled embeddings (Gemma3TextScaledWordEmbedding, x sqrt(dim))
# ---------------------------------------------------------------------------
class TestNativeTransformerWrapperEmbedScale:
    def test_embed_scale_multiplies_embedding_output(self):
        torch.manual_seed(8)
        dim, vocab = 8, 5
        emb = torch.nn.Embedding(vocab, dim)
        wrapper_plain = NativeTransformerWrapper(
            token_emb=emb, post_emb_norm=torch.nn.Identity(), stacks=[], rotary_emb=None,
            to_logits=None, return_only_embed=True,
        )
        wrapper_scaled = NativeTransformerWrapper(
            token_emb=emb, post_emb_norm=torch.nn.Identity(), stacks=[], rotary_emb=None,
            to_logits=None, return_only_embed=True, embed_scale=dim ** 0.5,
        )
        x = torch.tensor([[0, 1, 2]])
        out_plain = wrapper_plain(x)
        out_scaled = wrapper_scaled(x)
        assert torch.allclose(out_scaled, out_plain * (dim ** 0.5), atol=1e-6)
