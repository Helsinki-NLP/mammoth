"""
Unit tests for the true decoder-only DecoderBlock mode (`use_cross_attn=False`).

See CLAUDE.md "Gemma3-270M -> Mammoth Conversion" -> "Decision: fake-encoder
path chosen for this conversion" for context: the fake-encoder converter
(convert_gemma3_native.py) bolts on a small randomly-initialized encoder plus
a zero-initialized cross-attention module to satisfy Mammoth's NMT-shaped
task plumbing. This test file exercises the alternative named there as the
preferred long-term direction: a DecoderBlock that never builds or runs
cross-attention at all, used by convert_gemma3_decoder_only.py.

Written before the corresponding implementation (see block.py DecoderBlock).
"""
import torch

from mammoth.modules.transformer.block import DecoderBlock


torch.manual_seed(0)


class TestDecoderBlockNoCrossAttn:
    def test_cross_attn_module_not_built(self):
        block = DecoderBlock(dim=16, heads=4, use_cross_attn=False)
        assert not hasattr(block, "cross_attn") or block.cross_attn is None
        assert not hasattr(block, "norm2") or block.norm2 is None

    def test_default_still_builds_cross_attn(self):
        # use_cross_attn defaults to True: existing fake-encoder callers must
        # be unaffected by this change.
        block = DecoderBlock(dim=16, heads=4)
        assert block.cross_attn is not None
        assert block.norm2 is not None

    def test_forward_without_context_runs(self):
        block = DecoderBlock(dim=16, heads=4, use_cross_attn=False)
        x = torch.randn(2, 5, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_forward_ignores_context_argument_if_given_anyway(self):
        # Even if a caller mistakenly passes context, a decoder-only block
        # must not use it -- there is no cross_attn module to consume it.
        block = DecoderBlock(dim=16, heads=4, use_cross_attn=False)
        x = torch.randn(2, 5, 16)
        context = torch.randn(2, 3, 16)
        out_with_context = block(x, context=context)
        out_without_context = block(x, context=None)
        torch.testing.assert_close(out_with_context, out_without_context)

    def test_matches_manual_self_attn_and_ff_only_computation(self):
        block = DecoderBlock(dim=16, heads=4, use_cross_attn=False)
        block.eval()
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            out = block(x)

            h, _ = block.self_attn(block.norm1(x))
            expected = x + h
            h = block.ff(block.norm3(expected))
            expected = expected + h

        torch.testing.assert_close(out, expected)

    def test_sandwich_norm_still_applies_without_cross_attn(self):
        block = DecoderBlock(dim=16, heads=4, use_cross_attn=False, sandwich_norm=True)
        block.eval()
        assert hasattr(block, "norm1_post")
        assert hasattr(block, "norm3_post")
        x = torch.randn(2, 5, 16)
        with torch.no_grad():
            out = block(x)

            h, _ = block.self_attn(block.norm1(x))
            h = block.norm1_post(h)
            expected = x + h
            h = block.ff(block.norm3(expected))
            h = block.norm3_post(h)
            expected = expected + h

        torch.testing.assert_close(out, expected)

    def test_no_cross_attn_has_fewer_parameters_than_default(self):
        block_with = DecoderBlock(dim=16, heads=4, use_cross_attn=True)
        block_without = DecoderBlock(dim=16, heads=4, use_cross_attn=False)
        params_with = sum(p.numel() for p in block_with.parameters())
        params_without = sum(p.numel() for p in block_without.parameters())
        assert params_without < params_with

    def test_kv_cache_forward_without_context(self):
        from mammoth.modules.transformer.cache import LayerCache

        block = DecoderBlock(dim=16, heads=4, use_cross_attn=False)
        block.eval()
        x = torch.randn(1, 5, 16)
        cache = LayerCache()
        with torch.no_grad():
            full_out = block(x)
            out_chunks = []
            for i in range(x.size(1)):
                out_chunk = block(x[:, i:i + 1, :], cache=cache)
                out_chunks.append(out_chunk)
            incremental_out = torch.cat(out_chunks, dim=1)
        torch.testing.assert_close(incremental_out, full_out, atol=1e-5, rtol=1e-5)
