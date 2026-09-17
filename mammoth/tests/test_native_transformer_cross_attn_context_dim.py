"""
Tests for cross-attention supporting a context (encoder output) dimension
that differs from the decoder's own dimension, via an explicit linear
projection on to_k/to_v -- the native-backend equivalent of the old
x_transformers-backend `dec_cross_attn_dim_context` option.

Without this, MultiHeadAttention.to_k/to_v always projected from the
block's own `dim`, silently requiring enc_model_dim == dec_model_dim
whenever cross-attention is used (see CLAUDE.md "Gemma3-270M -> Mammoth
Conversion" known-limitations note, which this closes).
"""
from argparse import Namespace

import torch

from mammoth.distributed.components import Side
from mammoth.model_builder import build_xcoder
from mammoth.modules.transformer.attention import MultiHeadAttention
from mammoth.modules.transformer.block import DecoderBlock
from mammoth.tests.test_model_builder_gemma3_wiring import TASK_SPECS, VOCABS, _build_tqm


class TestMultiHeadAttentionContextDim:
    def test_default_context_dim_equals_dim(self):
        attn = MultiHeadAttention(dim=16, heads=2, dim_head=8, is_cross_attn=True)
        assert attn.to_k.in_features == 16
        assert attn.to_v.in_features == 16

    def test_explicit_context_dim_overrides_kv_input_size(self):
        attn = MultiHeadAttention(dim=16, heads=2, dim_head=8, is_cross_attn=True, context_dim=8)
        assert attn.to_k.in_features == 8
        assert attn.to_v.in_features == 8
        assert attn.to_q.in_features == 16  # query still projects from the block's own dim

    def test_forward_with_mismatched_context_dim(self):
        attn = MultiHeadAttention(dim=16, heads=2, dim_head=8, is_cross_attn=True, context_dim=8)
        x = torch.randn(2, 5, 16)
        context = torch.randn(2, 3, 8)
        out, _ = attn(x, context=context)
        assert out.shape == (2, 5, 16)

    def test_forward_backward_compatible_without_context_dim(self):
        attn = MultiHeadAttention(dim=16, heads=2, dim_head=8, is_cross_attn=True)
        x = torch.randn(2, 5, 16)
        context = torch.randn(2, 3, 16)
        out, _ = attn(x, context=context)
        assert out.shape == (2, 5, 16)


class TestDecoderBlockContextDim:
    def test_context_dim_propagates_to_cross_attn(self):
        block = DecoderBlock(dim=16, heads=2, ff_mult=2.0, context_dim=8)
        assert block.cross_attn.to_k.in_features == 8
        assert block.cross_attn.to_v.in_features == 8

    def test_forward_with_smaller_context_dim(self):
        block = DecoderBlock(dim=16, heads=2, ff_mult=2.0, context_dim=8)
        block.eval()
        x = torch.randn(1, 4, 16)
        context = torch.randn(1, 3, 8)
        out = block(x, context=context)
        assert out.shape == (1, 4, 16)

    def test_default_context_dim_backward_compatible(self):
        block = DecoderBlock(dim=16, heads=2, ff_mult=2.0)
        assert block.cross_attn.to_k.in_features == 16


class TestModelBuilderCrossAttnDim:
    def test_decoder_cross_attn_defaults_to_enc_model_dim(self):
        opts = Namespace()
        opts.seed = 1
        opts.dec_layers = [2]
        opts.enc_layers = [1]
        opts.dec_model_dim = 32
        opts.enc_model_dim = 16  # smaller than the decoder -- the old gap
        opts.rotary_pos_emb = False
        opts.post_emb_norm = False
        tqm = _build_tqm(opts)
        vocabs_dict = {('src', 'a'): VOCABS[('src', 'a')], ('tgt', 'b'): VOCABS[('tgt', 'b')]}
        dec = build_xcoder(Side.decoder, opts, vocabs_dict, 'cpu', task_queue_manager=tqm)
        stack = dec.get_attention_layers_by_xcoder_id(0, 'bar')
        for block in stack.blocks:
            assert block.cross_attn.to_k.in_features == 16
            assert block.cross_attn.to_v.in_features == 16

    def test_end_to_end_forward_with_smaller_encoder(self):
        opts = Namespace()
        opts.seed = 1
        opts.dec_layers = [2]
        opts.enc_layers = [1]
        opts.dec_model_dim = 32
        opts.enc_model_dim = 16
        opts.rotary_pos_emb = False
        opts.post_emb_norm = False
        tqm = _build_tqm(opts)
        vocabs_dict = {('src', 'a'): VOCABS[('src', 'a')], ('tgt', 'b'): VOCABS[('tgt', 'b')]}
        dec = build_xcoder(Side.decoder, opts, vocabs_dict, 'cpu', task_queue_manager=tqm)
        task = TASK_SPECS['dummy_a-b']
        active = dec.activate(task_id=task.corpus_id)
        x = torch.randint(0, len(VOCABS[('tgt', 'b')]), (2, 5))
        context = torch.randn(2, 3, 16)  # encoder output at enc_model_dim, not dec_model_dim
        logits = active(x, context=context)
        assert logits.shape == (2, 5, len(VOCABS[('tgt', 'b')]))

    def test_explicit_dec_cross_attn_dim_context_overrides_enc_model_dim(self):
        opts = Namespace()
        opts.seed = 1
        opts.dec_layers = [1]
        opts.enc_layers = [1]
        opts.dec_model_dim = 32
        opts.enc_model_dim = 16
        opts.dec_cross_attn_dim_context = 24
        opts.rotary_pos_emb = False
        opts.post_emb_norm = False
        tqm = _build_tqm(opts)
        vocabs_dict = {('src', 'a'): VOCABS[('src', 'a')], ('tgt', 'b'): VOCABS[('tgt', 'b')]}
        dec = build_xcoder(Side.decoder, opts, vocabs_dict, 'cpu', task_queue_manager=tqm)
        stack = dec.get_attention_layers_by_xcoder_id(0, 'bar')
        for block in stack.blocks:
            assert block.cross_attn.to_k.in_features == 24
