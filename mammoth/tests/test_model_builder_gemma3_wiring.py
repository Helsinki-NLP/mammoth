"""
Tests that model_builder.py::build_xcoder correctly wires the Gemma3-style
per-layer decoder config (dual RoPE + sliding window pattern borrowed from
transformers.models.gemma3.configuration_gemma3.Gemma3TextConfig, GQA,
QK-norm, sandwich norm, GeGLU) through model_opts. See CLAUDE.md
"Gemma3-270M -> Mammoth Conversion" for the full design.
"""
from argparse import Namespace

import torch

from mammoth.model_builder import build_xcoder
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.distributed.components import Side
from mammoth.distributed.tasks import TaskSpecs, TaskQueueManager, RoundRobinTaskDistributionStrategy
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum


VOCABS = {
    ('src', 'a'): Vocab(None, items=['a'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
    ('tgt', 'b'): Vocab(None, items=[f'_{i}' for i in range(20)], tag='dummy', specials=list(DEFAULT_SPECIALS)),
}

TASK_SPECS = {
    'dummy_a-b': TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang='a',
        tgt_lang='b',
        encoder_id=['foo'],
        decoder_id=['bar'],
        corpus_id='a-b',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=VOCABS[('src', 'a')],
        tgt_vocab=VOCABS[('tgt', 'b')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    ),
}


def _build_tqm(model_opts):
    world_context = WorldContext(DeviceContextEnum.MULTI_GPU, n_nodes=1, gpus_per_node=2)
    tqm = TaskQueueManager(
        tasks=[TASK_SPECS['dummy_a-b']],
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=RoundRobinTaskDistributionStrategy,
        uses_adapters=False,
    ).global_to_local(node_rank=0, local_rank=0, opts=model_opts)
    tqm.create_all_distributed_components(use_attention_bridge=False)
    return tqm


def _gemma3_style_opts(num_layers=6, pattern=3):
    opts = Namespace()
    opts.seed = 1
    opts.dec_layers = [num_layers]
    opts.enc_layers = [1]
    opts.dec_model_dim = 32
    opts.dec_heads = 4
    opts.dec_attn_dim_head = 16
    opts.dec_attn_kv_heads = 1
    opts.dec_attn_qk_norm = True
    opts.dec_sandwich_norm = True
    opts.dec_ff_activation = "geglu"
    opts.dec_sliding_window = 4
    opts.dec_global_attn_every_n_layers = pattern
    opts.dec_global_rope_theta = 1_000_000.0
    opts.dec_local_rope_theta = 10_000.0
    opts.dec_scaled_embeddings = True
    opts.rotary_pos_emb = True
    opts.post_emb_norm = False
    return opts


class TestGemma3StyleDecoderWiring:
    def test_layer_pattern_matches_gemma3(self):
        # Gemma3TextConfig: "sliding_attention" if (i+1) % pattern else "full_attention"
        # i.e. full/global attention lands on the LAST layer of every group of `pattern`.
        opts = _gemma3_style_opts(num_layers=6, pattern=3)
        tqm = _build_tqm(opts)
        vocabs_dict = {('src', 'a'): VOCABS[('src', 'a')], ('tgt', 'b'): VOCABS[('tgt', 'b')]}
        dec = build_xcoder(Side.decoder, opts, vocabs_dict, 'cpu', task_queue_manager=tqm)
        stack = dec.get_attention_layers_by_xcoder_id(0, 'bar')
        blocks = list(stack.blocks)
        assert len(blocks) == 6
        for i, block in enumerate(blocks):
            is_global = (i + 1) % 3 == 0
            if is_global:
                assert block.sliding_window is None
                assert block.rotary_emb.base == 1_000_000.0
            else:
                assert block.sliding_window == 4
                assert block.rotary_emb.base == 10_000.0
            assert block.self_attn.kv_heads == 1
            assert block.self_attn.qk_norm is True
            assert block.sandwich_norm is True
            assert block.ff.activation == "geglu"

    def test_forward_pass_shapes(self):
        opts = _gemma3_style_opts(num_layers=4, pattern=2)
        tqm = _build_tqm(opts)
        vocabs_dict = {('src', 'a'): VOCABS[('src', 'a')], ('tgt', 'b'): VOCABS[('tgt', 'b')]}
        dec = build_xcoder(Side.decoder, opts, vocabs_dict, 'cpu', task_queue_manager=tqm)
        task = TASK_SPECS['dummy_a-b']
        active = dec.activate(task_id=task.corpus_id)
        x = torch.randint(0, len(VOCABS[('tgt', 'b')]), (2, 5))
        context = torch.randn(2, 3, 32)
        logits = active(x, context=context)
        assert logits.shape == (2, 5, len(VOCABS[('tgt', 'b')]))

    def test_backward_compatible_without_gemma3_opts(self):
        # Plain decoder config (no Gemma3-style opts set) must build/run exactly
        # like before these features existed.
        opts = Namespace()
        opts.seed = 1
        opts.dec_layers = [2]
        opts.enc_layers = [1]
        opts.dec_model_dim = 16
        opts.dec_heads = 2
        opts.rotary_pos_emb = False
        opts.post_emb_norm = False
        tqm = _build_tqm(opts)
        vocabs_dict = {('src', 'a'): VOCABS[('src', 'a')], ('tgt', 'b'): VOCABS[('tgt', 'b')]}
        dec = build_xcoder(Side.decoder, opts, vocabs_dict, 'cpu', task_queue_manager=tqm)
        stack = dec.get_attention_layers_by_xcoder_id(0, 'bar')
        for block in stack.blocks:
            assert block.sliding_window is None
            assert block.rotary_emb is None
            assert block.self_attn.kv_heads == block.self_attn.heads
            assert block.self_attn.qk_norm is False
            assert block.sandwich_norm is False
