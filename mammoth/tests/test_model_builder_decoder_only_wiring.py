"""
Tests that model_builder.py::build_model can build a TRUE decoder-only
Mammoth model: no encoder is built at all (model.encoder is None), and the
decoder's DecoderBlocks never build/run cross-attention.

See CLAUDE.md "Gemma3-270M -> Mammoth Conversion" -> "Decision: fake-encoder
path chosen for this conversion (2026-09-16)", which names this ("option 1")
as the preferred long-term direction once Mammoth needs to host more
decoder-only LMs. convert_gemma3_decoder_only.py builds on this wiring.

Written before the corresponding implementation (see build_model /
build_xcoder in model_builder.py and NMTModel.forward in models/model.py).
"""
from argparse import Namespace

import torch

from mammoth.model_builder import build_model
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.distributed.components import DistributedEncoderAttentionLayersBlock
from mammoth.distributed.tasks import TaskSpecs, TaskQueueManager, RoundRobinTaskDistributionStrategy
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum


VOCABS = {
    ('tgt', 'b'): Vocab(None, items=[f'_{i}' for i in range(20)], tag='dummy', specials=list(DEFAULT_SPECIALS)),
}

# A decoder-only task has no encoder component at all: encoder_id=[].
# src_lang/src_vocab are dataclass-required but never consulted, since
# build_model must skip build_xcoder(Side.encoder) entirely for such tasks.
DECODER_ONLY_TASK = TaskSpecs(
    node_rank=0,
    local_rank=0,
    src_lang='b',
    tgt_lang='b',
    encoder_id=[],
    decoder_id=['bar'],
    corpus_id='dummy-lm',
    weight=1,
    introduce_at_training_step=0,
    corpus_opts=dict(),
    src_vocab=VOCABS[('tgt', 'b')],
    tgt_vocab=VOCABS[('tgt', 'b')],
    encoder_adapter_ids=None,
    decoder_adapter_ids=None,
)


def _build_tqm(opts, task=DECODER_ONLY_TASK):
    world_context = WorldContext(DeviceContextEnum.MULTI_GPU, n_nodes=1, gpus_per_node=2)
    tqm = TaskQueueManager(
        tasks=[task],
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=RoundRobinTaskDistributionStrategy,
        uses_adapters=False,
    ).global_to_local(node_rank=0, local_rank=0, opts=opts)
    tqm.create_all_distributed_components(use_attention_bridge=False)
    return tqm


def _decoder_only_opts(num_layers=2):
    opts = Namespace()
    opts.seed = 1
    opts.decoder_only = True
    opts.dec_layers = [num_layers]
    opts.dec_model_dim = 16
    opts.dec_heads = 2
    opts.rotary_pos_emb = False
    opts.post_emb_norm = False
    opts.log_model_structure = False
    opts.param_init = 0.0
    opts.param_init_glorot = True
    return opts


class TestDecoderOnlyModelBuild:
    def test_encoder_is_not_built(self):
        opts = _decoder_only_opts()
        tqm = _build_tqm(opts)
        model = build_model(opts, opts, VOCABS, task_queue_manager=tqm)
        assert model.encoder is None
        assert model.attention_bridge is None

    def test_decoder_blocks_have_no_cross_attn(self):
        opts = _decoder_only_opts()
        tqm = _build_tqm(opts)
        model = build_model(opts, opts, VOCABS, task_queue_manager=tqm)
        stack = model.decoder.get_attention_layers_by_xcoder_id(0, 'bar')
        assert len(stack.blocks) == 2
        for block in stack.blocks:
            assert not hasattr(block, 'cross_attn') or block.cross_attn is None

    def test_no_distributed_encoder_components_created(self):
        opts = _decoder_only_opts()
        tqm = _build_tqm(opts)
        comps = list(tqm.get_my_distributed_components())
        assert not any(isinstance(c, DistributedEncoderAttentionLayersBlock) for c in comps)

    def test_nmtmodel_forward_with_no_encoder_matches_decoder_wrapper_directly(self):
        opts = _decoder_only_opts()
        tqm = _build_tqm(opts)
        model = build_model(opts, opts, VOCABS, task_queue_manager=tqm)
        model.eval()

        metadata = DECODER_ONLY_TASK.get_serializable_metadata()
        ids = torch.randint(0, len(VOCABS[('tgt', 'b')]), (2, 5))

        with torch.no_grad():
            logits, hidden = model(None, ids, None, metadata=metadata)

            active_decoder = model.decoder.activate(task_id=DECODER_ONLY_TASK.corpus_id)
            expected_logits, expected_hidden = active_decoder(
                ids, context=None, return_logits_and_embeddings=True,
            )

        torch.testing.assert_close(logits, expected_logits)
        torch.testing.assert_close(hidden, expected_hidden)

    def test_dual_rope_decoder_does_not_build_dead_shared_rotary_emb(self):
        # Gemma3-style dual-theta rope: every DecoderBlock builds its own
        # per-layer RotaryEmbedding (see model_builder.py's `make_block`) and
        # always overrides whatever `rotary` tuple the wrapper computes (see
        # DecoderBlock.forward: `if self.rotary_emb is not None: ...`). The
        # generic per-component RotaryEmbedding that build_xcoder used to
        # build unconditionally whenever `rotary_pos_emb=True` was therefore
        # pure dead weight in this case -- never used in forward, but still
        # held as a submodule and written into the wrapper checkpoint file.
        opts = _decoder_only_opts()
        opts.rotary_pos_emb = True
        opts.dec_attn_dim_head = 8
        opts.dec_sliding_window = 4
        opts.dec_global_attn_every_n_layers = 2
        opts.dec_global_rope_theta = 1e6
        opts.dec_local_rope_theta = 1e4
        tqm = _build_tqm(opts)
        model = build_model(opts, opts, VOCABS, task_queue_manager=tqm)

        stack = model.decoder.get_attention_layers_by_xcoder_id(0, 'bar')
        for block in stack.blocks:
            assert block.rotary_emb is not None

        assert len(model.decoder.shared_rotary_embs) == 0
        assert model.decoder.get_pos_emb_by_component(('bar',)) is None

    def test_single_theta_rotary_still_builds_shared_rotary_emb(self):
        # Backward compatibility: when blocks do NOT own a per-layer rotary
        # (no dual global/local theta configured), the wrapper-level shared
        # rotary embedding is still the one actually used in forward, so it
        # must still be built.
        opts = _decoder_only_opts()
        opts.rotary_pos_emb = True
        opts.dec_attn_dim_head = 8
        tqm = _build_tqm(opts)
        model = build_model(opts, opts, VOCABS, task_queue_manager=tqm)

        stack = model.decoder.get_attention_layers_by_xcoder_id(0, 'bar')
        for block in stack.blocks:
            assert block.rotary_emb is None

        assert len(model.decoder.shared_rotary_embs) == 1
        assert model.decoder.get_pos_emb_by_component(('bar',)) is not None

    def test_regular_encoder_decoder_model_unaffected(self):
        # Backward compatibility: without decoder_only, the encoder must
        # still be built exactly as before.
        from mammoth.inputters.vocab import Vocab as _Vocab

        vocabs_dict = {
            ('src', 'a'): _Vocab(None, items=['a'], tag='dummy', specials=list(DEFAULT_SPECIALS)),
            ('tgt', 'b'): VOCABS[('tgt', 'b')],
        }
        task = TaskSpecs(
            node_rank=0, local_rank=0, src_lang='a', tgt_lang='b',
            encoder_id=['foo'], decoder_id=['bar'], corpus_id='a-b', weight=1,
            introduce_at_training_step=0, corpus_opts=dict(),
            src_vocab=vocabs_dict[('src', 'a')], tgt_vocab=vocabs_dict[('tgt', 'b')],
            encoder_adapter_ids=None, decoder_adapter_ids=None,
        )
        opts = Namespace()
        opts.seed = 1
        opts.enc_layers = [1]
        opts.dec_layers = [2]
        opts.enc_model_dim = 16
        opts.dec_model_dim = 16
        opts.enc_heads = 2
        opts.dec_heads = 2
        opts.rotary_pos_emb = False
        opts.post_emb_norm = False
        opts.log_model_structure = False
        opts.param_init = 0.0
        opts.param_init_glorot = True
        opts.ab_layers = []
        tqm = _build_tqm(opts, task=task)
        model = build_model(opts, opts, vocabs_dict, task_queue_manager=tqm)
        assert model.encoder is not None
        stack = model.decoder.get_attention_layers_by_xcoder_id(0, 'bar')
        for block in stack.blocks:
            assert block.cross_attn is not None
