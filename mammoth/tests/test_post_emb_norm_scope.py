"""
Tests for the post-embedding norm scope. It is not a user option: it is
recorded in the checkpoint frame (opts.post_emb_norm_scope).

- 'language' (every new model): one norm per language and side, stored with
  that language's embedding in {src,tgt}_embeddings_*.pt.
- 'component' (checkpoints that predate per-language norms, frame has no key):
  one norm per xcoder path in {encoder,decoder}_wrapper_*.pt. Such checkpoints
  are loaded as they are (for inference); scopes are never converted.
"""
import os
from argparse import Namespace

import pytest
import torch

from mammoth.distributed.components import Side, DistributedEmbedding, DistributedWrapperModules, _EmbeddingWithNorm
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum
from mammoth.distributed.tasks import TaskSpecs, TaskQueueManager, RoundRobinTaskDistributionStrategy
from mammoth.hf_integration.to_hf.mammoth_hub import component_to_hf
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
from mammoth.model_builder import build_xcoder, _freeze_embeddings
from mammoth.models import NMTModel
from mammoth.utils.model_saver import load_parameters_from_checkpoint


def _vocab(n):
    return Vocab(None, items=[f'_{i}' for i in range(n)], tag='dummy', specials=list(DEFAULT_SPECIALS))


# Different vocab sizes per language, so a mixed-up embedding would not even load.
VOCABS = {
    ('src', 'swe'): _vocab(10),
    ('src', 'nob'): _vocab(12),
    ('src', 'dan'): _vocab(14),
    ('tgt', 'eng'): _vocab(16),
}


def _task(src):
    return TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang=src,
        tgt_lang='eng',
        encoder_id=['nordic'],
        decoder_id=['eng'],
        corpus_id=f'{src}-eng',
        weight=1,
        introduce_at_training_step=0,
        corpus_opts=dict(),
        src_vocab=VOCABS[('src', src)],
        tgt_vocab=VOCABS[('tgt', 'eng')],
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )


def _opts(scope=None):
    opts = Namespace()
    opts.seed = 1
    opts.enc_layers = [1]
    opts.dec_layers = [1]
    opts.model_dim = 16
    opts.heads = 2
    opts.rotary_pos_emb = False
    opts.post_emb_norm = True
    if scope is not None:
        opts.post_emb_norm_scope = scope
    return opts


def _build(opts, src_langs):
    world_context = WorldContext(DeviceContextEnum.MULTI_GPU, n_nodes=1, gpus_per_node=1)
    tqm = TaskQueueManager(
        tasks=[_task(src) for src in src_langs],
        accum_count=1,
        world_context=world_context,
        task_distribution_strategy_cls=RoundRobinTaskDistributionStrategy,
        uses_adapters=False,
    ).global_to_local(node_rank=0, local_rank=0, opts=opts)
    tqm.create_all_distributed_components(use_attention_bridge=False)
    kwargs = dict(model_opts=opts, vocabs_dict=VOCABS, device='cpu', task_queue_manager=tqm)
    model = NMTModel(
        encoder=build_xcoder(side=Side.encoder, **kwargs),
        decoder=build_xcoder(side=Side.decoder, **kwargs),
        attention_bridge=None,
    )
    # Non-trivial norms, so a wrong or unloaded norm changes the outputs.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'post_emb_norm' in name:
                p.uniform_(0.5, 1.5)
    return model.eval(), tqm


def _save(model, tqm, prefix):
    """What ModelSaver writes: one file per distributed component."""
    for component in tqm.get_my_distributed_components():
        torch.save(component.state_dict(model), f'{prefix}_{component.get_name()}.pt')


def _encode(model, src):
    ids = torch.arange(2, 8).unsqueeze(0)
    with torch.no_grad():
        return model.encoder.activate(f'{src}-eng')(ids)


def _component(tqm, cls, name):
    return next(c for c in tqm.get_my_distributed_components() if isinstance(c, cls) and c.get_name() == name)


class TestScopeFromFrame:
    def test_new_model_is_language_scope(self):
        model, _ = _build(_opts(), ['swe', 'nob'])
        assert set(model.encoder.shared_lang_post_emb_norms) == {'swe', 'nob'}

    def test_fresh_training_records_language_scope_in_the_frame(self):
        from mammoth.train_single import _get_model_opts
        assert _get_model_opts(Namespace()).post_emb_norm_scope == 'language'

    def test_frame_without_the_key_is_component_scope(self):
        from mammoth.utils.parse import ArgumentParser
        old_frame_opts = Namespace(model_dim=16, heads=2, post_emb_norm=True)
        assert ArgumentParser.checkpoint_model_opts(old_frame_opts).post_emb_norm_scope == 'component'

    def test_train_from_keeps_the_frame_scope(self):
        from mammoth.train_single import _get_model_opts
        for frame_scope, expected in ((None, 'component'), ('language', 'language')):
            frame_opts = Namespace(model_dim=16, heads=2, post_emb_norm=True)
            if frame_scope is not None:
                frame_opts.post_emb_norm_scope = frame_scope
            model_opts = _get_model_opts(Namespace(), frame_checkpoint={'opts': frame_opts})
            assert model_opts.post_emb_norm_scope == expected


class TestComponentScope:
    """Checkpoints that predate per-language norms load and run unchanged."""

    def test_one_norm_shared_by_the_path(self):
        model, tqm = _build(_opts('component'), ['swe', 'nob'])
        assert len(model.encoder.shared_lang_post_emb_norms) == 0
        norm = model.encoder.get_post_emb_norm_by_component(('nordic',))
        assert model.encoder.task_wrappers['swe-eng']._post_emb_norm is norm
        assert model.encoder.task_wrappers['nob-eng']._post_emb_norm is norm

    def test_files_unchanged(self):
        model, tqm = _build(_opts('component'), ['swe', 'nob'])
        emb = _component(tqm, DistributedEmbedding, 'src_embeddings_swe')
        assert set(emb.state_dict(model)) == {'weight'}
        wrapper = _component(tqm, DistributedWrapperModules, 'encoder_wrapper_nordic')
        assert set(wrapper.state_dict(model)) == {'post_emb_norm.weight'}

    def test_component_scope_round_trip_is_exact(self, tmp_path):
        model, tqm = _build(_opts('component'), ['swe', 'nob'])
        prefix = os.path.join(tmp_path, 'model_best')
        _save(model, tqm, prefix)
        torch.manual_seed(123)
        reloaded, reloaded_tqm = _build(_opts('component'), ['swe', 'nob'])
        load_parameters_from_checkpoint(f'{prefix}_frame.pt', reloaded, None, reloaded_tqm, reset_optim=True)
        for src in ('swe', 'nob'):
            assert torch.equal(_encode(model, src), _encode(reloaded, src))


class TestLanguageScope:
    def test_each_language_owns_its_norm(self):
        model, tqm = _build(_opts('language'), ['swe', 'nob'])
        swe = model.encoder.get_post_emb_norm_by_lang('swe')
        nob = model.encoder.get_post_emb_norm_by_lang('nob')
        assert swe is not nob
        assert model.encoder.task_wrappers['swe-eng']._post_emb_norm is swe
        assert model.encoder.task_wrappers['nob-eng']._post_emb_norm is nob
        assert model.decoder.get_post_emb_norm_by_lang('eng') is not None
        assert isinstance(model.encoder.get_post_emb_norm_by_component(('nordic',)), torch.nn.Identity)

    def test_norm_is_saved_and_trained_with_the_embedding(self):
        model, tqm = _build(_opts('language'), ['swe', 'nob'])
        emb = _component(tqm, DistributedEmbedding, 'src_embeddings_swe')
        assert set(emb.state_dict(model)) == {'weight', 'post_emb_norm.weight'}
        params = dict(emb.named_parameters(model))
        assert params['weight'] is model.encoder.get_embedding_by_lang('swe').weight
        assert params['post_emb_norm.weight'] is model.encoder.get_post_emb_norm_by_lang('swe').weight
        wrapper = _component(tqm, DistributedWrapperModules, 'encoder_wrapper_nordic')
        assert wrapper.state_dict(model) == {}

    def test_language_scope_round_trip_is_exact(self, tmp_path):
        model, tqm = _build(_opts('language'), ['swe', 'nob'])
        prefix = os.path.join(tmp_path, 'model_best')
        _save(model, tqm, prefix)
        torch.manual_seed(123)
        reloaded, reloaded_tqm = _build(_opts('language'), ['swe', 'nob'])
        load_parameters_from_checkpoint(f'{prefix}_frame.pt', reloaded, None, reloaded_tqm, reset_optim=True)
        for src in ('swe', 'nob'):
            assert torch.equal(_encode(model, src), _encode(reloaded, src))

    def test_freeze_embeddings_freezes_language_norms(self):
        model, _ = _build(_opts('language'), ['swe', 'nob'])
        _freeze_embeddings(model.encoder)
        for lang in ('swe', 'nob'):
            assert not model.encoder.get_post_emb_norm_by_lang(lang).weight.requires_grad

    def test_adding_a_language_leaves_the_others_untouched(self, tmp_path):
        # Danish next to an already trained Swedish/Norwegian path: only Danish's
        # embedding file (with its own norm) is missing; nothing shared changes.
        model, tqm = _build(_opts('language'), ['swe', 'nob'])
        prefix = os.path.join(tmp_path, 'model_best')
        _save(model, tqm, prefix)
        extended, extended_tqm = _build(_opts('language'), ['swe', 'nob', 'dan'])
        load_parameters_from_checkpoint(
            f'{prefix}_frame.pt', extended, None, extended_tqm,
            reset_optim=True, yes_i_messed_with_the_checkpoint=True,
        )
        for src in ('swe', 'nob'):
            assert torch.equal(_encode(model, src), _encode(extended, src))
        dan = _component(extended_tqm, DistributedEmbedding, 'src_embeddings_dan')
        assert set(dan.state_dict(extended)) == {'weight', 'post_emb_norm.weight'}


class TestHFMapping:
    def test_embedding_component_with_norm(self):
        weight, norm = torch.randn(10, 16), torch.randn(16)
        mapped = component_to_hf('src_emb', {'weight': weight, 'post_emb_norm.weight': norm})
        assert set(mapped) == {'encoder.token_emb.weight', 'encoder.post_emb_norm.weight'}
        assert mapped['encoder.post_emb_norm.weight'] is norm

    def test_embedding_component_without_norm_is_unchanged(self):
        mapped = component_to_hf('tgt_emb', {'weight': torch.randn(10, 16)})
        assert set(mapped) == {'decoder.token_emb.weight'}


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    """A language-scope sharded checkpoint + its original component modules."""
    from mammoth.tests.test_hf_modular import _Vocab, _opts, _randomize, PREFIX, TASKS, VOCAB_SIZES
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import config_from_opts
    from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration

    torch.manual_seed(0)
    ckpt = tmp_path_factory.mktemp('ckpt_lang')
    opts = _opts()
    opts.post_emb_norm_scope = 'language'
    vocab = {key: _Vocab(size, key[1]) for key, size in VOCAB_SIZES.items()}
    torch.save({'opts': opts, 'vocab': vocab}, ckpt / f'{PREFIX}_frame.pt')

    modules = {}  # shard name -> module

    def save(name, module):
        if name not in modules:
            _randomize(module)
            modules[name] = module
            torch.save(module.state_dict(), ckpt / f'{PREFIX}_{name}.pt')

    for name, (enc, dec) in TASKS.items():
        src, tgt = name.split('-')
        m = MammothForConditionalGeneration(
            config_from_opts(opts, VOCAB_SIZES[('src', src)], VOCAB_SIZES[('tgt', tgt)]))
        # Same keys DistributedEmbedding writes in language scope: weight + post_emb_norm.*
        save(f'src_embeddings_{src}', _EmbeddingWithNorm(m.encoder.token_emb, m.encoder.post_emb_norm))
        save(f'tgt_embeddings_{tgt}', _EmbeddingWithNorm(m.decoder.token_emb, m.decoder.post_emb_norm))
        for i, x in enumerate(enc):
            save(f'encoder_{i}_{x}', m.encoder.stacks[i])
        for i, x in enumerate(dec):
            save(f'decoder_{i}_{x}', m.decoder.stacks[i])
        save(f"encoder_wrapper_{'_'.join(enc)}", torch.nn.ModuleDict())
        save(f"decoder_wrapper_{'_'.join(dec)}", torch.nn.ModuleDict({'to_logits': m.decoder.to_logits}))
    return ckpt, opts, modules


class TestHFExportLanguageScope:
    """HF export of a language-scope checkpoint: the norm comes from the embedding shard."""

    def _reference(self, opts, modules, src, tgt, enc, dec):
        from mammoth.tests.test_hf_modular import VOCAB_SIZES
        from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import config_from_opts
        from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration
        m = MammothForConditionalGeneration(
            config_from_opts(opts, VOCAB_SIZES[('src', src)], VOCAB_SIZES[('tgt', tgt)]))
        _EmbeddingWithNorm(m.encoder.token_emb, m.encoder.post_emb_norm).load_state_dict(
            modules[f'src_embeddings_{src}'].state_dict())
        _EmbeddingWithNorm(m.decoder.token_emb, m.decoder.post_emb_norm).load_state_dict(
            modules[f'tgt_embeddings_{tgt}'].state_dict())
        for i, x in enumerate(enc):
            m.encoder.stacks[i].load_state_dict(modules[f'encoder_{i}_{x}'].state_dict())
        for i, x in enumerate(dec):
            m.decoder.stacks[i].load_state_dict(modules[f'decoder_{i}_{x}'].state_dict())
        m.decoder.to_logits.load_state_dict(modules[f"decoder_wrapper_{'_'.join(dec)}"]['to_logits'].state_dict())
        return m.eval()

    def test_modular_export_exact_match_including_zero_shot(self, checkpoint, tmp_path):
        from mammoth.tests.test_hf_modular import _logits
        from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import convert_modular
        from mammoth.hf_integration.to_hf.mammoth_hub import MammothHub
        ckpt, opts, modules = checkpoint
        convert_modular(str(ckpt), str(tmp_path))
        for src, tgt, enc, dec in [
            ('eng', 'spa', ['eng', 'all'], ['spa', 'all']),
            ('deu', 'fra', ['deu', 'all'], ['fra', 'all']),  # zero-shot combination
        ]:
            model = MammothHub.from_pretrained(str(tmp_path), src=src, tgt=tgt)
            ref = self._reference(opts, modules, src, tgt, enc, dec)
            assert torch.equal(_logits(model, src, tgt), _logits(ref, src, tgt))

    def test_flat_export_takes_norm_from_embedding(self, checkpoint):
        from mammoth.tests.test_hf_modular import PREFIX
        from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import assemble_state_dict
        ckpt, _, modules = checkpoint
        sd = assemble_state_dict(str(ckpt), PREFIX, 'deu', 'spa', ['deu', 'all'], ['spa', 'all'])
        assert torch.equal(sd['encoder.post_emb_norm.weight'], modules['src_embeddings_deu'].post_emb_norm.weight)
        assert torch.equal(sd['decoder.post_emb_norm.weight'], modules['tgt_embeddings_spa'].post_emb_norm.weight)
