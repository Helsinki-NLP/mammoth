"""
freeze_components / train_components: freeze whole DistributedComponents by name.

Scenario from the TODO: a model was trained on swe-eng and nob-eng (shared encoder 'nordic').
We add sme-eng and freeze everything that came from the checkpoint, so only the new sme
embedding learns.
"""
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from mammoth.distributed.components import (
    DistributedAdapter,
    DistributedAttentionBridge,
    DistributedEmbedding,
    DistributedEncoderAttentionLayersBlock,
    DistributedWrapperModules,
    Side,
)
from mammoth.distributed.contexts import WorldContext, DeviceContextEnum
from mammoth.distributed.tasks import TaskSpecs, TaskQueueManager, RoundRobinTaskDistributionStrategy
from mammoth.inputters.vocab import Vocab, DEFAULT_SPECIALS
import mammoth.model_builder as model_builder
from mammoth.model_builder import (
    build_xcoder,
    freeze_model_components,
    resolve_frozen_components,
    validate_optimizer_coverage,
)
from mammoth.models import NMTModel
from mammoth.utils.model_saver import load_parameters_from_checkpoint


# ---------------------------------------------------------------------------
# resolve_frozen_components: pure name selection
# ---------------------------------------------------------------------------

def _component(cls, **kwargs):
    return cls(global_ranks={0}, task_ids={'t'}, group=None, **kwargs)


COMPONENTS = [
    _component(DistributedEmbedding, side=Side.encoder, lang='swe'),
    _component(DistributedEmbedding, side=Side.encoder, lang='sme'),
    _component(DistributedEmbedding, side=Side.decoder, lang='eng'),
    _component(DistributedEncoderAttentionLayersBlock, layer_stack_index=0, xcoder_id='nordic'),
    _component(DistributedWrapperModules, side=Side.decoder, component_key=('eng',)),
    _component(DistributedAdapter, side=Side.encoder, layer_stack_index=0, adapter_group='g', sub_id='x'),
    _component(DistributedAttentionBridge),
]
ALL = {c.get_name() for c in COMPONENTS}
ADAPTER = 'encoder_adapter_0_g_x'
TQM = SimpleNamespace(distributed_components=COMPONENTS)


def _resolve(prefix=None, **opts):
    return resolve_frozen_components(Namespace(**opts), TQM, prefix)


def test_no_selector_freezes_nothing():
    assert _resolve() == set()
    assert _resolve(freeze_components=[], train_components=[]) == set()


def test_freeze_patterns_are_wildcards():
    assert _resolve(freeze_components=['encoder_*', 'tgt_embeddings_eng']) == {'encoder_0_nordic', 'tgt_embeddings_eng'}


def test_train_components_freezes_the_rest_except_adapters():
    assert _resolve(train_components=['*_sme']) == ALL - {'src_embeddings_sme', ADAPTER}


def test_adapters_frozen_only_when_pattern_names_them():
    assert ADAPTER not in _resolve(freeze_components=['*'])
    assert _resolve(freeze_components=['*adapter*']) == {ADAPTER}


def test_pattern_matching_only_adapters_is_an_error():
    with pytest.raises(ValueError, match='only adapters'):
        _resolve(freeze_components=['*_g_x'])


def test_unmatched_pattern_is_an_error():
    with pytest.raises(ValueError, match="'encoder_0_nordc' matches no component"):
        _resolve(freeze_components=['encoder_0_nordic', 'encoder_0_nordc'])


def test_freeze_and_train_are_mutually_exclusive():
    with pytest.raises(ValueError, match='mutually exclusive'):
        _resolve(freeze_components=['*'], train_components=['*_sme'])


def test_pretrained_requires_train_from():
    with pytest.raises(ValueError, match='requires train_from'):
        _resolve(freeze_components=['@pretrained'])


def test_pretrained_selects_components_with_a_shard(tmp_path):
    prefix = str(tmp_path / 'model_step_10')
    for name in ['src_embeddings_swe', 'encoder_0_nordic', ADAPTER]:
        torch.save({}, f'{prefix}_{name}.pt')
    torch.save({}, f'{prefix}_encoder_0_nordic_optim.pt')
    # Pretrained adapters stay trainable: '@pretrained' does not name them
    assert _resolve(prefix, freeze_components=['@pretrained']) == {'src_embeddings_swe', 'encoder_0_nordic'}
    assert _resolve(prefix, train_components=['@pretrained']) == ALL - {'src_embeddings_swe', 'encoder_0_nordic', ADAPTER}


def test_pretrained_counts_legacy_wrapper_files(tmp_path):
    """Old checkpoints store wrapper modules in per-task files, which loading migrates."""
    prefix = str(tmp_path / 'model_step_10')
    torch.save({}, f'{prefix}_decoder_task_swe-eng.pt')
    assert _resolve(prefix, freeze_components=['@pretrained']) == {'decoder_wrapper_eng'}


def test_pretrained_without_shards_is_an_error(tmp_path):
    with pytest.raises(ValueError, match='matches no component'):
        _resolve(str(tmp_path / 'model_step_10'), freeze_components=['@pretrained'])


# ---------------------------------------------------------------------------
# End to end: extend a trained model with sme, train, check what moved
# ---------------------------------------------------------------------------

def _vocab(n):
    return Vocab(None, items=[f'_{i}' for i in range(n)], tag='dummy', specials=list(DEFAULT_SPECIALS))


VOCABS = {
    ('src', 'swe'): _vocab(10),
    ('src', 'nob'): _vocab(12),
    ('src', 'sme'): _vocab(14),
    ('tgt', 'eng'): _vocab(16),
}


def _task(src):
    return TaskSpecs(
        node_rank=0, local_rank=0, src_lang=src, tgt_lang='eng',
        encoder_id=['nordic'], decoder_id=['eng'], corpus_id=f'{src}-eng',
        weight=1, introduce_at_training_step=0, corpus_opts=dict(),
        src_vocab=VOCABS[('src', src)], tgt_vocab=VOCABS[('tgt', 'eng')],
        encoder_adapter_ids=None, decoder_adapter_ids=None,
    )


def _build(src_langs, **freeze_opts):
    opts = Namespace(
        seed=1, enc_layers=[1], dec_layers=[1], model_dim=16, heads=2,
        rotary_pos_emb=False, post_emb_norm=True, **freeze_opts,
    )
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
    return model, opts, tqm


def _state(model, tqm):
    return {
        c.get_name(): {k: v.clone() for k, v in c.state_dict(model).items()}
        for c in tqm.get_my_distributed_components()
    }


@pytest.mark.parametrize('freeze_opts', [
    {'freeze_components': ['@pretrained']},
    {'train_components': ['*_sme']},
])
def test_extending_with_sme_trains_only_the_new_components(tmp_path, freeze_opts):
    prefix = str(tmp_path / 'model_step_100')
    torch.manual_seed(0)
    pretrained, _, pretrained_tqm = _build(['swe', 'nob'])
    for c in pretrained_tqm.get_my_distributed_components():
        torch.save(c.state_dict(pretrained), f'{prefix}_{c.get_name()}.pt')

    torch.manual_seed(1)
    model, opts, tqm = _build(['swe', 'nob', 'sme'], **freeze_opts)
    load_parameters_from_checkpoint(
        f'{prefix}_frame.pt', model, optim=None, task_queue_manager=tqm,
        reset_optim=True, allow_partial_checkpoint=True,  # sme is new
    )
    freeze_model_components(model, opts, tqm, frame_checkpoint_path=f'{prefix}_frame.pt')

    # Weight decay would move frozen weights if the optimizer saw them
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2, weight_decay=0.1)
    validate_optimizer_coverage(model, optimizer)
    before = _state(model, tqm)

    metadata = SimpleNamespace(corpus_id='sme-eng', encoder_adapter_ids=None, decoder_adapter_ids=None)
    for _ in range(5):
        src = torch.randint(4, 14, (2, 6))
        tgt = torch.randint(4, 16, (2, 5))
        logits, _ = model(src=src, decoder_input=tgt[:, :-1], src_mask=torch.ones_like(src, dtype=torch.bool),
                          metadata=metadata)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), tgt[:, 1:].flatten())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    after = _state(model, tqm)
    for name in before:
        changed = any(not torch.equal(before[name][k], after[name][k]) for k in before[name])
        assert changed == (name == 'src_embeddings_sme'), name


def test_rank_with_nothing_to_train_is_an_error():
    """Without this check the rank crashes later in loss.backward() with 'does not require grad'."""
    model, opts, tqm = _build(['swe', 'nob'], freeze_components=['*'])
    with pytest.raises(ValueError, match=r"Every parameter on this GPU is frozen.*\['swe-eng', 'nob-eng'\]"):
        freeze_model_components(model, opts, tqm)


def test_summary_marks_components_without_parameters_as_empty(monkeypatch):
    messages = []
    monkeypatch.setattr(model_builder.logger, 'info', messages.append)
    model, opts, tqm = _build(['swe'], freeze_components=['tgt_embeddings_eng'])
    freeze_model_components(model, opts, tqm)
    status = {line.split()[1]: line.split()[0] for m in messages if 'freezing status' in m for line in m.splitlines()[1:]}
    assert status['tgt_embeddings_eng'] == 'frozen'
    assert status['encoder_0_nordic'] == 'trainable'
    empty = {name for name, s in status.items() if s == 'empty'}
    assert empty and all(sum(p.numel() for _, p in c.named_parameters(model)) == 0
                         for c in tqm.get_my_distributed_components() if c.get_name() in empty)
