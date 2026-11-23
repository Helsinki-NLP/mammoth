"""A run started from train_from saves the whole model, not only the components it trained.

Scenario: a model trained on eng-swe is extended with deu-swe. The new run's model has no eng
components, but its checkpoints must still contain them, and its frame must list both tasks.
"""
import os
from argparse import Namespace

import torch

from mammoth.utils.model_saver import carry_over_shards, merge_frame_with_train_from


def _write(prefix, keys):
    for key in keys:
        torch.save({'w': torch.tensor([float(len(key))])}, f'{prefix}_{key}.pt')


def _keys(prefix):
    return sorted(
        os.path.basename(p)[len(os.path.basename(prefix)) + 1:-len('.pt')]
        for p in __import__('glob').glob(f'{prefix}_*.pt')
    )


def test_carries_only_components_missing_from_this_run(tmp_path):
    old, new = str(tmp_path / 'old' / '_best'), str(tmp_path / 'new' / '_step_3500')
    os.makedirs(tmp_path / 'old')
    os.makedirs(tmp_path / 'new')
    _write(old, ['frame', 'encoder_0_eng', 'encoder_0_eng_optim', 'src_embeddings_eng', 'decoder_0_swe'])
    _write(new, ['frame', 'decoder_0_swe', 'encoder_0_deu'])
    fresh_swe = torch.load(f'{new}_decoder_0_swe.pt')

    carried = carry_over_shards(old, new, current_names={'decoder_0_swe', 'encoder_0_deu'})

    assert sorted(os.path.basename(p) for p in carried) == [
        '_step_3500_encoder_0_eng.pt', '_step_3500_encoder_0_eng_optim.pt', '_step_3500_src_embeddings_eng.pt',
    ]
    assert _keys(new) == [
        'decoder_0_swe', 'encoder_0_deu', 'encoder_0_eng', 'encoder_0_eng_optim', 'frame', 'src_embeddings_eng',
    ]
    # This run's own shards and frame are never replaced by the old ones
    assert torch.equal(torch.load(f'{new}_decoder_0_swe.pt')['w'], fresh_swe['w'])
    assert torch.equal(torch.load(f'{new}_encoder_0_eng.pt')['w'], torch.load(f'{old}_encoder_0_eng.pt')['w'])


def test_step_prefix_does_not_match_a_longer_step(tmp_path):
    _write(str(tmp_path / '_step_300'), ['encoder_0_eng'])
    _write(str(tmp_path / '_step_3000'), ['encoder_0_fin'])
    carried = carry_over_shards(str(tmp_path / '_step_300'), str(tmp_path / 'new'), current_names=set())
    assert [os.path.basename(p) for p in carried] == ['new_encoder_0_eng.pt']


def test_carried_files_survive_deleting_the_old_checkpoint(tmp_path):
    old, new = str(tmp_path / 'old'), str(tmp_path / 'new')
    _write(old, ['encoder_0_eng'])
    carry_over_shards(old, new, current_names=set())
    os.remove(f'{old}_encoder_0_eng.pt')
    assert torch.load(f'{new}_encoder_0_eng.pt')['w'].item() == len('encoder_0_eng')


def test_frame_lists_old_and_new_tasks_and_vocabs():
    model_opts = Namespace(model_dim=512, tasks={'eng-swe': {'src_tgt': 'eng-swe', 'node_gpu': '0:1'}})
    old_frame = {
        'opts': model_opts,
        'vocab': {('src', 'eng'): 'old-eng', ('tgt', 'swe'): 'old-swe'},
    }
    current_tasks = {'deu-swe': {'src_tgt': 'deu-swe'}, 'eng-swe': {'src_tgt': 'eng-swe', 'node_gpu': '0:0'}}
    vocabs = {('src', 'deu'): 'deu', ('tgt', 'swe'): 'swe'}

    opts, vocab = merge_frame_with_train_from(model_opts, vocabs, current_tasks, old_frame)

    assert opts.tasks == current_tasks  # this run wins on equal task ids
    assert opts.model_dim == 512
    assert vocab == {('src', 'eng'): 'old-eng', ('src', 'deu'): 'deu', ('tgt', 'swe'): 'swe'}
    # The model_opts used to build the model are not modified
    assert model_opts.tasks == {'eng-swe': {'src_tgt': 'eng-swe', 'node_gpu': '0:1'}}


def test_frame_without_train_from_lists_this_runs_tasks():
    model_opts = Namespace(tasks={'a-b': {}})
    opts, vocab = merge_frame_with_train_from(model_opts, {('src', 'a'): 'a'}, {'a-b': {}}, None)
    assert opts.tasks == {'a-b': {}}
    assert vocab == {('src', 'a'): 'a'}
