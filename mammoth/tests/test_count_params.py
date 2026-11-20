"""count_params.py: both views come from one table and add up consistently.

Uses a synthetic checkpoint directory (tiny shards + a frame with task opts), so no GPU or real model is needed.
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mammoth.scripts import count_params

SCRIPT = Path(count_params.__file__)

# eng-spa and eng-fra share the encoder stack and the src embeddings; decoders are per target language.
TASKS = {
    'eng-spa': {'src_tgt': 'eng-spa', 'enc_sharing_group': ['all'], 'dec_sharing_group': ['spa']},
    'eng-fra': {'src_tgt': 'eng-fra', 'enc_sharing_group': ['all'], 'dec_sharing_group': ['fra']},
}
SHARDS = {
    'src_embeddings_eng': 100,
    'tgt_embeddings_spa': 80,
    'tgt_embeddings_fra': 90,
    'encoder_0_all': 1000,
    'decoder_0_spa': 500,
    'decoder_0_fra': 600,
    'encoder_wrapper_all': 7,
    'decoder_wrapper_spa': 11,
    'decoder_wrapper_fra': 13,
    'encoder_adapter_0_g_x': 5,  # used by no task
}


@pytest.fixture
def ckpt(tmp_path):
    torch.save({'opts': SimpleNamespace(tasks=TASKS)}, tmp_path / 'model_step_10_frame.pt')
    for name, n in SHARDS.items():
        torch.save({'w': torch.zeros(n), 'rotary.inv_freq': torch.zeros(3)}, tmp_path / f'model_step_10_{name}.pt')
    torch.save({'w': torch.zeros(999)}, tmp_path / 'model_step_10_encoder_0_all_optim.pt')  # ignored
    return tmp_path


def test_components_add_up_to_total(ckpt):
    data = count_params.collect(str(ckpt))
    assert {n: c['params'] for n, c in data['components'].items()} == SHARDS
    assert data['total'] == sum(SHARDS.values())
    assert data['total_buffers'] == 3 * len(SHARDS)


def test_task_full_is_sum_of_its_components(ckpt):
    data = count_params.collect(str(ckpt))
    for row in data['tasks']:
        assert row['missing_shards'] == []
        assert row['full'] == sum(SHARDS[n] for n in row['components'])
        assert row['full'] == row['exclusive'] + row['shared']
    spa = next(r for r in data['tasks'] if r['task'] == 'eng-spa')
    assert spa['shared'] == SHARDS['src_embeddings_eng'] + SHARDS['encoder_0_all'] + SHARDS['encoder_wrapper_all']


def test_component_users_match_sharing_groups(ckpt):
    comps = count_params.collect(str(ckpt))['components']
    assert sorted(comps['encoder_0_all']['tasks']) == ['eng-fra', 'eng-spa']
    assert comps['decoder_0_spa']['tasks'] == ['eng-spa']
    assert comps['encoder_adapter_0_g_x']['tasks'] == []


@pytest.mark.parametrize('extra', [[], ['--by', 'component'], ['--by', 'component', '-v'],
                                   ['--by', 'component', '--task', 'eng-spa'], ['--task', 'eng-fra']])
def test_cli_runs(ckpt, tmp_path, extra):
    out_json = tmp_path / 'out.json'
    res = subprocess.run([sys.executable, str(SCRIPT), '--checkpoint-dir', str(ckpt), '--json', str(out_json), *extra],
                         capture_output=True, text=True, check=True)
    assert f"{sum(SHARDS.values()):,}" in res.stdout
    dumped = json.loads(out_json.read_text())
    assert set(dumped) >= {'tasks', 'components', 'total'}


def test_cli_unknown_task(ckpt):
    res = subprocess.run([sys.executable, str(SCRIPT), '--checkpoint-dir', str(ckpt), '--task', 'nope'],
                         capture_output=True, text=True)
    assert res.returncode != 0 and 'Unknown task' in res.stderr
