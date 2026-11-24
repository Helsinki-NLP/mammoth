"""mammoth_translate with several task ids: per-task src/tgt/output in the inference config."""
from argparse import Namespace

import pytest

from mammoth.bin.translate import _get_parser, resolve_task_runs


def _opts(task_id, tasks, src=None, tgt=None, output='pred.txt'):
    return Namespace(task_id=task_id, tasks=tasks, src=src, tgt=tgt, output=output)


def test_single_task_top_level_io_unchanged():
    # The pre-existing format: one task, src/output at the top level.
    runs = resolve_task_runs(_opts(['eng-spa'], {'eng-spa': {'src_tgt': 'eng-spa'}}, src='in.eng', output='out.spa'))
    assert [(r.task_id, r.src, r.tgt, r.output) for r in runs] == [('eng-spa', 'in.eng', None, 'out.spa')]


def test_scalar_task_id_is_accepted():
    runs = resolve_task_runs(_opts('eng-spa', {'eng-spa': {}}, src='in.eng'))
    assert runs[0].task_id == 'eng-spa'


def test_per_task_io_overrides_top_level():
    tasks = {
        'eng-spa': {'src': 'a.eng', 'tgt': 'a.spa', 'output': 'out/eng-spa'},
        'eng-fra': {'output': 'out/eng-fra'},  # src falls back to the top level
    }
    opts = _opts(['eng-spa', 'eng-fra'], tasks, src='shared.eng')
    runs = resolve_task_runs(opts)
    assert [(r.task_id, r.src, r.tgt, r.output) for r in runs] == [
        ('eng-spa', 'a.eng', 'a.spa', 'out/eng-spa'),
        ('eng-fra', 'shared.eng', None, 'out/eng-fra'),
    ]
    # Runs are copies: the parsed opts are left alone.
    assert opts.task_id == ['eng-spa', 'eng-fra'] and opts.src == 'shared.eng'


def test_runs_follow_task_id_order():
    tasks = {'a-b': {'src': 'x', 'output': 'o1'}, 'c-d': {'src': 'y', 'output': 'o2'}}
    assert [r.task_id for r in resolve_task_runs(_opts(['c-d', 'a-b'], tasks))] == ['c-d', 'a-b']


@pytest.mark.parametrize('task_id, tasks, src, match', [
    (['eng-spa', 'xxx-yyy'], {'eng-spa': {}}, 'in', 'Unknown task_id'),
    (['eng-spa', 'eng-spa'], {'eng-spa': {}}, 'in', 'more than once'),
    (['eng-spa'], {'eng-spa': {}}, None, 'no source file'),
    (['eng-spa', 'eng-fra'], {'eng-spa': {}, 'eng-fra': {}}, 'in', 'Several tasks would write'),
])
def test_invalid_selection(task_id, tasks, src, match):
    with pytest.raises(ValueError, match=match):
        resolve_task_runs(_opts(task_id, tasks, src=src))


@pytest.mark.parametrize('yaml_task_id, expected', [
    ('eng-spa', ['eng-spa']),
    ('[eng-spa, eng-fra]', ['eng-spa', 'eng-fra']),
])
def test_config_task_id_scalar_or_list(tmp_path, yaml_task_id, expected):
    config = tmp_path / 'infer.yaml'
    config.write_text(
        f'task_id: {yaml_task_id}\n'
        'tasks:\n'
        '  eng-spa: {src_tgt: eng-spa, src: a.eng, output: a.spa}\n'
        '  eng-fra: {src_tgt: eng-fra, src: a.eng, output: a.fra}\n'
        'src_vocab: {eng: tok.json}\n'
        'model: model/\n'
        'seed: 1\n'
        'max_length: 64\n'
    )
    opts = _get_parser().parse_args(['-config', str(config)])
    assert opts.task_id == expected


def test_command_line_task_ids(tmp_path):
    config = tmp_path / 'infer.yaml'
    config.write_text(
        'tasks: {eng-spa: {src_tgt: eng-spa}, eng-fra: {src_tgt: eng-fra}}\n'
        'src_vocab: {eng: tok.json}\nmodel: model/\nseed: 1\nmax_length: 64\n'
    )
    opts = _get_parser().parse_args(['-config', str(config), '--task_id', 'eng-spa', 'eng-fra'])
    assert opts.task_id == ['eng-spa', 'eng-fra']
