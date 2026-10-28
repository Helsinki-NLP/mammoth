"""--metric_for_best_model: mean over all tasks on all devices (mean_bleu / mean_chrf / mean_ppl)."""
import math
from argparse import Namespace

import pytest
import torch

from mammoth.utils.earlystopping import EarlyStopping
from mammoth.utils.model_saver import ModelSaver
from mammoth.utils.statistics import Statistics


class _Ctx:
    def __init__(self, world_size=1, master=True):
        self.world_size = world_size
        self._master = master

    def is_distributed(self):
        return self.world_size > 1

    def is_master(self):
        return self._master


def _saver(tmp_path, metric='mean_bleu', save_strategy='steps', keep_checkpoint=-1, use_opts=True):
    run_opts = Namespace(save_strategy=save_strategy, metric_for_best_model=metric)
    return ModelSaver(
        str(tmp_path / 'model'),
        model=None,
        model_opts=run_opts if not use_opts else Namespace(),
        vocabs_dict={},
        optim=None,
        keep_checkpoint=keep_checkpoint,
        task_queue_manager=object(),
        opts=run_opts if use_opts else None,
    )


def _stats(loss=0.0, n_words=0, **metrics):
    s = Statistics(loss=loss, n_words=n_words)
    s.validation_metrics = dict(metrics)
    return s


def test_default_is_mean_ppl(tmp_path):
    saver = ModelSaver(
        str(tmp_path / 'model'), None, Namespace(), {}, None, task_queue_manager=object(), opts=Namespace()
    )
    assert saver.metric_for_best_model == 'mean_ppl'
    assert saver.greater_is_better is False


@pytest.mark.parametrize('metric,greater', [('mean_bleu', True), ('mean_chrf', True), ('mean_ppl', False)])
def test_direction_follows_metric(tmp_path, metric, greater):
    assert _saver(tmp_path, metric).greater_is_better is greater


def test_unsupported_metric_rejected_for_run_opts(tmp_path):
    with pytest.raises(ValueError):
        _saver(tmp_path, 'bleu')


def test_legacy_metric_in_model_opts_falls_back(tmp_path):
    # Conversion scripts pass only model_opts (an old frame may say 'bleu').
    saver = _saver(tmp_path, 'bleu', use_opts=False)
    assert saver.metric_for_best_model == 'mean_ppl'


def test_mean_bleu_uses_per_task_scores(tmp_path):
    # Two tasks on one device count twice; the per-device 'bleu' key is ignored.
    saver = _saver(tmp_path, 'mean_bleu')
    stats = _stats(**{'bleu/a': 10.0, 'bleu/b': 30.0, 'bleu': 20.0, 'chrf/a': 50.0})
    assert saver._mean_metric_across_devices(stats, _Ctx()) == pytest.approx(20.0)
    assert _saver(tmp_path, 'mean_chrf')._mean_metric_across_devices(stats, _Ctx()) == pytest.approx(50.0)


def test_mean_ppl_uses_device_ppl(tmp_path):
    saver = _saver(tmp_path, 'mean_ppl')
    assert saver._mean_metric_across_devices(_stats(loss=2.0, n_words=1), _Ctx()) == pytest.approx(math.e ** 2)


def test_no_scores_is_none(tmp_path):
    saver = _saver(tmp_path, 'mean_bleu')
    assert saver._mean_metric_across_devices(None, _Ctx()) is None
    assert saver._mean_metric_across_devices(_stats(), _Ctx()) is None


def test_distributed_mean_not_worst(tmp_path, monkeypatch):
    # Rank 0: one task, rank 1: two tasks, rank 2: no validation data.
    def fake_gather(out, obj):
        out[:] = [[10.0], [20.0, 60.0], []]

    monkeypatch.setattr(torch.distributed, 'all_gather_object', fake_gather)
    saver = _saver(tmp_path, 'mean_bleu')
    value = saver._mean_metric_across_devices(_stats(**{'bleu/a': 10.0}), _Ctx(world_size=3))
    assert value == pytest.approx(30.0)


def test_track_metric_records_mean_and_drives_early_stopping(tmp_path, monkeypatch):
    saver = _saver(tmp_path, 'mean_bleu')
    es = EarlyStopping(2)
    # (master task bleu, other device's task bleu): the master's own task keeps
    # improving, but the mean stalls after step 2.
    rounds = [(1, 10.0, 30.0), (2, 12.0, 32.0), (3, 13.0, 30.0), (4, 14.0, 29.0)]
    for step, mine, other in rounds:
        monkeypatch.setattr(
            torch.distributed, 'all_gather_object', lambda out, obj, o=other: out.__setitem__(slice(None), [obj, [o]])
        )
        is_new_best = saver.track_metric(step, _stats(**{'bleu/a': mine}), _Ctx(world_size=2))
        es.update(is_new_best, step)

    assert saver.best_checkpoint_step == 2
    assert saver.best_metric_value == pytest.approx(22.0)
    assert saver.checkpoint_metadata[4]['mean_bleu'] == pytest.approx(21.5)
    assert saver.checkpoint_metadata[4]['bleu/a'] == 14.0  # master's own task kept for reference
    assert es.has_stopped() and es.current_step_best == 2


def test_master_without_validation_still_records_step(tmp_path):
    saver = _saver(tmp_path, 'mean_bleu')
    saver.track_metric(5, None, _Ctx())
    assert saver.checkpoint_metadata[5] == {'mean_bleu': None}


def test_best_n_ranks_by_mean_not_master_task(tmp_path):
    saver = _saver(tmp_path, 'mean_bleu', save_strategy='best_n', keep_checkpoint=2)
    removed = []
    saver._rm_checkpoint_by_step = removed.append
    # 'bleu/a' (the master's task) would rank step 3 first; the mean ranks it last.
    saver.checkpoint_metadata = {
        1: {'bleu/a': 10.0, 'mean_bleu': 25.0},
        2: {'bleu/a': 11.0, 'mean_bleu': None},
        3: {'bleu/a': 40.0, 'mean_bleu': 20.0},
        4: {'bleu/a': 12.0, 'mean_bleu': 26.0},
    }
    saver._rotate_checkpoints_by_metric()
    assert sorted(removed) == [2, 3]
    assert sorted(saver.checkpoint_metadata) == [1, 4]


def test_opts_option_definition():
    from mammoth import opts
    from mammoth.utils.parse import ArgumentParser

    parser = ArgumentParser()
    opts.train_opts(parser)
    action = parser._option_string_actions['--metric_for_best_model']
    assert action.default == 'mean_ppl'
    assert set(action.choices) == {'mean_bleu', 'mean_chrf', 'mean_ppl'}
    assert '--greater_is_better' not in parser._option_string_actions


# ---------------------------------------------------------------------------
# Resume (train_from): best tracking is stored in the frame and restored
# ---------------------------------------------------------------------------

def test_restore_best_metric_from_frame(tmp_path):
    saver = _saver(tmp_path, 'mean_bleu')
    frame = {'best_metric': {'metric': 'mean_bleu', 'value': 33.6, 'step': 37500}}
    assert saver.restore_best_metric(frame)
    assert saver.best_metric_value == 33.6 and saver.best_checkpoint_step == 37500
    # A worse first validation after resuming is no longer a "new best".
    assert saver.track_metric(40000, _stats(**{'bleu/a': 33.0}), _Ctx()) is False
    assert saver.track_metric(42500, _stats(**{'bleu/a': 34.0}), _Ctx()) is True


@pytest.mark.parametrize('frame', [{}, None, {'best_metric': {'metric': 'mean_ppl', 'value': 10.0, 'step': 5}}])
def test_restore_skipped_for_old_frame_or_other_metric(tmp_path, frame):
    saver = _saver(tmp_path, 'mean_bleu')
    assert not saver.restore_best_metric(frame)
    assert saver.best_checkpoint_step is None and saver.best_metric_value == float('-inf')


def test_new_best_frame_records_itself(tmp_path, monkeypatch):
    # save_with_metric must update the best before _save() writes the frame.
    saver = _saver(tmp_path, 'mean_bleu', save_strategy='best_and_last', keep_checkpoint=1)
    seen = {}

    def fake_save(step, model, data_state, tqm):
        seen[step] = saver._best_metric_state()
        return []

    monkeypatch.setattr(saver, '_save', fake_save)
    monkeypatch.setattr(saver, '_apply_best_checkpoint_naming', lambda *a: None)
    monkeypatch.setattr(saver, '_revert_best_checkpoint_naming', lambda *a: None)
    monkeypatch.setattr(saver, '_rm_checkpoint_by_step', lambda *a: None)

    saver.save_with_metric(1, {}, _stats(**{'bleu/a': 10.0}), _Ctx())
    saver.save_with_metric(2, {}, _stats(**{'bleu/a': 9.0}), _Ctx())
    assert seen[1] == {'metric': 'mean_bleu', 'value': 10.0, 'step': 1}
    assert seen[2] == {'metric': 'mean_bleu', 'value': 10.0, 'step': 1}
