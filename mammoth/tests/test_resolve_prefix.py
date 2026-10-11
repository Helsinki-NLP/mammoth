import pytest

from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import resolve_prefix


def _touch(d, *names):
    for n in names:
        (d / n).touch()


@pytest.mark.parametrize('name', ['', 'model'])
def test_best_keeps_model_name(tmp_path, name):
    _touch(tmp_path, f'{name}_best_frame.pt', f'{name}_step_10_frame.pt')
    assert resolve_prefix(str(tmp_path), None) == f'{name}_best'


@pytest.mark.parametrize('name', ['', 'model'])
def test_latest_step_fallback_keeps_model_name(tmp_path, name):
    _touch(tmp_path, f'{name}_step_9_frame.pt', f'{name}_step_100_frame.pt', f'{name}_step_20_frame.pt')
    assert resolve_prefix(str(tmp_path), None) == f'{name}_step_100'


@pytest.mark.parametrize('name', ['', 'model'])
def test_explicit_step_keeps_model_name(tmp_path, name):
    _touch(tmp_path, f'{name}_step_5_frame.pt', f'{name}_step_50_frame.pt', f'{name}_best_frame.pt')
    assert resolve_prefix(str(tmp_path), 5) == f'{name}_step_5'


def test_explicit_step_missing_raises(tmp_path):
    _touch(tmp_path, 'model_step_5_frame.pt')
    with pytest.raises(FileNotFoundError):
        resolve_prefix(str(tmp_path), 6)


def test_no_checkpoint_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_prefix(str(tmp_path), None)


def test_prefix_builds_existing_frame_path(tmp_path):
    _touch(tmp_path, 'model_best_frame.pt')
    assert (tmp_path / f"{resolve_prefix(str(tmp_path), None)}_frame.pt").exists()
