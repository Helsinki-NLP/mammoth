"""Tests for multi-task Mammoth→HF conversion.

Covers:
- Multi-task checkpoint converts every task by default, producing one HF subdir per task.
- --task filter converts only the requested task.
- Each converted task loads via AutoModelForSeq2SeqLM and produces finite logits.
- Single-task checkpoint still converts flat into the output directory (no subdir).

Run:
    pytest mammoth/tests/test_multi_task_conversion.py -v

Skipped when the fixture checkpoints aren't present.
"""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MULTI_TASK_CKPT = REPO_ROOT / "mammoth/hf_integration/to_hf/mammoth_two_tasks/model"
SINGLE_TASK_CKPT = REPO_ROOT / "mammoth/hf_integration/to_hf/mammoth_double_stack"


def _load_and_forward(model_dir: Path, src_vocab_size: int | None = None):
    """Load a converted HF dir and run a small forward pass; return logits shape."""
    from transformers import AutoModelForSeq2SeqLM

    model = AutoModelForSeq2SeqLM.from_pretrained(str(model_dir), trust_remote_code=True)
    model.eval()
    V_src = src_vocab_size or model.config.src_vocab_size
    B, S, T = 2, 6, 4
    src = torch.randint(4, max(5, V_src - 1), (B, S))
    tgt = torch.randint(4, max(5, model.config.tgt_vocab_size - 1), (B, T))
    with torch.no_grad():
        out = model(input_ids=src, decoder_input_ids=tgt)
    assert torch.isfinite(out.logits).all(), "logits contain NaN/Inf"
    return tuple(out.logits.shape)


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_multi_task_default_converts_all_tasks(tmp_path):
    """Default (no --task) on a multi-task checkpoint produces one subdir per task."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import convert_checkpoint

    convert_checkpoint(
        ckpt_dir=str(MULTI_TASK_CKPT),
        output_dir=str(tmp_path),
        task=None,
        step=None,
    )

    eng_spa = tmp_path / "eng-spa"
    eng_fra = tmp_path / "eng-fra"
    assert eng_spa.is_dir(), f"missing per-task subdir: {eng_spa}"
    assert eng_fra.is_dir(), f"missing per-task subdir: {eng_fra}"

    for sub in (eng_spa, eng_fra):
        assert (sub / "config.json").is_file()
        assert (sub / "configuration_mammoth.py").is_file()
        assert (sub / "modeling_mammoth.py").is_file()
        assert (sub / "x_transformers.py").is_file()

    spa_shape = _load_and_forward(eng_spa)
    fra_shape = _load_and_forward(eng_fra)
    # Both tasks share enc; tgt vocab is per-language. Last dim is tgt_vocab_size.
    assert spa_shape[:2] == (2, 4)
    assert fra_shape[:2] == (2, 4)


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_multi_task_filter_converts_single_task(tmp_path):
    """--task eng-spa on a multi-task checkpoint converts only that task."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import convert_checkpoint

    convert_checkpoint(
        ckpt_dir=str(MULTI_TASK_CKPT),
        output_dir=str(tmp_path),
        task="eng-spa",
        step=None,
    )

    assert (tmp_path / "eng-spa").is_dir()
    assert not (tmp_path / "eng-fra").exists(), "filter leaked: eng-fra should not be converted"
    _load_and_forward(tmp_path / "eng-spa")


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_multi_task_filter_unknown_task_errors(tmp_path):
    """--task with a pair not in opts.tasks raises a clear error."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import convert_checkpoint

    with pytest.raises(ValueError, match="not found"):
        convert_checkpoint(
            ckpt_dir=str(MULTI_TASK_CKPT),
            output_dir=str(tmp_path),
            task="zzz-yyy",
            step=None,
        )


@pytest.mark.skipif(not SINGLE_TASK_CKPT.is_dir(), reason=f"missing {SINGLE_TASK_CKPT}")
def test_single_task_writes_flat(tmp_path):
    """Single-task checkpoint writes directly into output_dir — no subdir."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import convert_checkpoint

    convert_checkpoint(
        ckpt_dir=str(SINGLE_TASK_CKPT),
        output_dir=str(tmp_path),
        task=None,
        step=None,
    )

    assert (tmp_path / "config.json").is_file(), "single-task should write flat, not into a subdir"
    assert (tmp_path / "configuration_mammoth.py").is_file()
    _load_and_forward(tmp_path)
