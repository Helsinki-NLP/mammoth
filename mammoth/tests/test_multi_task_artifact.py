"""Tests for bundled multi-task Mammoth→HF conversion (single repo, flat shards).

Covers:
- Conversion bundles all tasks into one directory with per-task safetensors + shared code.
- Wrapper loads a specific task and produces finite logits.
- Both tasks load and produce correct vocab-sized outputs.
- generate() works end-to-end.
- Unknown task raises a clear error.

Run:
    pytest mammoth/tests/test_multi_task_artifact.py -v

Skipped when the fixture checkpoint isn't present.
"""

import json
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MULTI_TASK_CKPT = REPO_ROOT / "mammoth/hf_integration/to_hf/mammoth_two_tasks/model"


def _convert_and_load(tmp_path, task):
    """Run conversion, then load a specific task via the wrapper."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
        convert_multi_task_artifact,
    )
    from mammoth.hf_integration.to_hf.mammoth_hub import MammothHub

    convert_multi_task_artifact(
        ckpt_dir=str(MULTI_TASK_CKPT),
        output_dir=str(tmp_path),
    )
    model = MammothHub.from_pretrained(str(tmp_path), task=task)
    model.eval()
    return model


# ---- Tests ----


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_bundled_artifact_has_all_files(tmp_path):
    """Conversion produces per-task safetensors, shared code, and a manifest."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
        convert_multi_task_artifact,
    )

    convert_multi_task_artifact(
        ckpt_dir=str(MULTI_TASK_CKPT),
        output_dir=str(tmp_path),
    )

    # Manifest config
    assert (tmp_path / "config.json").is_file()
    with open(tmp_path / "config.json") as f:
        manifest = json.load(f)
    assert "eng-spa" in manifest["tasks"]
    assert "eng-fra" in manifest["tasks"]

    # Per-task weight files
    assert (tmp_path / "eng-spa.safetensors").is_file()
    assert (tmp_path / "eng-fra.safetensors").is_file()

    # Shared code (vendored once, not per-task)
    assert (tmp_path / "configuration_mammoth.py").is_file()
    assert (tmp_path / "modeling_mammoth.py").is_file()
    assert (tmp_path / "x_transformers.py").is_file()


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_load_eng_spa_forward(tmp_path):
    """Wrapper loads eng-spa and forward produces finite logits."""
    model = _convert_and_load(tmp_path, "eng-spa")
    config = model.config

    B, S, T = 2, 6, 4
    src = torch.randint(4, max(5, config.src_vocab_size - 1), (B, S))
    tgt = torch.randint(4, max(5, config.tgt_vocab_size - 1), (B, T))

    with torch.no_grad():
        out = model(input_ids=src, decoder_input_ids=tgt)

    assert torch.isfinite(out.logits).all(), "logits contain NaN/Inf"
    assert out.logits.shape == (B, T, config.tgt_vocab_size)


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_load_eng_fra_forward(tmp_path):
    """Wrapper loads eng-fra and forward produces finite logits with its vocab size."""
    model = _convert_and_load(tmp_path, "eng-fra")
    config = model.config

    B, S, T = 2, 6, 4
    src = torch.randint(4, max(5, config.src_vocab_size - 1), (B, S))
    tgt = torch.randint(4, max(5, config.tgt_vocab_size - 1), (B, T))

    with torch.no_grad():
        out = model(input_ids=src, decoder_input_ids=tgt)

    assert torch.isfinite(out.logits).all(), "logits contain NaN/Inf"
    assert out.logits.shape == (B, T, config.tgt_vocab_size)


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_generate_eng_spa(tmp_path):
    """generate() produces tokens end-to-end."""
    model = _convert_and_load(tmp_path, "eng-spa")
    config = model.config

    B, S = 2, 6
    src = torch.randint(4, max(5, config.src_vocab_size - 1), (B, S))

    with torch.no_grad():
        output = model.generate(input_ids=src, max_length=10)

    assert output.shape[0] == B
    assert output.shape[1] >= 1
    assert output.shape[1] <= 10


@pytest.mark.skipif(not MULTI_TASK_CKPT.is_dir(), reason=f"missing {MULTI_TASK_CKPT}")
def test_unknown_task_raises(tmp_path):
    """Loading an unknown task raises a clear error."""
    from mammoth.hf_integration.to_hf.convert_mammoth_to_hf import (
        convert_multi_task_artifact,
    )
    from mammoth.hf_integration.to_hf.mammoth_hub import MammothHub

    convert_multi_task_artifact(
        ckpt_dir=str(MULTI_TASK_CKPT),
        output_dir=str(tmp_path),
    )

    with pytest.raises(ValueError, match="not found"):
        MammothHub.from_pretrained(str(tmp_path), task="zzz-yyy")
