"""get_gemma3_rope_thetas must work with both the transformers 4.x and 5.x Gemma3 config layouts."""
import os
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
CONVERTER_DIR = os.path.join(REPO_ROOT, "mammoth/hf_integration/from_hf/gemma3")
HF_MODEL_PATH = os.path.join(REPO_ROOT, "..", "hf_models", "gemma3_270m")

sys.path.insert(0, CONVERTER_DIR)
from gemma3_rope import get_gemma3_rope_thetas  # noqa: E402


def test_flat_attributes_transformers_4x():
    config = SimpleNamespace(rope_theta=1_000_000.0, rope_local_base_freq=10_000.0, rope_scaling=None)
    assert get_gemma3_rope_thetas(config) == (1_000_000.0, 10_000.0)


def test_nested_rope_parameters_transformers_5x():
    config = SimpleNamespace(
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0},
        }
    )
    assert get_gemma3_rope_thetas(config) == (1_000_000.0, 10_000.0)


def test_nested_rope_parameters_win_over_flat_attributes():
    config = SimpleNamespace(
        rope_parameters={
            "sliding_attention": {"rope_theta": 2.0},
            "full_attention": {"rope_theta": 1.0},
        },
        rope_theta=100.0,
        rope_local_base_freq=200.0,
    )
    assert get_gemma3_rope_thetas(config) == (1.0, 2.0)


def test_missing_thetas_raise_clear_error():
    with pytest.raises(ValueError, match="Gemma3 RoPE thetas"):
        get_gemma3_rope_thetas(SimpleNamespace())


@pytest.mark.skipif(
    not os.path.exists(os.path.join(HF_MODEL_PATH, "config.json")),
    reason=f"gemma3_270m config not found at {HF_MODEL_PATH}",
)
def test_real_gemma3_270m_config():
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(HF_MODEL_PATH)
    assert get_gemma3_rope_thetas(config) == (1_000_000.0, 10_000.0)
