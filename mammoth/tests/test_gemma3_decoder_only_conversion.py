"""
End-to-end equivalence test for the Gemma3-270M -> Mammoth TRUE decoder-only
conversion (convert_gemma3_decoder_only.py). Unlike
test_gemma3_native_conversion.py (which exercises the fake-encoder
converter), this test's converted model has no encoder at all
(model.encoder is None) and no cross-attention modules -- the decoder is
architecturally identical to Gemma3, not just numerically equivalent to it
via a zeroed cross-attn contribution.

See CLAUDE.md "Gemma3-270M -> Mammoth Conversion" -> "Decision: fake-encoder
path chosen for this conversion" for why this alternative ("option 1") was
deferred, and is now implemented here.

Skipped if the real checkpoint isn't available locally (this repo's
hf_models/gemma3_270m is a local asset, not something to fetch in CI).
"""
import os
import sys

import pytest
import torch

HF_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../hf_models/gemma3_270m")
)
CONVERTER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../hf_integration/from_hf/gemma3")
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(HF_MODEL_PATH, "model.safetensors")),
    reason=f"gemma3_270m checkpoint not found at {HF_MODEL_PATH}",
)


@pytest.fixture(scope="module")
def converted():
    sys.path.insert(0, CONVERTER_DIR)
    from convert_gemma3_decoder_only import convert

    mammoth_model, hf_model, tqm = convert(HF_MODEL_PATH, save_path=None)
    mammoth_model.eval()
    hf_model.eval()
    return mammoth_model, hf_model, tqm


def test_model_has_no_encoder(converted):
    mammoth_model, _, _ = converted
    assert mammoth_model.encoder is None
    assert mammoth_model.attention_bridge is None


def test_decoder_blocks_have_no_cross_attn(converted):
    mammoth_model, hf_model, _ = converted
    stack = mammoth_model.decoder.get_attention_layers_by_xcoder_id(0, "gemma3_dec")
    assert len(stack.blocks) == hf_model.config.num_hidden_layers
    for block in stack.blocks:
        assert not hasattr(block, "cross_attn") or block.cross_attn is None


@pytest.mark.parametrize(
    "text",
    [
        "The quick brown fox jumps over the lazy dog.",
        "Mammoth is a multilingual machine translation framework.",
        "2 + 2 =",
    ],
)
def test_logits_match_hf_exactly(converted, text):
    from transformers import AutoTokenizer

    mammoth_model, hf_model, tqm = converted
    tok = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
    ids = tok(text, return_tensors="pt")["input_ids"]

    task = tqm.get_my_tasks()[0]
    metadata = task.get_serializable_metadata()

    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids).logits
        # No encoder / src input at all: this is a true decoder-only forward.
        mammoth_logits, _ = mammoth_model(None, ids, None, metadata=metadata)

    assert mammoth_logits.shape == hf_logits.shape
    torch.testing.assert_close(mammoth_logits, hf_logits, atol=1e-4, rtol=1e-4)


def test_task_distribution_strategy_is_always_weighted_sampling(converted):
    from mammoth.distributed.tasks import WeightedSamplingTaskDistributionStrategy

    _, _, tqm = converted
    assert isinstance(tqm.task_distribution_strategy, WeightedSamplingTaskDistributionStrategy)


def test_custom_task_id_decoder_group_and_lang():
    sys.path.insert(0, CONVERTER_DIR)
    from convert_gemma3_decoder_only import convert

    mammoth_model, hf_model, tqm = convert(
        HF_MODEL_PATH,
        save_path=None,
        task_id="fin-lm",
        decoder_group="my_dec_group",
        lang="fin",
    )
    mammoth_model.eval()
    hf_model.eval()

    task = tqm.get_my_tasks()[0]
    assert task.corpus_id == "fin-lm"
    assert task.encoder_id == []
    assert task.decoder_id == ["my_dec_group"]
    assert task.tgt_lang == "fin"

    stack = mammoth_model.decoder.get_attention_layers_by_xcoder_id(0, "my_dec_group")
    assert len(stack.blocks) == hf_model.config.num_hidden_layers

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]
    metadata = task.get_serializable_metadata()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids).logits
        mammoth_logits, _ = mammoth_model(None, ids, None, metadata=metadata)
    torch.testing.assert_close(mammoth_logits, hf_logits, atol=1e-4, rtol=1e-4)
