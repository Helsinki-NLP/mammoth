"""
End-to-end equivalence test for the Gemma3-270M -> Mammoth (native PyTorch
backend) conversion. See CLAUDE.md "Gemma3-270M -> Mammoth Conversion" for
the architecture gap analysis and the "fake encoder" design this exercises.

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
    from convert_gemma3_native import convert

    mammoth_model, hf_model, tqm = convert(HF_MODEL_PATH, save_path=None, enc_layers=1)
    mammoth_model.eval()
    hf_model.eval()
    return mammoth_model, hf_model, tqm


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
    # Fake-encoder input is irrelevant: cross_attn.to_out is zero-initialized
    # by the converter, so its contribution to the decoder is exactly zero
    # regardless of what the fake encoder produces.
    src = torch.randint(0, 100, (1, 4))
    src_mask = torch.ones(1, 4).bool()

    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids).logits
        mammoth_logits, _ = mammoth_model(src, ids, src_mask, metadata=metadata)

    assert mammoth_logits.shape == hf_logits.shape
    torch.testing.assert_close(mammoth_logits, hf_logits, atol=1e-4, rtol=1e-4)


def test_task_distribution_strategy_is_always_weighted_sampling(converted):
    from mammoth.distributed.tasks import WeightedSamplingTaskDistributionStrategy

    _, _, tqm = converted
    assert isinstance(tqm.task_distribution_strategy, WeightedSamplingTaskDistributionStrategy)


def test_custom_task_id_encoder_decoder_group_and_langs():
    sys.path.insert(0, CONVERTER_DIR)
    from convert_gemma3_native import convert

    mammoth_model, hf_model, tqm = convert(
        HF_MODEL_PATH,
        save_path=None,
        enc_layers=1,
        task_id="eng-fin",
        encoder_group="my_enc_group",
        decoder_group="my_dec_group",
        src_lang="eng",
        tgt_lang="fin",
    )
    mammoth_model.eval()
    hf_model.eval()

    task = tqm.get_my_tasks()[0]
    assert task.corpus_id == "eng-fin"
    assert task.encoder_id == ["my_enc_group"]
    assert task.decoder_id == ["my_dec_group"]
    assert task.src_lang == "eng"
    assert task.tgt_lang == "fin"

    stack = mammoth_model.decoder.get_attention_layers_by_xcoder_id(0, "my_dec_group")
    assert len(stack.blocks) == hf_model.config.num_hidden_layers

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_MODEL_PATH, local_files_only=True)
    ids = tok("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]
    metadata = task.get_serializable_metadata()
    src = torch.randint(0, 100, (1, 4))
    src_mask = torch.ones(1, 4).bool()
    with torch.no_grad():
        hf_logits = hf_model(input_ids=ids).logits
        mammoth_logits, _ = mammoth_model(src, ids, src_mask, metadata=metadata)
    torch.testing.assert_close(mammoth_logits, hf_logits, atol=1e-4, rtol=1e-4)
