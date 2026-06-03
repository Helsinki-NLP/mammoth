"""TDD tests for Mammoth → TFLite conversion via litert-torch.

Test ladder (each step builds on the previous):
  1. test_model_loads              – mammoth_single_stack checkpoint loads without error
  2. test_teacher_forced_forward   – NMTModel.forward produces logits of the right shape
  3. test_export_wrapper_matches_nmt – MammothExportWrapper logits == NMTModel logits
  4. test_tflite_conversion        – litert-torch produces a .tflite file (Linux only)
  5. test_tflite_interpreter       – Interpreter API logits ≈ PyTorch logits (Linux only)

Run with:
    pytest mammoth/tests/test_litert_conversion.py -v
"""

import argparse
import os
import platform
import sys
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

MODEL_DIR = os.path.join(REPO_ROOT, "mammoth/hf_integration/to_hf/mammoth_single_stack/")
TFLITE_PATH = os.path.join(REPO_ROOT, "mammoth/litert/mammoth_es_en.tflite")

TASK_ID = "mt_es-en"
SRC_LANG = "es"
TGT_LANG = "en"
SEQ_LEN = 16   # short sequences for fast testing

# Skip the whole module if the checkpoint is absent.
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_DIR),
    reason=f"mammoth_single_stack not found at {MODEL_DIR}",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def nmt_and_vocabs():
    """Load mammoth_single_stack and return (vocabs_dict, model, model_opts)."""
    from mammoth.translate.translator import load_model_for_translation
    from mammoth.distributed import TaskSpecs, TaskQueueManager
    from mammoth.distributed.contexts import WorldContext, DeviceContextEnum

    task = TaskSpecs(
        node_rank=0,
        local_rank=0,
        src_lang=SRC_LANG,
        tgt_lang=TGT_LANG,
        encoder_id=[SRC_LANG],
        decoder_id=[TGT_LANG],
        corpus_id=TASK_ID,
        weight=1.0,
        introduce_at_training_step=0,
        corpus_opts={"src_tgt": f"{SRC_LANG}-{TGT_LANG}"},
        src_vocab=None,
        tgt_vocab=None,
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )

    # Minimal opts: CPU, no gpu_ranks so use_gpu() returns False.
    opts = argparse.Namespace(gpu=-1, gpu_ranks=[], seed=42, log_model_structure=False)

    world_context = WorldContext(
        context=DeviceContextEnum.CPU, n_nodes=1, gpus_per_node=0
    )
    tqm = (
        TaskQueueManager(
            tasks=[task],
            accum_count=1,
            world_context=world_context,
            task_distribution_strategy_cls=None,
            uses_adapters=False,
        )
        .global_to_local(node_rank=0, local_rank=0, opts=opts)
    )
    tqm.create_all_distributed_components(use_attention_bridge=False)

    vocabs_dict, model, model_opts = load_model_for_translation(
        opts=opts,
        task_queue_manager=tqm,
        task=task,
        model_path=MODEL_DIR,
    )
    return vocabs_dict, model, model_opts


@pytest.fixture(scope="module")
def test_tensors(nmt_and_vocabs):
    """Tokenise a Spanish sentence and return padded tensors for the test."""
    vocabs_dict, _, _ = nmt_and_vocabs
    src_vocab = vocabs_dict[("src", SRC_LANG)]
    tgt_vocab = vocabs_dict[("tgt", TGT_LANG)]

    sentence = "Hola, ¿cómo estás?"
    raw_ids = [src_vocab.stoi[t] for t in src_vocab.tokenize(sentence)]

    pad_id = src_vocab.stoi["<pad>"]
    ids = raw_ids[:SEQ_LEN]
    ids += [pad_id] * (SEQ_LEN - len(ids))

    src_tokens = torch.tensor([ids], dtype=torch.long)           # (1, SEQ_LEN)
    src_mask = src_tokens != pad_id                               # (1, SEQ_LEN)

    bos_id = tgt_vocab.stoi.get("<s>", 0)
    decoder_tokens = torch.tensor([[bos_id, bos_id]], dtype=torch.long)  # (1, 2)

    return src_tokens, decoder_tokens, src_mask, src_vocab, tgt_vocab


# ---------------------------------------------------------------------------
# Tests 1 & 2 – baseline NMTModel
# ---------------------------------------------------------------------------

def test_model_loads(nmt_and_vocabs):
    _, model, _ = nmt_and_vocabs
    assert model is not None
    assert hasattr(model, "encoder")
    assert hasattr(model, "decoder")


def test_teacher_forced_forward(nmt_and_vocabs, test_tensors):
    _, model, _ = nmt_and_vocabs
    src_tokens, decoder_tokens, src_mask, _, tgt_vocab = test_tensors

    metadata = SimpleNamespace(
        corpus_id=TASK_ID,
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )

    with torch.no_grad():
        logits, decoder_out = model(src_tokens, decoder_tokens, src_mask, metadata=metadata)

    assert logits.ndim == 3
    assert logits.shape[0] == 1                         # batch
    assert logits.shape[1] == decoder_tokens.shape[1]   # tgt_len
    assert logits.shape[2] == len(tgt_vocab)            # vocab_size


# ---------------------------------------------------------------------------
# Test 3 – export wrapper parity
# ---------------------------------------------------------------------------

def test_export_wrapper_matches_nmt(nmt_and_vocabs, test_tensors):
    """MammothExportWrapper must produce bitwise-identical logits to NMTModel."""
    from mammoth.litert.converter import MammothExportWrapper, patch_adapted_attention_layers

    vocabs_dict, model, _ = nmt_and_vocabs
    src_tokens, decoder_tokens, src_mask, _, _ = test_tensors

    metadata = SimpleNamespace(
        corpus_id=TASK_ID,
        encoder_adapter_ids=None,
        decoder_adapter_ids=None,
    )

    # Cast to float32: the checkpoint is bf16; we want a clean float32 baseline.
    model_f32 = model.float().eval()

    patch_adapted_attention_layers(model_f32)
    wrapper = MammothExportWrapper(model_f32, TASK_ID).eval()

    src_f32 = src_tokens            # integer tokens — no dtype change needed
    dec_f32 = decoder_tokens

    with torch.no_grad():
        logits_ref, _ = model_f32(src_f32, dec_f32, src_mask, metadata=metadata)
        logits_wrap = wrapper(src_f32, dec_f32, src_mask)

    torch.testing.assert_close(logits_ref, logits_wrap, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Test 4 – TFLite conversion (Linux only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="litert-torch only supports Linux",
)
def test_tflite_conversion(nmt_and_vocabs, tmp_path):
    """convert_to_tflite() produces a non-empty .tflite file."""
    from mammoth.litert.converter import convert_to_tflite

    _, model, _ = nmt_and_vocabs
    out = str(tmp_path / "test_mammoth_es_en.tflite")

    convert_to_tflite(
        model,
        task_id=TASK_ID,
        output_path=out,
        seq_len=SEQ_LEN,
        tgt_len=4,
        quantize=False,
    )

    assert os.path.exists(out)
    assert os.path.getsize(out) > 1024  # at minimum a few KB


# ---------------------------------------------------------------------------
# Test 5 – TFLite Interpreter API (Linux only, .tflite must exist)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    platform.system() != "Linux",
    reason="litert-torch only supports Linux",
)
@pytest.mark.skipif(
    not os.path.exists(TFLITE_PATH),
    reason=f"Pre-built .tflite not found at {TFLITE_PATH}",
)
def test_tflite_interpreter_matches_pytorch(nmt_and_vocabs, test_tensors):
    """TFLite Interpreter output must be close to the PyTorch float32 reference."""
    import numpy as np
    from ai_edge_litert.interpreter import Interpreter
    from mammoth.litert.converter import MammothExportWrapper, patch_adapted_attention_layers

    _, model, _ = nmt_and_vocabs
    src_tokens, decoder_tokens, src_mask, _, _ = test_tensors

    # ---- PyTorch reference ------------------------------------------------
    model_f32 = model.float().eval()
    patch_adapted_attention_layers(model_f32)
    wrapper = MammothExportWrapper(model_f32, TASK_ID).eval()

    with torch.no_grad():
        logits_ref = wrapper(src_tokens, decoder_tokens, src_mask)

    # ---- TFLite Interpreter -----------------------------------------------
    interp = Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()

    in_details = interp.get_input_details()
    out_details = interp.get_output_details()

    # The litert-torch converter names inputs args_0, args_1, args_2 in order.
    interp.set_tensor(in_details[0]["index"], src_tokens.numpy().astype(np.int32))
    interp.set_tensor(in_details[1]["index"], decoder_tokens.numpy().astype(np.int32))
    interp.set_tensor(in_details[2]["index"], src_mask.numpy())
    interp.invoke()

    logits_tflite = torch.tensor(interp.get_tensor(out_details[0]["index"]))

    # Quantised models have slightly lower precision; allow 1e-2 tolerance.
    torch.testing.assert_close(logits_ref, logits_tflite, atol=1e-2, rtol=1e-2)
