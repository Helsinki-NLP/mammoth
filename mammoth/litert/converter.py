"""Convert a Mammoth NMTModel to TFLite using litert-torch.

Typical usage:

    from mammoth.litert.converter import convert_to_tflite
    convert_to_tflite(nmt_model, task_id='mt_es-en', output_path='mammoth_es_en.tflite')

The produced .tflite runs a teacher-forced forward pass:
    inputs:  src_tokens (int32, [1, seq_len])
             decoder_tokens (int32, [1, tgt_len])
             src_mask (bool, [1, seq_len])  True = valid token
    outputs: logits (float32, [1, tgt_len, vocab_size])

litert-torch only supports Linux.  Running convert_to_tflite on macOS raises RuntimeError.
"""

import sys
import types

import torch
import torch.nn as nn

from mammoth.x_transformers.x_transformers import AttentionLayers


class MammothExportWrapper(nn.Module):
    """Teacher-forced wrapper around a single-task NMTModel, export-safe.

    The task_id is baked in at construction time so the forward signature
    contains only tensors — a hard requirement for torch.export / TFLite.

    All AdaptedAttentionLayers in the model must be patched with
    patch_adapted_attention_layers() before creating this wrapper to avoid
    the self.layers reassignment that torch.export rejects.
    """

    def __init__(self, nmt_model: nn.Module, task_id: str):
        super().__init__()
        # activate() is a plain dict lookup; with adapter_ids=None it sets
        # self.active_task (a Python string) and returns the TransformerWrapper.
        self.encoder = nmt_model.encoder.activate(task_id=task_id, adapter_ids=None)
        self.decoder = nmt_model.decoder.activate(task_id=task_id, adapter_ids=None)
        self.attention_bridge = nmt_model.attention_bridge

    def forward(
        self,
        src_tokens: torch.Tensor,      # int64, (batch, src_len)
        decoder_tokens: torch.Tensor,  # int64, (batch, tgt_len)
        src_mask: torch.Tensor,        # bool,  (batch, src_len), True = valid
    ) -> torch.Tensor:
        encoder_output = self.encoder(x=src_tokens, mask=src_mask, return_embeddings=True)

        if self.attention_bridge is not None:
            encoder_output, _ = self.attention_bridge(encoder_output, src_mask)
            if self.attention_bridge.is_fixed_length:
                src_mask = None

        retval = self.decoder(
            decoder_tokens,
            context=encoder_output,
            context_mask=src_mask,
            return_attn=False,
            return_logits_and_embeddings=True,
        )
        logits, _ = retval
        return logits


def patch_adapted_attention_layers(model: nn.Module) -> None:
    """Replace AdaptedAttentionLayers.forward with AttentionLayers.forward.

    torch.export (used internally by litert-torch) disallows reassigning a
    registered nn.ModuleList submodule inside forward().
    AdaptedAttentionLayers._inject_adapters() always does:
        self.layers = adapted_layers   # nn.ModuleList reassignment — EXPORT BLOCKER
    even when no adapters are active.

    With zero adapters the injection is a no-op, so we can bypass it entirely
    by calling AttentionLayers.forward directly.  This patch must be applied
    to the live model *before* creating MammothExportWrapper.
    """
    from mammoth.modules.adapters import AdaptedAttentionLayers

    for module in model.modules():
        if isinstance(module, AdaptedAttentionLayers):
            module.forward = types.MethodType(AttentionLayers.forward, module)


def _make_sample_inputs(seq_len: int, tgt_len: int) -> tuple:
    """Return (src_tokens, decoder_tokens, src_mask) dummy tensors for tracing."""
    src_tokens = torch.zeros((1, seq_len), dtype=torch.long)
    decoder_tokens = torch.zeros((1, tgt_len), dtype=torch.long)
    src_mask = torch.ones((1, seq_len), dtype=torch.bool)
    return src_tokens, decoder_tokens, src_mask


def convert_to_tflite(
    nmt_model: nn.Module,
    task_id: str,
    output_path: str,
    seq_len: int = 128,
    tgt_len: int = 128,
    quantize: bool = True,
) -> None:
    """Convert a Mammoth NMTModel to a TFLite flatbuffer.

    Args:
        nmt_model: loaded Mammoth NMTModel (CPU, eval mode).
        task_id: the corpus_id to bake into the wrapper (e.g. 'mt_es-en').
        output_path: where to write the .tflite file.
        seq_len: fixed encoder sequence length for the traced graph.
        tgt_len: fixed decoder sequence length for the traced graph.
        quantize: apply dynamic int8 quantisation (reduces file size ~4x).

    Raises:
        RuntimeError: if not running on Linux (litert-torch hard requirement).
    """
    if sys.platform != 'linux':
        raise RuntimeError(
            f"litert-torch only supports Linux, got platform: {sys.platform!r}"
        )

    import litert_torch

    # TFLite does not support bfloat16 — cast the whole model to float32.
    nmt_model = nmt_model.float().eval()

    patch_adapted_attention_layers(nmt_model)
    wrapper = MammothExportWrapper(nmt_model, task_id).eval()

    sample_inputs = _make_sample_inputs(seq_len, tgt_len)

    quant_config = None
    if quantize:
        from litert_torch._convert.interface import _default_quant_config
        try:
            quant_config = _default_quant_config()
        except Exception:
            # _default_quant_config is an internal; fall back gracefully.
            quant_config = None

    edge_model = litert_torch.convert(
        wrapper,
        sample_args=sample_inputs,
        quant_config=quant_config,
    )
    edge_model.export(output_path)
