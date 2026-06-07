"""
Tests for MammothLiteRT{Encoder,Prefill,Decode} wrappers.

All tests use random weights (no checkpoint required) with a tiny config so they
run fast on CPU.  The test matrix is:

  - Shape tests       : output tensor count and shapes
  - Correctness tests : wrapper outputs match MammothForConditionalGeneration
  - Export tests      : torch.export.export() succeeds (litert_torch not required)
"""

import pytest
import torch

from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration
from mammoth.litert.wrapper import (
    MammothLiteRTEncoder,
    MammothLiteRTPrefill,
    MammothLiteRTDecode,
    make_encoder_sample_inputs,
    make_prefill_sample_inputs,
    make_decode_sample_inputs,
)

# ── Tiny config: fast on CPU, two stacks to exercise multi-stack paths ─────────

SMALL_CFG = MammothConfig(
    model_dim=128,
    heads=4,
    ff_mult=2.0,
    ff_swiglu=True,
    enc_layers=[2, 1],
    dec_layers=[2, 1],
    src_vocab_size=100,
    tgt_vocab_size=100,
    rotary_pos_emb=True,
    post_emb_norm=True,
    attn_dropout=0.0,
    ff_dropout=0.0,
    emb_dropout=0.0,
)

ENC_MAX = 8
DEC_MAX = 6
HEADS = SMALL_CFG.heads                      # 4
DIM_HEAD = SMALL_CFG.model_dim // HEADS      # 32
N_DEC_LAYERS = sum(SMALL_CFG.dec_layers)     # 3


@pytest.fixture(scope="module")
def hf_model():
    torch.manual_seed(42)
    return MammothForConditionalGeneration(SMALL_CFG).eval()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _zero_kv_flat(n_layers, heads, dim_head, enc_max, dec_max):
    """Return the 4N zero KV tensors expected as prefill/decode input."""
    self_kv = [torch.zeros(1, heads, dec_max, dim_head) for _ in range(2 * n_layers)]
    cross_kv = [torch.zeros(1, heads, enc_max, dim_head) for _ in range(2 * n_layers)]
    return tuple(self_kv + cross_kv)


# ══════════════════════════════════════════════════════════════════════════════
#  1. Shape tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderShape:
    def test_output_is_tuple_of_one(self, hf_model):
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        inputs = make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size)
        with torch.no_grad():
            out = enc(*inputs)
        assert isinstance(out, tuple)
        assert len(out) == 1

    def test_enc_hidden_shape(self, hf_model):
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        inputs = make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size)
        with torch.no_grad():
            out = enc(*inputs)
        assert out[0].shape == (1, ENC_MAX, SMALL_CFG.model_dim)


class TestPrefillShape:
    @pytest.fixture(autouse=True)
    def _build(self, hf_model):
        self.prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        enc_inputs = make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size)
        with torch.no_grad():
            self.enc_out = enc(*enc_inputs)

    def test_output_count(self, hf_model):
        prefill_inputs = make_prefill_sample_inputs(
            self.enc_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.prefill(*prefill_inputs)
        # logits + 2N self-KV + 2N cross-KV = 1 + 4*N
        assert len(out) == 1 + 4 * N_DEC_LAYERS

    def test_logit_shape(self, hf_model):
        prefill_inputs = make_prefill_sample_inputs(
            self.enc_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.prefill(*prefill_inputs)
        assert out[0].shape == (1, DEC_MAX, SMALL_CFG.tgt_vocab_size)

    def test_self_kv_shape(self, hf_model):
        prefill_inputs = make_prefill_sample_inputs(
            self.enc_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.prefill(*prefill_inputs)
        # self KV tensors are outputs 1 .. 2N (inclusive)
        for i in range(2 * N_DEC_LAYERS):
            assert out[1 + i].shape == (1, HEADS, DEC_MAX, DIM_HEAD), \
                f"self-KV[{i}] shape mismatch"

    def test_cross_kv_shape(self, hf_model):
        prefill_inputs = make_prefill_sample_inputs(
            self.enc_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.prefill(*prefill_inputs)
        # cross KV tensors are outputs 1+2N .. 1+4N-1
        for i in range(2 * N_DEC_LAYERS):
            assert out[1 + 2 * N_DEC_LAYERS + i].shape == (1, HEADS, ENC_MAX, DIM_HEAD), \
                f"cross-KV[{i}] shape mismatch"


class TestDecodeShape:
    @pytest.fixture(autouse=True)
    def _build(self, hf_model):
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        enc_inputs = make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size)
        with torch.no_grad():
            self.enc_out = enc(*enc_inputs)

        self.prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
        prefill_inputs = make_prefill_sample_inputs(
            self.enc_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            self.prefill_out = self.prefill(*prefill_inputs)

        self.decode = MammothLiteRTDecode(hf_model, ENC_MAX, DEC_MAX).eval()

    def test_output_count(self, hf_model):
        decode_inputs = make_decode_sample_inputs(
            self.enc_out, self.prefill_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.decode(*decode_inputs)
        # logits + 2N updated self-KV
        assert len(out) == 1 + 2 * N_DEC_LAYERS

    def test_logit_shape(self, hf_model):
        decode_inputs = make_decode_sample_inputs(
            self.enc_out, self.prefill_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.decode(*decode_inputs)
        assert out[0].shape == (1, 1, SMALL_CFG.tgt_vocab_size)

    def test_updated_self_kv_shape(self, hf_model):
        decode_inputs = make_decode_sample_inputs(
            self.enc_out, self.prefill_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        with torch.no_grad():
            out = self.decode(*decode_inputs)
        for i in range(2 * N_DEC_LAYERS):
            assert out[1 + i].shape == (1, HEADS, DEC_MAX, DIM_HEAD), \
                f"updated self-KV[{i}] shape mismatch"


# ══════════════════════════════════════════════════════════════════════════════
#  2. Numerical correctness tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEncoderCorrectness:
    """Encoder wrapper output must exactly match MammothEncoder.forward()."""

    def test_matches_hf_encoder_no_padding(self, hf_model):
        enc_wrapper = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        torch.manual_seed(0)
        input_ids = torch.randint(0, SMALL_CFG.src_vocab_size, (1, ENC_MAX))

        # LiteRT wrapper: all-zero float pad_mask (no padding)
        pad_mask = torch.zeros(1, ENC_MAX)
        # HF encoder: all-one bool attention_mask (all valid)
        attn_mask = torch.ones(1, ENC_MAX, dtype=torch.long)

        with torch.no_grad():
            wrapper_out = enc_wrapper(input_ids, pad_mask)
            hf_out = hf_model.encoder(input_ids, attention_mask=attn_mask)

        torch.testing.assert_close(
            wrapper_out[0], hf_out.last_hidden_state, atol=1e-5, rtol=1e-5
        )

    def test_enc_hidden_changes_with_input(self, hf_model):
        enc_wrapper = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        pad_mask = torch.zeros(1, ENC_MAX)
        ids_a = torch.zeros(1, ENC_MAX, dtype=torch.long)
        ids_b = torch.ones(1, ENC_MAX, dtype=torch.long)
        with torch.no_grad():
            out_a = enc_wrapper(ids_a, pad_mask)[0]
            out_b = enc_wrapper(ids_b, pad_mask)[0]
        assert not torch.allclose(out_a, out_b), "different inputs should produce different outputs"


class TestPrefillCorrectness:
    """Prefill logits must match MammothDecoder teacher-forced output."""

    def test_logits_match_hf_decoder(self, hf_model):
        torch.manual_seed(1)
        input_ids = torch.randint(0, SMALL_CFG.src_vocab_size, (1, ENC_MAX))
        dec_ids = torch.randint(0, SMALL_CFG.tgt_vocab_size, (1, DEC_MAX))

        # Run encoder
        enc_wrapper = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        pad_mask = torch.zeros(1, ENC_MAX)
        with torch.no_grad():
            enc_out = enc_wrapper(input_ids, pad_mask)

        # Run LiteRT prefill (no padding)
        prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
        prefill_inputs = make_prefill_sample_inputs(enc_out, ENC_MAX, DEC_MAX, SMALL_CFG)
        # Replace the placeholder input_ids with our test dec_ids
        prefill_inputs = (enc_out[0], dec_ids) + prefill_inputs[2:]
        with torch.no_grad():
            litert_logits = prefill(*prefill_inputs)[0]

        # Run HF decoder (teacher-forced, no mask → cross-attn sees all encoder positions)
        with torch.no_grad():
            hf_logits = hf_model.decoder(
                dec_ids,
                context=enc_out[0],
                context_mask=None,
                cache=None,
            )

        torch.testing.assert_close(litert_logits, hf_logits, atol=1e-5, rtol=1e-5)

    def test_cross_kv_is_nonzero(self, hf_model):
        """Cross-KV cache must be populated (not left at zeros) after prefill."""
        enc_wrapper = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        pad_mask = torch.zeros(1, ENC_MAX)
        input_ids = torch.randint(0, SMALL_CFG.src_vocab_size, (1, ENC_MAX))
        with torch.no_grad():
            enc_out = enc_wrapper(input_ids, pad_mask)
            prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
            prefill_inputs = make_prefill_sample_inputs(enc_out, ENC_MAX, DEC_MAX, SMALL_CFG)
            out = prefill(*prefill_inputs)
        cross_k_0 = out[1 + 2 * N_DEC_LAYERS]  # first cross-K tensor
        assert cross_k_0.abs().max() > 0, "cross-K cache should be non-zero after prefill"


class TestDecodeCorrectness:
    """
    Consistency test: decode at step (DEC_MAX-1) using the KV from prefill must
    give the same logits as prefill at that final position.

    At the last position (step = DEC_MAX-1) the self-attention sees all positions
    0..DEC_MAX-1, so the causal mask is all-zero — the same attention pattern as
    in the corresponding prefill row.
    """

    @pytest.fixture(autouse=True)
    def _run_prefill(self, hf_model):
        torch.manual_seed(7)
        self.dec_ids = torch.randint(0, SMALL_CFG.tgt_vocab_size, (1, DEC_MAX))
        input_ids = torch.randint(0, SMALL_CFG.src_vocab_size, (1, ENC_MAX))

        enc_wrapper = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        with torch.no_grad():
            self.enc_out = enc_wrapper(input_ids, torch.zeros(1, ENC_MAX))
            prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
            prefill_inputs = make_prefill_sample_inputs(
                self.enc_out, ENC_MAX, DEC_MAX, SMALL_CFG
            )
            prefill_inputs = (self.enc_out[0], self.dec_ids) + prefill_inputs[2:]
            self.prefill_out = prefill(*prefill_inputs)
            self.prefill_logits = self.prefill_out[0]

        self.decode = MammothLiteRTDecode(hf_model, ENC_MAX, DEC_MAX).eval()

    def test_last_step_matches_prefill(self, hf_model):
        """
        Decode at the last position should match prefill's logit for that position.
        step_index = DEC_MAX - 1 means the token can attend to all previous
        positions: self-attn mask is all-zero, identical to the last prefill row.
        """
        step = DEC_MAX - 1
        token_at_step = self.dec_ids[:, step:step + 1]  # (1, 1)

        decode_inputs = make_decode_sample_inputs(
            self.enc_out, self.prefill_out, ENC_MAX, DEC_MAX, SMALL_CFG,
            step_index=step,
            input_ids=token_at_step,
        )
        with torch.no_grad():
            decode_out = self.decode(*decode_inputs)
        decode_logits = decode_out[0]  # (1, 1, vocab)

        expected = self.prefill_logits[:, step:step + 1, :]  # (1, 1, vocab)
        torch.testing.assert_close(decode_logits, expected, atol=1e-4, rtol=1e-4)

    def test_decode_updates_self_kv_at_step(self, hf_model):
        """After a decode step, the returned self-K at step_index should be nonzero."""
        step = 0
        token_at_step = self.dec_ids[:, step:step + 1]
        decode_inputs = make_decode_sample_inputs(
            self.enc_out, self.prefill_out, ENC_MAX, DEC_MAX, SMALL_CFG,
            step_index=step,
            input_ids=token_at_step,
        )
        with torch.no_grad():
            decode_out = self.decode(*decode_inputs)
        updated_self_k0 = decode_out[1]  # first updated self-K
        # Position 0 should now have a non-zero K from the new token
        assert updated_self_k0[:, :, step, :].abs().max() > 0


# ══════════════════════════════════════════════════════════════════════════════
#  3. torch.export compatibility
#
#  These tests verify that each wrapper can be traced by torch.export.export()
#  without raising — the exported program can then be lowered to LiteRT.
# ══════════════════════════════════════════════════════════════════════════════

class TestTorchExport:
    """torch.export must succeed for all three wrappers."""

    def test_encoder_is_exportable(self, hf_model):
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        inputs = make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size)
        exported = torch.export.export(enc, inputs)
        assert exported is not None

    def test_prefill_is_exportable(self, hf_model):
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        enc_out = enc(*make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size))
        prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
        prefill_inputs = make_prefill_sample_inputs(enc_out, ENC_MAX, DEC_MAX, SMALL_CFG)
        exported = torch.export.export(prefill, prefill_inputs)
        assert exported is not None

    def test_decode_is_exportable(self, hf_model):
        enc = MammothLiteRTEncoder(hf_model, ENC_MAX).eval()
        enc_out = enc(*make_encoder_sample_inputs(ENC_MAX, SMALL_CFG.src_vocab_size))
        prefill = MammothLiteRTPrefill(hf_model, ENC_MAX, DEC_MAX).eval()
        prefill_out = prefill(
            *make_prefill_sample_inputs(enc_out, ENC_MAX, DEC_MAX, SMALL_CFG)
        )
        decode = MammothLiteRTDecode(hf_model, ENC_MAX, DEC_MAX).eval()
        decode_inputs = make_decode_sample_inputs(
            enc_out, prefill_out, ENC_MAX, DEC_MAX, SMALL_CFG
        )
        exported = torch.export.export(decode, decode_inputs)
        assert exported is not None
