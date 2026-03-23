"""Tests for FLOP counting and TFLOPs reporting."""
import time
import pytest

from mammoth.utils.flops import compute_transformer_flops
from mammoth.utils.statistics import Statistics


# Shared test config
BASE_KWARGS = dict(
    n_src_tokens=1024,
    n_tgt_tokens=1024,
    src_seq_len=128,
    tgt_seq_len=128,
    model_dim=512,
    enc_layers=6,
    dec_layers=6,
    vocab_size=32000,
)


class TestComputeTransformerFlops:
    def test_flops_known_values(self):
        """Verify FLOP count for a known config against hand-calculated value."""
        flops = compute_transformer_flops(**BASE_KWARGS)

        # Hand calculation:
        # ff_inner = 512 * 4.0 = 2048
        # mlp_flops = 4 * 512 * 2048 = 4_194_304
        # enc_self_attn = 4 * 512 * 512 + 2 * 512 * 128 = 1_048_576 + 131_072 = 1_179_648
        # dec_self_attn = same = 1_179_648
        # cross_attn = 4 * 512 * 512 + 2 * 512 * 128 = 1_179_648
        # enc_per_token = 6 * (4_194_304 + 1_179_648) = 6 * 5_373_952 = 32_243_712
        # dec_per_token = 6 * (4_194_304 + 1_179_648 + 1_179_648) = 6 * 6_553_600 = 39_321_600
        # logit_per_token = 2 * 512 * 32000 = 32_768_000
        # total = 1024 * 32_243_712 + 1024 * (39_321_600 + 32_768_000)
        #       = 1024 * 32_243_712 + 1024 * 72_089_600
        #       = 33_017_561_088 + 73_819_750_400
        #       = 106_837_311_488
        # * 6 (3x fwd+bwd * 2x FMA) = 641_023_868_928
        expected = 641_023_868_928
        assert flops == expected, f"Expected {expected}, got {flops}"

    def test_flops_scales_with_layers(self):
        """Doubling layers should approximately double FLOPs (logit term is constant)."""
        flops_base = compute_transformer_flops(**BASE_KWARGS)
        kwargs_double = {**BASE_KWARGS, 'enc_layers': 12, 'dec_layers': 12}
        flops_double = compute_transformer_flops(**kwargs_double)

        # The logit term doesn't scale with layers, so ratio < 2.0
        # With a large vocab (32k) the logit term is significant
        ratio = flops_double / flops_base
        assert 1.5 < ratio < 2.0, f"Layer scaling ratio {ratio} not in expected range"

    def test_flops_scales_with_tokens(self):
        """Doubling token count should exactly double FLOPs."""
        flops_base = compute_transformer_flops(**BASE_KWARGS)
        kwargs_double = {**BASE_KWARGS, 'n_src_tokens': 2048, 'n_tgt_tokens': 2048}
        flops_double = compute_transformer_flops(**kwargs_double)

        assert flops_double == 2 * flops_base

    def test_flops_glu_increases(self):
        """GLU should increase FLOPs via 1.5x MLP inner dimension."""
        flops_no_glu = compute_transformer_flops(**BASE_KWARGS, use_glu=False)
        flops_glu = compute_transformer_flops(**BASE_KWARGS, use_glu=True)

        assert flops_glu > flops_no_glu

        # The MLP component should be 1.5x larger with GLU.
        # Extract the MLP contribution by computing with and without attention.
        # Simpler: just check the ratio is > 1 and < 1.5 (since attention doesn't change)
        ratio = flops_glu / flops_no_glu
        assert 1.0 < ratio < 1.5, f"GLU ratio {ratio} not in expected range"

    def test_flops_zero_tokens(self):
        """Zero tokens should produce zero FLOPs."""
        flops = compute_transformer_flops(
            n_src_tokens=0,
            n_tgt_tokens=0,
            src_seq_len=128,
            tgt_seq_len=128,
            model_dim=512,
            enc_layers=6,
            dec_layers=6,
            vocab_size=32000,
        )
        assert flops == 0


class TestStatisticsTflops:
    def test_tflops_computation(self):
        """Test that Statistics.tflops() returns correct value."""
        stat = Statistics()
        stat.flops_per_step = 1e12  # 1 TFLOP

        # Manually set start_time so elapsed_time is predictable
        stat.start_time = time.time() - 1.0  # 1 second ago

        tflops = stat.tflops()
        # 1e12 flops / ~1.0 sec / 1e12 = ~1.0 TFLOP/s
        assert 0.9 < tflops < 1.1, f"Expected ~1.0 TFLOP/s, got {tflops}"

    def test_tflops_zero_flops(self):
        """Zero flops_per_step should return 0."""
        stat = Statistics()
        assert stat.tflops() == 0.0

    def test_update_sums_flops(self):
        """Statistics.update() should sum flops_per_step."""
        stat1 = Statistics()
        stat1.flops_per_step = 100

        stat2 = Statistics()
        stat2.flops_per_step = 200

        stat1.update(stat2)
        assert stat1.flops_per_step == 300
