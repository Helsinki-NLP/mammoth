"""Analytical FLOP counting for encoder-decoder transformers.

Follows the Megatron-LM approach: count matrix multiply FLOPs analytically,
then apply a 3x factor (forward + backward wgrad + backward dgrad)
and a 2x FMA factor (each fused multiply-add = 2 FLOPs).
"""


def compute_transformer_flops(
    n_src_tokens: int,
    n_tgt_tokens: int,
    src_seq_len: int,
    tgt_seq_len: int,
    model_dim: int,
    enc_layers: int,
    dec_layers: int,
    vocab_size: int,
    ff_mult: float = 4.0,
    use_glu: bool = False,
) -> int:
    """Compute total FLOPs (forward + backward) for one training step.

    Args:
        n_src_tokens: Total source tokens in the batch.
        n_tgt_tokens: Total target tokens in the batch.
        src_seq_len: Source sequence length (for attention score term).
        tgt_seq_len: Target sequence length (for attention score term).
        model_dim: Hidden dimension of the transformer (d_model).
        enc_layers: Number of encoder layers.
        dec_layers: Number of decoder layers.
        vocab_size: Size of the target vocabulary (for logit projection).
        ff_mult: FFN inner dimension multiplier (ff_inner = model_dim * ff_mult).
            Read from x_transformers_opts in the training config; defaults to 4.0.
        use_glu: If True, GLU-style FFN multiplies inner dim by 1.5.
            Read from x_transformers_opts (ff_glu) in the training config.

    Returns:
        Total FLOPs as an integer (forward + backward, with FMA factor).
    """
    if n_src_tokens == 0 and n_tgt_tokens == 0:
        return 0

    ff_inner_dim = model_dim * ff_mult
    if use_glu:
        ff_inner_dim *= 1.5

    # Per-token FLOPs per layer (before the 2x FMA and 3x fwd+bwd factors)
    # MLP: two linear projections  →  4 * dim * ff_inner_dim
    mlp_flops = 4 * model_dim * ff_inner_dim

    # Self-attention: QKV proj + output proj = 4 * dim * dim
    #                 attention scores      = 2 * dim * seq_len
    enc_self_attn_flops = 4 * model_dim * model_dim + 2 * model_dim * src_seq_len
    dec_self_attn_flops = 4 * model_dim * model_dim + 2 * model_dim * tgt_seq_len

    # Cross-attention (decoder only): same structure but with src_seq_len
    cross_attn_flops = 4 * model_dim * model_dim + 2 * model_dim * src_seq_len

    # Per-token FLOPs aggregated over all layers
    encoder_flops_per_token = enc_layers * (mlp_flops + enc_self_attn_flops)
    decoder_flops_per_token = dec_layers * (mlp_flops + dec_self_attn_flops + cross_attn_flops)

    # Logit projection: 2 * dim * vocab_size per target token
    logit_flops_per_token = 2 * model_dim * vocab_size

    total_flops = (
        n_src_tokens * encoder_flops_per_token
        + n_tgt_tokens * (decoder_flops_per_token + logit_flops_per_token)
    )

    # 3x: forward + backward_wgrad + backward_dgrad
    # 2x: FMA factor (each GEMM multiply-add = 2 FLOPs)
    total_flops *= 6

    return int(total_flops)
