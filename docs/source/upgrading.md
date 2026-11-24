# Upgrading from 0.2

Mammoth 0.2 built its transformer layers on a bundled copy of the [x-transformers](https://github.com/lucidrains/x-transformers) library. Since 0.3, Mammoth uses its own small transformer implementation built from standard `torch.nn` modules (`mammoth/modules/transformer/`). Multi-task training, parameter sharing (`enc_sharing_group` / `dec_sharing_group`) and the rest of the YAML structure are unchanged.

## Config changes

Most options are unchanged. There are two differences:

1. The `x_transformers_opts` block is gone. Its keys are now top-level options, at the same level as `model_dim` and `enc_layers`.
2. The supported architecture options are narrower:
   - **Positional embedding:** RoPE only.
   - **Normalization:** RMSNorm only, without bias.
   - **Feed-forward activation:** `swiglu` (default) or `gelu`.

**Old config (0.2, no longer valid):**

```yaml
enc_layers: [6]
dec_layers: [6]
model_dim: 1024
model_dtype: bf16

x_transformers_opts:
  attn_flash: true
  rotary_pos_emb: true
  tie_embedding: false
  heads: 16
  pre_norm: true
  post_emb_norm: true
  post_emb_norm_bias: false
  attn_dropout: 0.1
  ff_dropout: 0.1
  layernorm_bias: false
  use_abs_pos_emb: false
  use_fused_rmsnorm: true
```

**New config:**

```yaml
enc_layers: [6]
dec_layers: [6]
model_dim: 1024
model_dtype: bf16

heads: 16              # attention heads (model_dim must be divisible by heads)
rotary_pos_emb: true   # rotary positional embeddings (RoPE)
post_emb_norm: true    # RMSNorm after the token embedding
attn_dropout: 0.1
ff_dropout: 0.1
ff_activation: swiglu  # "swiglu" (default) or "gelu"
```

The inference config needs no architecture options: they are read from the checkpoint.

## Removed options

| Old key | Why it is gone |
|---|---|
| `attn_flash` | Attention always uses `torch.nn.functional.scaled_dot_product_attention`, which picks FlashAttention or memory-efficient attention automatically when the hardware supports it |
| `use_abs_pos_emb` | RoPE is the only supported positional embedding |
| `tie_embedding` | Tied input/output embeddings are not supported |
| `layernorm_bias` | All norms are bias-free |
| `use_fused_rmsnorm` | `nn.RMSNorm` uses PyTorch's fused kernel automatically |
| `pre_norm` | Pre-norm is always on |
| `post_emb_norm_bias` | The post-embedding norm is always bias-free |
