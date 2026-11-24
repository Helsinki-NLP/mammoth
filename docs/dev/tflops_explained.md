# How TFLOPs/GPU Throughput Is Calculated in MAMMOTH

## What Is a TFLOP/s?

**TFLOP/s** = **T**era **FL**oating-point **O**perations **P**er **S**econd.

- 1 TFLOP = 10^12 floating-point operations.
- If your GPU does 50 TFLOP/s, it means it's performing 50 trillion math operations every second.
- This metric tells you **how efficiently your hardware is being used** during training.

---

## The Big Picture: Why Count FLOPs Analytically?

There are two ways to measure GPU performance:

1. **Profiling** (measure actual hardware counters) -- accurate but adds overhead and is hardware-specific.
2. **Analytical counting** (calculate from model architecture) -- no overhead, portable, widely used.

MAMMOTH uses approach #2, following the **Megatron-LM convention**. The idea is:

> "Given the model architecture, we can **exactly calculate** how many multiply-add operations happen per training step, without ever touching a profiler."

---

## Step-by-Step: What Gets Counted

The calculation lives in `mammoth/utils/flops.py:compute_transformer_flops()`.

### 1. Feed-Forward Network (MLP) -- per token, per layer

A standard transformer FFN has two linear projections:

```
input (dim) --> Linear --> (ff_inner) --> Linear --> output (dim)
```

Each linear layer does a matrix multiply. For a single token:

```
mlp_flops = 4 * model_dim * ff_inner_dim
```

**Where does the 4 come from?**

- First linear: `dim -> ff_inner` = `dim * ff_inner` multiplies
- Second linear: `ff_inner -> dim` = `ff_inner * dim` multiplies
- Each matrix multiply counts as `2 * rows * cols` (one multiply + one add per element)
- So: `2 * dim * ff_inner + 2 * ff_inner * dim = 4 * dim * ff_inner`

If using **GLU** (Gated Linear Unit, common in modern architectures like LLaMA), the inner dimension is multiplied by 1.5x because there's an extra gating projection:

```python
ff_inner_dim = model_dim * ff_mult      # e.g., 512 * 4.0 = 2048
if use_glu:
    ff_inner_dim *= 1.5                  # e.g., 2048 * 1.5 = 3072
```

### 2. Self-Attention -- per token, per layer

Self-attention has two parts:

**a) QKV + Output projections (four linear layers):**

```
Q = input @ W_Q    (dim -> dim)
K = input @ W_K    (dim -> dim)
V = input @ W_V    (dim -> dim)
out = attn @ W_O   (dim -> dim)
```

Each is a `dim x dim` matrix multiply, and there are 4 of them:

```
projection_flops = 4 * model_dim * model_dim
```

(Same logic as MLP: each matmul = `2 * dim * dim`, times 4 projections, but we can simplify: `4 * 2 * dim * dim / 2` ... actually let's be precise: each matmul on a single token is `2 * dim * dim` operations, 4 projections = `8 * dim * dim`. But the code writes `4 * dim * dim` because the `2x` FMA factor is applied globally at the end -- see Step 5.)

**b) Attention score computation:**

```
scores = Q @ K^T    (seq_len x dim) @ (dim x seq_len) = seq_len x seq_len
```

Per token, this costs `2 * dim * seq_len` (one dot product of length `dim` against each position in the sequence). Again, the factor of 2 from FMA is applied globally.

The code writes it as:

```python
enc_self_attn_flops = 4 * model_dim * model_dim + 2 * model_dim * src_seq_len
```

**Note:** For the decoder, self-attention uses `tgt_seq_len`, while cross-attention uses `src_seq_len`.

### 3. Cross-Attention (Decoder Only) -- per token, per layer

Same structure as self-attention, but the Keys and Values come from the encoder output:

```python
cross_attn_flops = 4 * model_dim * model_dim + 2 * model_dim * src_seq_len
```

### 4. Logit Projection -- per target token (once, not per layer)

The final layer projects the decoder output to vocabulary size to produce logits:

```
logits = decoder_output @ W_vocab    (dim -> vocab_size)
```

```python
logit_flops_per_token = 2 * model_dim * vocab_size
```

### 5. Putting It All Together

```python
# Per-token costs, summed across all layers
encoder_flops_per_token = enc_layers * (mlp_flops + enc_self_attn_flops)
decoder_flops_per_token = dec_layers * (mlp_flops + dec_self_attn_flops + cross_attn_flops)

# Total across all tokens in the batch
total_flops = (
    n_src_tokens * encoder_flops_per_token
    + n_tgt_tokens * (decoder_flops_per_token + logit_flops_per_token)
)

# Apply the global multiplier
total_flops *= 6
```

### The Magic Number: Why Multiply by 6?

This `6x` factor is two things combined:

| Factor | Reason |
|--------|--------|
| **3x** | Training does 3 passes: forward pass, backward pass for weight gradients (wgrad), backward pass for data/input gradients (dgrad). Each involves roughly the same matrix multiplies. |
| **2x** | **FMA (Fused Multiply-Add)** factor. GPUs execute `a * b + c` as a single instruction, but it counts as **2** floating-point operations (one multiply, one add). Every matrix multiply element is an FMA. |

So: `3 (fwd + bwd_wgrad + bwd_dgrad) * 2 (FMA) = 6`

This is the standard Megatron-LM convention used across the industry.

---

## How It Flows Through Training

Here's how the pieces connect at runtime:

### 1. Configuration (`mammoth/trainer.py:108-152`)

When the trainer is built, model config is extracted and stored:

```python
flops_config = {
    'model_dim': opts.model_dim,
    'enc_layers': sum(opts.enc_layers),
    'dec_layers': sum(opts.dec_layers),
    'vocab_size': vocab_size,
    'ff_mult': ff_mult,
    'use_glu': use_glu,
}
```

### 2. Per-Step Calculation (`mammoth/trainer.py:1030-1044`)

After each training step, actual batch statistics feed the formula:

```python
if self.report_tflops and self.flops_config.get('model_dim', 0) > 0:
    n_src = report_stats.n_src_words
    n_tgt = report_stats.n_words
    batch_size = report_stats.n_sents
    src_seq_len = n_src // batch_size
    tgt_seq_len = n_tgt // batch_size

    report_stats.flops_per_step = compute_transformer_flops(
        n_src_tokens=n_src,
        n_tgt_tokens=n_tgt,
        src_seq_len=src_seq_len,
        tgt_seq_len=tgt_seq_len,
        **self.flops_config,
    )
```

### 3. Conversion to TFLOP/s (`mammoth/utils/statistics.py:183-187`)

```python
def tflops(self):
    return self.flops_per_step / elapsed_time / 1e12
```

Dividing total FLOPs by wall-clock time gives **operations per second**, then dividing by 10^12 converts to **tera**-operations per second.

### 4. Reporting (`mammoth/utils/statistics.py:209-219`)

The value shows up in the training log and TensorBoard:

```
Step 100; acc: 45.23; ppl: 12.50; xent: 2.53; 23.45 TFLOP/s; 1500/1200 tok/s; ...
```

---

## A Concrete Example

Using the test values from `tests/test_flops.py`:

| Parameter | Value |
|-----------|-------|
| model_dim | 512 |
| enc_layers | 6 |
| dec_layers | 6 |
| vocab_size | 32,000 |
| ff_mult | 4.0 |
| n_src_tokens | 1,024 |
| n_tgt_tokens | 1,024 |
| src/tgt_seq_len | 128 |

**Step-by-step:**

```
ff_inner = 512 * 4.0 = 2,048

mlp_flops = 4 * 512 * 2048 = 4,194,304

enc_self_attn = 4 * 512 * 512 + 2 * 512 * 128
             = 1,048,576 + 131,072
             = 1,179,648

(dec_self_attn = same = 1,179,648)
(cross_attn    = same = 1,179,648)

enc_per_token = 6 * (4,194,304 + 1,179,648) = 32,243,712
dec_per_token = 6 * (4,194,304 + 1,179,648 + 1,179,648) = 39,321,600
logit_per_token = 2 * 512 * 32,000 = 32,768,000

total = 1024 * 32,243,712 + 1024 * (39,321,600 + 32,768,000)
      = 33,017,561,088 + 73,819,750,400
      = 106,837,311,488

total * 6 = 641,023,868,928  (~641 GFLOPs per step)
```

If this step took 1 second: **641 GFLOP / 1s = 0.641 TFLOP/s per GPU**.

---

## What's NOT Counted

This analytical method intentionally skips:

- **LayerNorm / RMSNorm** -- negligible compared to matrix multiplies
- **Activation functions** (ReLU, GELU, SiLU) -- negligible
- **Dropout** -- negligible
- **Embedding lookups** -- not matrix multiplies
- **Softmax in attention** -- negligible
- **Communication overhead** (all-reduce, etc.) -- not compute

This is standard practice. The matrix multiplies dominate (>99% of FLOPs), so counting only those gives a reliable estimate.

---

**Key insight:** TFLOP/s is most useful as a **relative** metric. The absolute number depends heavily on model size, precision, and hardware.
