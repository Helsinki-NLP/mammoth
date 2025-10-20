# ModernBERT-Style Unpadding for MAMMOTH

## Overview

This directory contains the implementation of ModernBERT-style unpadding for efficient attention computation with variable-length sequences. Unpadding removes padding tokens before processing, significantly improving performance when dealing with sequences of varying lengths.

## Implementation Status

### ✅ Completed
- **`bert_padding.py`**: Core unpadding/repadding functions
  - `unpad_input()`: Removes padding tokens and returns metadata
  - `pad_input()`: Restores padding after processing
  - `unpad_input_only()`: Lightweight version for simple cases
  - Custom autograd functions for efficient gradient computation

### 🔧 Partially Integrated
- **`x_transformers.py`**: Attention class prepared for unpadding
  - Added `use_unpadding` parameter to `Attention` class
  - Unpadding logic integrated but **requires flash attention for full functionality**

## Current Limitations

The unpadding technique as implemented in ModernBERT relies on:
1. **Flash Attention**: Understands variable-length sequences via `cu_seqlens` format
2. **Custom CUDA kernels**: Handle the flattened token representation efficiently

MAMMOTH's current attention implementation:
- Uses standard PyTorch operations that expect regular batch structure
- Does not support the `cu_seqlens` (cumulative sequence lengths) format
- Would require significant refactoring to support unpadded attention

## How Unpadding Works

### Input Processing
```python
# Before: (batch=3, seqlen=6, dim=768) with padding
# Sequence lengths: [5, 3, 4] (padded to 6)
hidden_states = torch.randn(3, 6, 768)
attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 0],  # 5 valid tokens
    [1, 1, 1, 0, 0, 0],  # 3 valid tokens
    [1, 1, 1, 1, 0, 0],  # 4 valid tokens
])

# After unpadding: (total_tokens=12, dim=768) - no padding!
unpadded, indices, cu_seqlens, max_seqlen = unpad_input(hidden_states, attention_mask)
# unpadded.shape: (12, 768)  # 5+3+4 = 12 valid tokens
# cu_seqlens: [0, 5, 8, 12]  # cumulative indices
# max_seqlen: 5
```

### Attention Computation
For flash attention with unpadding:
```python
# Flash attention uses cu_seqlens to know sequence boundaries
output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens,
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen
)
```

### Output Restoration
```python
# Restore original batch structure
output = pad_input(unpadded_output, indices, batch=3, seqlen=6)
# output.shape: (3, 6, 768) - back to padded format
```

## Performance Benefits

Unpadding provides significant speedups when:
- Sequences have variable lengths
- Average sequence length is much shorter than max length
- Batch sizes are large

**Example**: If average sequence length is 50% of max length, unpadding can provide ~2x speedup by skipping computation on padding tokens.

## Future Integration Path

To fully integrate unpadding into MAMMOTH:

### Option 1: Flash Attention Integration (Recommended)
1. Add flash-attention as a dependency
2. Update `Attend` class to support variable-length format
3. Enable `use_unpadding=True` with `flash=True`

### Option 2: Custom Attention Implementation
1. Implement attention kernel that understands `cu_seqlens`
2. Add support for variable-length batches throughout attention pipeline
3. Update position embeddings to work with unpadded sequences

### Option 3: Per-Sample Processing (Simple but slower)
1. Process each sequence individually (no padding needed)
2. Batch results after attention
3. Trade parallelism for simplicity

## Usage Example (When Fully Supported)

```python
from mammoth.x_transformers import Attention

# Create attention layer with unpadding enabled
attn = Attention(
    dim=768,
    heads=12,
    flash=True,  # Required for unpadding
    use_unpadding=True,  # Enable ModernBERT-style unpadding
)

# Forward pass
output = attn(
    x,  # (batch, seq_len, dim)
    mask=attention_mask  # (batch, seq_len) - 1=valid, 0=padding
)
# Unpadding happens automatically inside!
```

## Configuration

The unpadding feature can be controlled via model configuration:

```yaml
# In your MAMMOTH config YAML
model_opts:
  transformer:
    attn_use_unpadding: true  # Enable unpadding
    attn_flash: true  # Required for unpadding
```

## References

- **ModernBERT Paper**: https://arxiv.org/abs/2412.13663
- **Flash Attention**: https://arxiv.org/abs/2205.14135
- **Original Implementation**: https://github.com/HazyResearch/flash-attention

## Notes

- Unpadding is most beneficial with highly variable sequence lengths
- Consider the memory/speed tradeoff: unpadding adds overhead for homogeneous batches
- Works best with modern GPUs (A100, H100) that have optimized flash attention kernels