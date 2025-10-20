# RoPE Theta Quick Reference Guide

## TL;DR

✅ **Your implementation now supports per-layer RoPE theta values**
✅ **Backward compatible with standard single-theta RoPE**
✅ **All tests pass for ModernBERT configuration**

## Quick Comparison

### Before (Incorrect for ModernBERT)
```python
# ❌ All layers used single theta=10000
encoder = Encoder(
    dim=768,
    depth=22,
    rotary_pos_emb=True,
    # Missing: global_rope_theta, local_rope_theta
)
# Result: All layers had theta=10000 (default)
```

### After (Correct for ModernBERT)
```python
# ✅ Layers use different theta based on attention pattern
encoder = Encoder(
    dim=768,
    depth=22,
    rotary_pos_emb=True,
    global_rope_theta=160000.0,      # Global attention layers
    local_rope_theta=10000.0,         # Local attention layers
    global_attn_every_n_layers=3,
    sliding_window=128,
)
# Result:
#   Layers 0,3,6,9,12,15,18,21: theta=160000
#   Layers 1,2,4,5,7,8,...:     theta=10000
```

## Common Use Cases

### 1. ModernBERT (Global/Local Attention)
```python
from mammoth.x_transformers import Encoder

encoder = Encoder(
    dim=768,
    depth=22,
    heads=12,
    rotary_pos_emb=True,
    global_rope_theta=160000.0,
    local_rope_theta=10000.0,
    global_attn_every_n_layers=3,
    sliding_window=128,
)
```

### 2. Standard Transformer (Single Theta)
```python
# Option A: Set both thetas to same value
encoder = Encoder(
    dim=768,
    depth=12,
    heads=12,
    rotary_pos_emb=True,
    global_rope_theta=10000.0,
    local_rope_theta=10000.0,
)

# Option B: Use defaults (same result)
encoder = Encoder(
    dim=768,
    depth=12,
    heads=12,
    rotary_pos_emb=True,
    # global_rope_theta defaults to 160000.0
    # local_rope_theta defaults to 10000.0
    # But without global_attn_every_n_layers, all use local_rope_theta
)
```

### 3. Custom Theta for Long Context
```python
encoder = Encoder(
    dim=768,
    depth=12,
    heads=12,
    rotary_pos_emb=True,
    global_rope_theta=500000.0,  # Very long context
    local_rope_theta=500000.0,
)
```

## Configuration Matrix

| Pattern | `global_rope_theta` | `local_rope_theta` | `global_attn_every_n_layers` | Result |
|---------|---------------------|--------------------|-----------------------------|--------|
| ModernBERT | 160000.0 | 10000.0 | 3 | Layers 0,3,6,... use 160000, others use 10000 |
| Standard | 10000.0 | 10000.0 | -1 (default) | All layers use 10000 |
| Long Context | 100000.0 | 100000.0 | -1 (default) | All layers use 100000 |
| Custom Pattern | 200000.0 | 50000.0 | 4 | Layers 0,4,8,... use 200000, others use 50000 |

## Verification Commands

```bash
# Test per-layer theta (6 layers)
python test_rope_theta.py

# Test complete ModernBERT config (22 layers)
python test_rope_and_attention_pattern.py

# Test fallback to standard RoPE
python test_rope_fallback.py
```

## Expected Output (ModernBERT-22)

```
Layer 0  : Global (theta=160000.0, window=(-1,-1))  ✅
Layer 1  : Local  (theta=10000.0,  window=(64,64))  ✅
Layer 2  : Local  (theta=10000.0,  window=(64,64))  ✅
Layer 3  : Global (theta=160000.0, window=(-1,-1))  ✅
...
Layer 21 : Global (theta=160000.0, window=(-1,-1))  ✅
```

## HF Converter Integration

When converting ModernBERT from HuggingFace:

```python
# In hfModernBERT2mammoth.py
xt_model = XTransformer(
    dim=768,
    enc_rotary_pos_emb=True,
    enc_global_rope_theta=config.global_rope_theta,  # 160000.0
    enc_local_rope_theta=config.local_rope_theta,    # 10000.0
    enc_global_attn_every_n_layers=3,
    enc_sliding_window=128,
    ...
)
```

The converter automatically:
1. Extracts theta values from HF config
2. Passes them to XTransformer
3. Stores them in model_opts

## Key Parameters

### `global_rope_theta` (default: 160000.0)
RoPE theta for global attention layers (full attention, no sliding window)

### `local_rope_theta` (default: 10000.0)
RoPE theta for local attention layers (sliding window attention)

### `global_attn_every_n_layers` (default: -1)
- If > 0: Layers at positions 0, N, 2N, ... use global attention
- If -1: All layers use local configuration (standard transformer)

### `sliding_window` (default: -1)
- If > 0: Window size for local attention layers
- If -1: No sliding window (full attention)

## Theta Values Explained

**What is theta?**
- Controls the frequency of positional encodings in RoPE
- Higher theta = longer effective context
- Lower theta = shorter effective context

**Why different thetas?**
- **Global attention** (160000): Needs to capture long-range dependencies
- **Local attention** (10000): Focuses on nearby tokens within window

**Typical values:**
- Standard: 10000 (original RoPE paper)
- Long context: 100000-500000
- ModernBERT global: 160000
- ModernBERT local: 10000

## Debugging Tips

### Check Layer Configuration
```python
encoder = Encoder(...)

# Inspect a specific layer
layer_0_attn = encoder.layers[0][1]  # First attention layer
if hasattr(layer_0_attn, 'rotary_pos_emb'):
    rope = layer_0_attn.rotary_pos_emb
    print(f"Layer 0 has RoPE: {rope is not None}")
    if rope:
        # Check inv_freq to verify theta
        print(f"inv_freq shape: {rope.inv_freq.shape}")
```

### Verify Attention Window
```python
layer_0_attn = encoder.layers[0][1]
window = layer_0_attn.attend.window_size
print(f"Layer 0 window: {window}")
# Global: (-1, -1)
# Local:  (64, 64) for sliding_window=128
```

## Common Issues & Solutions

### Issue: All layers have same theta
**Cause**: Not setting `global_attn_every_n_layers`
**Solution**: Set `global_attn_every_n_layers=3` (or desired pattern)

### Issue: Layers have wrong theta values
**Cause**: Swapped global/local theta values
**Solution**: Check `global_rope_theta > local_rope_theta` for ModernBERT

### Issue: Test failures
**Cause**: Incorrect configuration
**Solution**: Run `python test_rope_and_attention_pattern.py` to diagnose

## Files to Check

- **Implementation**: `mammoth/x_transformers/x_transformers.py`
  - `Attention.__init__()`: Per-layer RoPE creation
  - `AttentionLayers.__init__()`: Theta configuration storage

- **Converter**: `mammoth/hf_integration/from_hf/modernBERT/hfModernBERT2mammoth.py`
  - `create_xtransformer_model()`: Theta extraction

- **Tests**: `test_rope_*.py`
  - Verification scripts

## Summary

✅ **What changed**: Each layer now has its own RoPE instance with layer-specific theta

✅ **Backward compatible**: Standard transformers still work with single theta

✅ **ModernBERT ready**: Correctly implements dual-theta pattern

✅ **Tested**: Comprehensive test suite verifies all configurations

For detailed documentation, see `ROPE_THETA_IMPLEMENTATION.md`