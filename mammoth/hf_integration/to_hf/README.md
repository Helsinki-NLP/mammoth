# MAMMOTH HuggingFace Integration

Convert MAMMOTH models to HuggingFace format without weight conversion, preserving the original architecture.

## Quick Start

```bash
# 1. Convert a MAMMOTH checkpoint
cd /Users/chaowang/Desktop/Mammoth/mammoth/hf_integration
python convert_to_hf_custom.py <checkpoint_path> <output_dir> --task <task_id>

# 2. Test the converted model
python test_hf_custom_model.py <output_dir>

# 3. Use it!
```

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(
    "./output_dir",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("./output_dir")

# Translate
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Files

- **`configuration_mammoth.py`** - HuggingFace config class
- **`modeling_mammoth.py`** - HuggingFace model wrapper
- **`convert_to_hf_custom.py`** - Conversion script
- **`test_hf_custom_model.py`** - Test suite
- **`HF_CUSTOM_MODEL_GUIDE.md`** - Full documentation

## How It Works

This approach **wraps** the native MAMMOTH model instead of converting weights:

```
HuggingFace API
    ↓
Format Conversion (batch,seq) ↔ (seq,batch)
    ↓
Native MAMMOTH Model (unchanged)
```

**Benefits:**
- ✅ No embedding mismatches
- ✅ Identical behavior to native MAMMOTH
- ✅ Full HuggingFace ecosystem support
- ✅ Easy to maintain

## Examples

### Basic Conversion
```bash
python convert_to_hf_custom.py \
    ~/checkpoints/mammoth_model \
    ~/hf_models/mammoth-en-es \
    --task en-es
```

### Push to Hub
```bash
python convert_to_hf_custom.py \
    ~/checkpoints/mammoth_model \
    ~/hf_models/mammoth-en-es \
    --task en-es \
    --push-to-hub \
    --hub-model-id username/mammoth-en-es
```

### Load Programmatically
```python
from mammoth.hf_integration import MammothForConditionalGeneration

# Load directly from checkpoint
model = MammothForConditionalGeneration.from_mammoth_checkpoint(
    checkpoint_path="~/checkpoints/mammoth_model",
    task_id="en-es"
)

# Save as HuggingFace format
model.save_pretrained("~/hf_models/mammoth-en-es")
```

## Documentation

See **`HF_CUSTOM_MODEL_GUIDE.md`** for comprehensive documentation including:
- Detailed architecture explanation
- Advanced generation options
- Multi-task model handling
- Troubleshooting guide

## Comparison: Weight Conversion vs Wrapper

| Aspect | BART Conversion | Custom Wrapper |
|--------|----------------|----------------|
| Embedding Issues | Common ❌ | Never ✅ |
| Maintenance | Brittle ❌ | Robust ✅ |
| Original Features | Lost ❌ | Preserved ✅ |
| Setup Complexity | High ❌ | Low ✅ |
| HF Features | All ✅ | All ✅ |
