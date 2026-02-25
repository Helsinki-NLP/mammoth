"""
MAMMOTH HuggingFace Integration

Bidirectional conversion between MAMMOTH and HuggingFace formats.

Structure:
    - to_hf/: Convert MAMMOTH → HuggingFace format
    - from_hf/: Convert HuggingFace → MAMMOTH format

Quick Start:

    # Convert HF model to MAMMOTH
    from mammoth.hf_integration.from_hf import convert_hf_to_mammoth
    convert_hf_to_mammoth("facebook/bart-base", "./models/mammoth_bart")

    # Convert MAMMOTH model to HF
    from mammoth.hf_integration.to_hf import convert_mammoth_to_hf_custom
    convert_mammoth_to_hf_custom("./checkpoints/my_model", "./hf_model")
"""

# Legacy compatibility - keep AutoModel registration for to_hf functionality
try:
    from transformers import AutoConfig, AutoModelForSeq2SeqLM
    from mammoth.hf_integration.to_hf.configuration_mammoth import MammothConfig
    from mammoth.hf_integration.to_hf.modeling_mammoth import MammothForConditionalGeneration, MammothPreTrainedModel

    AutoConfig.register("mammoth", MammothConfig)
    AutoModelForSeq2SeqLM.register(MammothConfig, MammothForConditionalGeneration)

    __all__ = [
        "MammothConfig",
        "MammothForConditionalGeneration",
        "MammothPreTrainedModel",
    ]
except ImportError:
    # HuggingFace not installed
    __all__ = []