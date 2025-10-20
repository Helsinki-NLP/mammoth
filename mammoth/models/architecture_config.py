"""
Model architecture configuration registry.

This module defines architecture-specific parameters for different transformer models
(BART, GPT-2, T5, etc.) to ensure compatibility when using the x-transformers backend.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelArchitectureConfig:
    """
    Configuration for transformer architecture-specific parameters.

    These parameters control how the x-transformers modules are initialized
    to match the architecture of specific pre-trained models.
    """

    # Positional embedding configuration
    positional_embedding_scale: Optional[float] = None
    """
    Scale factor for positional embeddings.
    - BART: 1.0 (no scaling)
    - GPT-2: None (uses default dim**-0.5)
    - T5: Uses relative positional bias instead
    """

    # Attention layer configuration
    qkv_bias: bool = True
    """Whether to include bias in Q, K, V projection layers"""

    attention_out_bias: bool = True
    """Whether to include bias in attention output projection"""

    # Normalization configuration
    layer_norm_bias: bool = True
    """Whether to include bias in LayerNorm layers"""

    layer_norm_eps: float = 1e-5
    """Epsilon value for LayerNorm numerical stability"""

    # Activation function
    activation_function: str = "gelu"
    """
    Activation function to use in feed-forward layers.
    Options: 'gelu', 'relu', 'swish', 'relu_squared'
    """

    # Feed-forward configuration
    feedforward_glu: bool = False
    """Whether to use GLU (Gated Linear Unit) in feed-forward layers"""

    # Initialization
    parameter_init_std: Optional[float] = None
    """Standard deviation for parameter initialization (if specified)"""

    def to_attention_kwargs(self) -> Dict[str, Any]:
        """
        Convert config to kwargs for x_transformers.Attention.__init__

        Note: Current x_transformers uses LinearNoBias for all attention projections,
        so qkv_bias and out_bias parameters are not applicable and are ignored.
        Kept for backward compatibility but have no effect.
        """
        kwargs = {}

        # Note: Activation functions in attention are not supported in current x_transformers
        # The parameters below are ignored but kept for potential future extensions

        return kwargs

    def to_feedforward_kwargs(self) -> Dict[str, Any]:
        """
        Convert config to kwargs for x_transformers.FeedForward.__init__

        Note: Parameters must be prefixed with 'ff_' for x-transformers routing.
        The groupby_prefix_and_trim function extracts these before passing to FeedForward.
        """
        kwargs = {
            'ff_glu': self.feedforward_glu,
            'ff_no_bias': not self.layer_norm_bias,  # Inverse logic: no_bias = not has_bias
        }

        # Activation function
        if self.activation_function == 'gelu':
            # Default, no special kwargs needed
            pass
        elif self.activation_function == 'relu_squared':
            kwargs['ff_relu_squared'] = True
        elif self.activation_function == 'swish':
            kwargs['ff_swish'] = True

        return kwargs

    def to_layer_norm_kwargs(self) -> Dict[str, Any]:
        """
        Convert config to kwargs for LayerNorm initialization

        Note: Current x_transformers LayerNorm is always bias-free (elementwise_affine=False).
        The layer_norm_bias config parameter is kept for compatibility but has no effect.
        """
        return {}

    def to_positional_embedding_kwargs(self, dim: int) -> Dict[str, Any]:
        """
        Convert config to kwargs for positional embedding initialization

        Args:
            dim: Model dimension (needed for scale calculation if not specified)
        """
        kwargs = {}

        if self.positional_embedding_scale is not None:
            kwargs['scale'] = self.positional_embedding_scale
        # Otherwise uses default from AbsolutePositionalEmbedding

        return kwargs


# =============================================================================
# Model-Specific Configurations
# =============================================================================

BART_CONFIG = ModelArchitectureConfig(
    positional_embedding_scale=1.0,  # BART uses unscaled positional embeddings
    qkv_bias=True,
    attention_out_bias=True,
    layer_norm_bias=True,
    layer_norm_eps=1e-5,
    activation_function="gelu",
    feedforward_glu=False,
)

GPT2_CONFIG = ModelArchitectureConfig(
    positional_embedding_scale=None,  # Uses default dim**-0.5 scaling
    qkv_bias=False,  # GPT-2 does not use bias in attention projections
    attention_out_bias=True,
    layer_norm_bias=True,
    layer_norm_eps=1e-5,
    activation_function="gelu",
    feedforward_glu=False,
)

T5_CONFIG = ModelArchitectureConfig(
    positional_embedding_scale=None,  # T5 uses relative position bias, not absolute
    qkv_bias=False,  # T5 does not use bias in QKV
    attention_out_bias=False,  # T5 does not use bias in output projection
    layer_norm_bias=False,  # T5 uses RMSNorm (bias-less)
    layer_norm_eps=1e-6,  # T5 uses smaller epsilon
    activation_function="relu",  # T5 uses ReLU, not GELU
    feedforward_glu=True,  # T5 uses gated activation (GLU variant)
)

LLAMA_CONFIG = ModelArchitectureConfig(
    positional_embedding_scale=None,  # LLaMA uses RoPE (rotary position embeddings)
    qkv_bias=False,
    attention_out_bias=False,
    layer_norm_bias=False,  # Uses RMSNorm
    layer_norm_eps=1e-5,
    activation_function="swish",  # SwiGLU activation
    feedforward_glu=True,
)

MODERNBERT_CONFIG = ModelArchitectureConfig(
    positional_embedding_scale=None,  # ModernBERT uses standard scaling
    qkv_bias=False,  # ModernBERT uses bias-free projections (configurable)
    attention_out_bias=False,  # No bias in output projection
    layer_norm_bias=False,  # ModernBERT supports LayerNorm bias (configurable)
    layer_norm_eps=1e-5,
    activation_function="gelu",  # GELU activation
    feedforward_glu=True,  # ModernBERT uses GLU in MLP
)

# Registry mapping model type strings to configurations
MODEL_ARCHITECTURE_REGISTRY = {
    'bart': BART_CONFIG,
    'gpt2': GPT2_CONFIG,
    'gpt-2': GPT2_CONFIG,  # Alias
    't5': T5_CONFIG,
    'llama': LLAMA_CONFIG,
    'modernbert': MODERNBERT_CONFIG,
    # Add more as needed
}


def get_model_architecture_config(model_type: str) -> ModelArchitectureConfig:
    """
    Get the architecture configuration for a specific model type.

    Args:
        model_type: Model type identifier (e.g., 'bart', 'gpt2', 't5')

    Returns:
        ModelArchitectureConfig for the specified model type

    Raises:
        ValueError: If model_type is not in the registry
    """
    model_type_lower = model_type.lower()

    if model_type_lower not in MODEL_ARCHITECTURE_REGISTRY:
        available = ', '.join(sorted(MODEL_ARCHITECTURE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available types: {available}"
        )

    return MODEL_ARCHITECTURE_REGISTRY[model_type_lower]