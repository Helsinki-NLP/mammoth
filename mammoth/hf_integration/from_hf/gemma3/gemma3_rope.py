"""Read Gemma3's two RoPE base frequencies from a HF config, across transformers versions.

transformers 4.x stores them as flat attributes: ``rope_theta`` (global/full-attention layers) and
``rope_local_base_freq`` (local/sliding-attention layers).
transformers 5.x nests them per layer type in ``rope_parameters``:
``{"full_attention": {"rope_theta": ...}, "sliding_attention": {"rope_theta": ...}}``.
"""


def get_gemma3_rope_thetas(config):
    """Return ``(global_theta, local_theta)`` for a ``Gemma3TextConfig``."""
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        full = rope_parameters.get("full_attention")
        sliding = rope_parameters.get("sliding_attention")
        if isinstance(full, dict) and isinstance(sliding, dict):
            return full["rope_theta"], sliding["rope_theta"]

    global_theta = getattr(config, "rope_theta", None)
    local_theta = getattr(config, "rope_local_base_freq", None)
    if global_theta is None or local_theta is None:
        raise ValueError(
            "Could not find Gemma3 RoPE thetas: expected config.rope_parameters with "
            "'full_attention' and 'sliding_attention' entries (transformers 5.x), or "
            "config.rope_theta and config.rope_local_base_freq (transformers 4.x)."
        )
    return global_theta, local_theta
