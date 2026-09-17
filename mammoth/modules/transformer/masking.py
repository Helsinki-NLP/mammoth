import torch
from torch import Tensor


def build_sliding_window_causal_mask(seq_len: int, window: int, device: torch.device) -> Tensor:
    """
    Boolean attention mask combining causality with a fixed-size sliding
    window, as used by Gemma3's local attention layers: query position i may
    attend to key position j iff j <= i and i - j < window.

    Returns a (seq_len, seq_len) bool tensor, True = allowed to attend.
    """
    positions = torch.arange(seq_len, device=device)
    query = positions.unsqueeze(1)
    key = positions.unsqueeze(0)
    causal = key <= query
    within_window = (query - key) < window
    return causal & within_window
