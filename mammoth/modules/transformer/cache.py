from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class LayerCache:
    self_k:  Optional[Tensor] = None  # (batch, heads, seq, dim_head)
    self_v:  Optional[Tensor] = None
    cross_k: Optional[Tensor] = None  # constant after first decode step
    cross_v: Optional[Tensor] = None


class KVCache:
    """Replaces x-transformers LayerIntermediates / Intermediates."""

    def __init__(self, num_layers: int):
        self.layers: list[LayerCache] = [LayerCache() for _ in range(num_layers)]

    @property
    def batch_size(self) -> Optional[int]:
        """Number of rows (batch * beam) currently held, or None if empty."""
        for layer in self.layers:
            if layer.self_k is not None:
                return layer.self_k.size(0)
            if layer.cross_k is not None:
                return layer.cross_k.size(0)
        return None

    def reorder_beams(self, beam_indices: Tensor, reorder_cross: bool = True) -> None:
        """Reorder cache tensors in-place. Called by decode_strategy.py.

        Cross-attn K/V are identical across the beams of one sentence, so a
        pure within-sentence beam reshuffle can pass ``reorder_cross=False``
        and skip copying them; they only need indexing when rows are dropped.
        """
        for layer in self.layers:
            if layer.self_k is not None:
                layer.self_k = layer.self_k[beam_indices]
                layer.self_v = layer.self_v[beam_indices]
            if reorder_cross and layer.cross_k is not None:
                layer.cross_k = layer.cross_k[beam_indices]
                layer.cross_v = layer.cross_v[beam_indices]
