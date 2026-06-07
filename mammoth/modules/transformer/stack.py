from typing import Optional

import torch.nn as nn
from torch import Tensor

from .block import DecoderBlock
from .cache import KVCache


class TransformerStack(nn.Module):
    """
    Stack of EncoderBlocks or DecoderBlocks for one (layer_stack_index, xcoder_id).
    Carries identity metadata so distributed/components.py can reference it by the
    same keys as before.
    """

    def __init__(
        self,
        blocks: nn.ModuleList,
        final_norm: Optional[nn.Module],
        dim: int,
        layer_stack_index: int,
        xcoder_id: str,
    ):
        super().__init__()
        self.blocks = blocks
        self.final_norm = final_norm
        self.dim = dim
        self.layer_stack_index = layer_stack_index
        self.xcoder_id = xcoder_id

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary: Optional[tuple[Tensor, Tensor]] = None,
        cache: Optional[KVCache] = None,
        cache_offset: int = 0,
    ) -> tuple[Tensor, Optional[KVCache]]:
        for i, block in enumerate(self.blocks):
            layer_cache = cache.layers[cache_offset + i] if cache is not None else None
            if isinstance(block, DecoderBlock):
                x = block(x, context=context, context_mask=context_mask,
                          rotary=rotary, cache=layer_cache)
            else:
                x = block(x, mask=mask, rotary=rotary, cache=layer_cache)
        if self.final_norm is not None:
            x = self.final_norm(x)
        return x, cache

    @property
    def depth(self) -> int:
        return len(self.blocks)

    def get_sub_modules(self) -> dict:
        return dict(self._modules)
