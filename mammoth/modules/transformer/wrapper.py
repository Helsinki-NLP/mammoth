from typing import Optional, Union

import torch.nn as nn
from torch import Tensor

from .rotary import RotaryEmbedding
from .stack import TransformerStack
from .cache import KVCache


class NativeTransformerWrapper(nn.Module):
    """
    token_emb -> post_emb_norm -> TransformerStack(s) -> to_logits.

    This wrapper is a lightweight routing view. All shared parameters
    (stacks, token_emb, post_emb_norm, rotary_emb, to_logits) are owned
    by StackXcoder.shared_* and stored here as plain Python references so
    PyTorch does not re-register them as children of this module.
    Only emb_dropout is per-task and registered normally.
    """

    def __init__(
        self,
        token_emb: nn.Embedding,
        post_emb_norm: nn.Module,
        stacks: list[TransformerStack],
        rotary_emb: Optional[RotaryEmbedding],
        to_logits: Optional[nn.Linear],
        emb_dropout: float = 0.0,
        return_only_embed: bool = False,
    ):
        super().__init__()
        # Bypass nn.Module.__setattr__ so these are not registered as children.
        # StackXcoder.shared_* owns these; .to()/.train()/.eval() reach them
        # through StackXcoder, not through this wrapper.
        object.__setattr__(self, '_token_emb', token_emb)
        object.__setattr__(self, '_post_emb_norm', post_emb_norm)
        object.__setattr__(self, '_stacks', stacks)
        object.__setattr__(self, '_rotary_emb', rotary_emb)
        object.__setattr__(self, '_to_logits', to_logits)
        object.__setattr__(self, '_return_only_embed', return_only_embed)
        # Per-task module — not shared, registered normally.
        self.emb_dropout = nn.Dropout(emb_dropout)

    @property
    def can_cache_kv(self) -> bool:
        return True

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        return_embeddings: bool = False,
        return_logits_and_embeddings: bool = False,
        return_attn: bool = False,
        return_intermediates: bool = False,
        cache: Optional[KVCache] = None,
        seq_start_pos: Optional[int] = None,
    ) -> Union[Tensor, tuple[Tensor, KVCache]]:
        h = self.emb_dropout(self._post_emb_norm(self._token_emb(x)))
        rotary = self._rotary_emb(h.size(1), h.device) if self._rotary_emb is not None else None
        # Reshape (batch, seq) bool mask → (batch, 1, 1, seq) for SDPA key masking
        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, None, :]
        if context_mask is not None and context_mask.dim() == 2:
            context_mask = context_mask[:, None, None, :]
        for stack in self._stacks:
            h, cache = stack(h, mask=mask, context=context,
                             context_mask=context_mask, rotary=rotary, cache=cache)
        if return_embeddings or self._return_only_embed:
            return h
        logits = self._to_logits(h)
        if return_intermediates:
            return logits, cache
        if return_logits_and_embeddings:
            return logits, h
        return logits
