from typing import Optional, Union

import torch.nn as nn
from torch import Tensor

from .rotary import RotaryEmbedding
from .stack import TransformerStack
from .cache import KVCache


class NativeTransformerWrapper(nn.Module):
    """
    token_emb -> post_emb_norm -> TransformerStack(s) -> to_logits.
    post_emb_norm and to_logits are shared per component (xcoder_id tuple),
    not per task -- same as the current x-transformers setup.
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
        self.token_emb = token_emb
        self.post_emb_norm = post_emb_norm
        self.stacks = nn.ModuleList(stacks)
        self.rotary_emb = rotary_emb
        self.to_logits = to_logits
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.return_only_embed = return_only_embed

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
        h = self.emb_dropout(self.post_emb_norm(self.token_emb(x)))
        rotary = self.rotary_emb(h.size(1), h.device) if self.rotary_emb is not None else None
        # Reshape (batch, seq) bool mask → (batch, 1, 1, seq) for SDPA key masking
        if mask is not None and mask.dim() == 2:
            mask = mask[:, None, None, :]
        if context_mask is not None and context_mask.dim() == 2:
            context_mask = context_mask[:, None, None, :]
        for stack in self.stacks:
            h, cache = stack(h, mask=mask, context=context,
                             context_mask=context_mask, rotary=rotary, cache=cache)
        if return_embeddings or self.return_only_embed:
            return h
        logits = self.to_logits(h)
        if return_intermediates:
            return logits, cache
        if return_logits_and_embeddings:
            return logits, h
        return logits
