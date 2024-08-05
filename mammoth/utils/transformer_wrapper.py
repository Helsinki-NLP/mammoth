"""
A slightly modified copy of x_transformers.TransformerWrapper.
Enables passing in pre-existing token embeddings.
"""
from __future__ import annotations

import math
import torch

from torch import nn, Tensor
from torch.nn import Module, ModuleList, ModuleDict

from typing import Dict, Optional

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange

from x_transformers.attend import Attend, Intermediates
from x_transformers.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    LayerIntermediates,
    LayerNorm,
    ScaledSinusoidalEmbedding,
    TokenEmbedding,
    Decoder,
    always,
    calc_z_loss,
    default,
    exists,
    pad_at_dim,
)


class TransformerWrapper(Module):
    def __init__(
        self,
        *,
        num_tokens,
        max_seq_len,
        attn_layers: AttentionLayers,
        embed_num_tokens: Dict[str, int] = dict(),
        emb_dim=None,
        max_mem_len=0,
        shift_mem_down=0,
        emb_dropout=0.,
        post_emb_norm=False,
        num_memory_tokens=None,
        memory_tokens_interspersed_every=None,
        tie_embedding=False,
        logits_dim=None,
        num_output_heads=1,
        use_abs_pos_emb=True,
        scaled_sinu_pos_emb=False,
        l2norm_embed=False,
        emb_frac_gradient=1.,  # GLM-130B and Cogview successfully used this, set at 0.1
        attn_z_loss_weight=1e-4,
        token_emb: Optional[TokenEmbedding] = None,
    ):
        super().__init__()

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)
        self.emb_dim = emb_dim
        self.num_tokens = num_tokens

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.l2norm_embed = l2norm_embed
        if token_emb:
            self.token_emb = token_emb
        else:
            self.token_emb = TokenEmbedding(emb_dim, num_tokens, l2norm_embed=l2norm_embed)

        no_abs_pos_emb = max_seq_len == 0 or not (use_abs_pos_emb and not attn_layers.disable_abs_pos_emb)

        if no_abs_pos_emb:
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(emb_dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len, l2norm_embed=l2norm_embed)

        # additional embeddings - say type embedding from BERT

        self.embeds = None

        if len(embed_num_tokens) > 0:
            self.embeds = ModuleDict({
                f'{name}_embed': nn.Embedding(num_tokens, emb_dim)
                for name, num_tokens in embed_num_tokens.items()
            })

        # fraction of the gradient that should go to the embedding, https://arxiv.org/abs/2105.13290

        self.emb_frac_gradient = emb_frac_gradient

        self.post_emb_norm = LayerNorm(emb_dim) if post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers

        self.init_()

        # output head, usually to logits of num_tokens

        logits_dim = default(logits_dim, num_tokens)

        self.has_multiple_heads = False

        if tie_embedding:
            self.to_logits = lambda t: t @ self.token_emb.emb.weight.t()
        elif num_output_heads > 1:
            self.has_multiple_heads = True
            self.to_logits = ModuleList([
                nn.Linear(dim, logits_dim, bias=False)
                for _ in range(num_output_heads)
            ])
        else:
            self.to_logits = nn.Linear(dim, logits_dim, bias=False)

        # memory tokens (like [cls]) from Memory Transformers paper

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        self.memory_tokens_interspersed_every = memory_tokens_interspersed_every

        # whether can do cached kv decoding

        self.can_cache_kv = self.num_memory_tokens == 0
        self.can_cache_kv_outside_max_seq_len = no_abs_pos_emb

    def init_(self):
        if self.l2norm_embed:
            nn.init.normal_(self.token_emb.emb.weight, std=1e-5)
            if not isinstance(self.pos_emb, always):
                nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        nn.init.kaiming_normal_(self.token_emb.emb.weight)

    def forward(
        self,
        x,
        return_embeddings=False,
        return_logits_and_embeddings=False,
        return_intermediates=False,
        mask=None,
        return_mems=False,
        return_attn=False,
        mems=None,
        mem_masks=None,
        pos=None,
        prepend_embeds=None,
        prepend_mask=None,
        embed_ids: Dict[str, Tensor] = dict(),
        sum_embeds=None,
        return_attn_z_loss=False,
        attn_z_loss_weight=1e-4,
        seq_start_pos=None,
        cache: LayerIntermediates | None = None,
        **kwargs
    ):
        b, n, device, num_mems, has_memory_tokens, emb_frac_gradient = x.shape[0], x.shape[1], x.device, self.num_memory_tokens, self.num_memory_tokens > 0, self.emb_frac_gradient
        return_hiddens = return_mems | return_attn | return_intermediates | return_attn_z_loss

        # absolute positional embedding

        external_pos_emb = exists(pos) and pos.dtype != torch.long
        pos_emb = self.pos_emb(x, pos=pos, seq_start_pos=seq_start_pos) if not external_pos_emb else pos
        x = self.token_emb(x) + pos_emb

        # add additional embeddings

        if exists(self.embeds):
            assert len(embed_ids) == len(self.embeds)

            for name, embed_id in embed_ids.items():
                embed_key = f'{name}_embed'

                assert embed_key in self.embeds
                embed = self.embeds[embed_key](embed_id)

                x = x + embed

        # for summing embeddings passed externally - needs this for self-conditioning in non-autoregressive training

        if exists(sum_embeds):
            x = x + sum_embeds

        # post embedding norm, purportedly leads to greater stabilization

        x = self.post_emb_norm(x)

        # whether to append embeds, as in PaLI, for image embeddings

        if exists(prepend_embeds):
            prepend_seq, prepend_dim = prepend_embeds.shape[1:]
            assert prepend_dim == x.shape[-1], \
                'prepended embeddings need to have same dimensions as text model dimensions'

            x = torch.cat((prepend_embeds, x), dim=-2)

            if exists(prepend_mask) or exists(mask):
                mask = default(mask, lambda: torch.ones((b, n), device=device, dtype=torch.bool))
                prepend_mask = default(
                    prepend_mask,
                    lambda: torch.ones((b, prepend_seq), device=device, dtype=torch.bool)
                )

                mask = torch.cat((prepend_mask, mask), dim=-1)

        # whether to reduce the gradient going to the embedding, from cogview paper, corroborated by GLM-130B model

        if emb_frac_gradient < 1:
            assert emb_frac_gradient > 0
            x = x * emb_frac_gradient + x.detach() * (1 - emb_frac_gradient)

        # embedding dropout

        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if has_memory_tokens:
            mem_every = self.memory_tokens_interspersed_every

            if exists(mem_every):
                assert mem_every > 0
                assert isinstance(self.attn_layers, Decoder), 'only for decoder'
                next_seq_len = math.ceil(n / mem_every) * mem_every

                x = pad_at_dim(x, (0, next_seq_len - n), dim=-2, value=0.)
                x = rearrange(x, 'b (n m) d -> (b n) m d', m=mem_every)

            mem = repeat(self.memory_tokens, 'n d -> b n d', b=x.shape[0])
            x, mem_packed_shape = pack((mem, x), 'b * d')

            # auto-handle masking after appending memory tokens
            if not exists(mem_every) and exists(mask):
                mask = pad_at_dim(mask, (num_mems, 0), dim=-1, value=True)

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b=b)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(
            x,
            mask=mask,
            mems=mems,
            mem_masks=mem_masks,
            cache=cache,
            return_hiddens=True,
            seq_start_pos=seq_start_pos,
            **kwargs
        )

        # handle memories post-attention

        if has_memory_tokens:
            if exists(mem_every):
                x = rearrange(x, 'b (n m) d -> (b n) m d', m=(mem_every + num_mems))

            mem, x = unpack(x, mem_packed_shape, 'b * d')

            intermediates.memory_tokens = mem

            if exists(mem_every):
                x = rearrange(x, '(b n) m d -> b (n m) d', b=b)

            x = x[:, :n]

        # projecting to logits

        if not return_embeddings:
            if self.has_multiple_heads:
                logits = tuple(fn(x) for fn in self.to_logits)
            else:
                logits = self.to_logits(x)

        # different returns

        if return_logits_and_embeddings:
            out = (logits, x)
        elif return_embeddings:
            out = x
        else:
            out = logits

        # aux loss

        if return_attn_z_loss:
            pre_softmax_attns = list(map(lambda t: t.pre_softmax_attn, intermediates.attn_intermediates))
            intermediates.attn_z_loss = calc_z_loss(pre_softmax_attns, weight=attn_z_loss_weight)
            return_intermediates = True

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda pair: torch.cat(pair, dim=-2), zip(mems, hiddens))) if exists(mems) else hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), new_mems))

            if not return_intermediates:
                return out, new_mems

            intermediates.mems = new_mems

        if return_intermediates:
            return out, intermediates

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out
