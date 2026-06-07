"""Mammoth HF model: native PyTorch encoder-decoder wrapped in PreTrainedModel."""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

try:
    from mammoth.modules.transformer import (
        TransformerStack, EncoderBlock, DecoderBlock, RotaryEmbedding,
    )
except ImportError:
    # Vendored standalone mode: native_transformer.py is copied into the
    # HF artifact alongside this file by convert_mammoth_to_hf.py.
    from .native_transformer import (  # noqa: F401
        TransformerStack, EncoderBlock, DecoderBlock, RotaryEmbedding,
    )

from .configuration_mammoth import MammothConfig


class MammothEncoder(nn.Module):
    main_input_name = "input_ids"

    def __init__(self, config: MammothConfig):
        super().__init__()
        dim = config.model_dim
        activation = "swiglu" if config.ff_swiglu else "gelu"

        self.token_emb = nn.Embedding(config.src_vocab_size, dim)
        self.post_emb_norm = nn.RMSNorm(dim) if config.post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(config.emb_dropout)

        stacks = []
        for i, depth in enumerate(config.enc_layers):
            is_last = (i == len(config.enc_layers) - 1)
            final_norm = nn.RMSNorm(dim) if is_last else None
            blocks = nn.ModuleList([
                EncoderBlock(
                    dim=dim,
                    heads=config.heads,
                    ff_mult=config.ff_mult,
                    attn_dropout=config.attn_dropout,
                    ff_dropout=config.ff_dropout,
                    activation=activation,
                )
                for _ in range(depth)
            ])
            stacks.append(TransformerStack(blocks, final_norm, dim, i, f"enc_{i}"))
        self.stacks = nn.ModuleList(stacks)

        self.rotary_emb = RotaryEmbedding(dim // config.heads) if config.rotary_pos_emb else None

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        h = self.emb_dropout(self.post_emb_norm(self.token_emb(input_ids)))
        rotary = self.rotary_emb(h.size(1), h.device) if self.rotary_emb is not None else None
        mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        for stack in self.stacks:
            h, _ = stack(h, mask=mask, rotary=rotary)
        if return_dict:
            return BaseModelOutput(last_hidden_state=h)
        return (h,)


class MammothDecoder(nn.Module):
    def __init__(self, config: MammothConfig):
        super().__init__()
        dim = config.model_dim
        activation = "swiglu" if config.ff_swiglu else "gelu"

        self.token_emb = nn.Embedding(config.tgt_vocab_size, dim)
        self.post_emb_norm = nn.RMSNorm(dim) if config.post_emb_norm else nn.Identity()
        self.emb_dropout = nn.Dropout(config.emb_dropout)

        stacks = []
        for i, depth in enumerate(config.dec_layers):
            is_last = (i == len(config.dec_layers) - 1)
            final_norm = nn.RMSNorm(dim) if is_last else None
            blocks = nn.ModuleList([
                DecoderBlock(
                    dim=dim,
                    heads=config.heads,
                    ff_mult=config.ff_mult,
                    attn_dropout=config.attn_dropout,
                    ff_dropout=config.ff_dropout,
                    activation=activation,
                )
                for _ in range(depth)
            ])
            stacks.append(TransformerStack(blocks, final_norm, dim, i, f"dec_{i}"))
        self.stacks = nn.ModuleList(stacks)

        self.rotary_emb = RotaryEmbedding(dim // config.heads) if config.rotary_pos_emb else None
        self.to_logits = nn.Linear(dim, config.tgt_vocab_size, bias=False)

    def forward(self, input_ids, context=None, context_mask=None, **kwargs):
        h = self.emb_dropout(self.post_emb_norm(self.token_emb(input_ids)))
        rotary = self.rotary_emb(h.size(1), h.device) if self.rotary_emb is not None else None
        for stack in self.stacks:
            h, _ = stack(h, context=context, context_mask=context_mask, rotary=rotary)
        return self.to_logits(h)


class _HFEncoderWrapper(nn.Module):
    """Adapts MammothEncoder for HF generate() (returns BaseModelOutput)."""

    def __init__(self, encoder: MammothEncoder):
        super().__init__()
        self._encoder = encoder

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        return self._encoder(input_ids, attention_mask=attention_mask, return_dict=return_dict)


class MammothPreTrainedModel(PreTrainedModel):
    config_class = MammothConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass  # weights are loaded from checkpoint, not randomly initialized


class MammothForConditionalGeneration(MammothPreTrainedModel, GenerationMixin):
    """Seq2seq LM wrapping native Mammoth encoder-decoder."""

    def __init__(self, config: MammothConfig):
        super().__init__(config)
        self.encoder = MammothEncoder(config)
        self.decoder = MammothDecoder(config)
        self.post_init()

    def get_encoder(self):
        return _HFEncoderWrapper(self.encoder)

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Seq2SeqLMOutput:
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask)

        if isinstance(encoder_outputs, BaseModelOutput):
            encoder_hidden = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, (tuple, list)):
            encoder_hidden = encoder_outputs[0]
        else:
            encoder_hidden = encoder_outputs

        context_mask = attention_mask[:, None, None, :].bool() if attention_mask is not None else None
        logits = self.decoder(decoder_input_ids, context=encoder_hidden, context_mask=context_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            loss = loss_fct(logits.view(-1, self.config.tgt_vocab_size), labels.view(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            encoder_last_hidden_state=encoder_hidden,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        encoder_outputs=None,
        **kwargs,
    ):
        return {
            "decoder_input_ids": decoder_input_ids,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
        }
