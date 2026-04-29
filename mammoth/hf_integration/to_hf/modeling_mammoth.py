"""Mammoth HF model: wraps x-transformers TransformerWrapper inside PreTrainedModel."""

import torch
import torch.nn as nn
from typing import Optional
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

try:
    # End-user mode: x_transformers.py and its leaf deps (attend.py, autoregressive_wrapper.py)
    # are vendored as sibling files in the HF model dir. We import the leaf deps explicitly
    # so HF's trust_remote_code loader (which does NOT recurse into relative imports) copies
    # all four files into its cache. Without these two lines, HF copies only x_transformers.py
    # and the runtime `from .attend import ...` inside it fails with FileNotFoundError.
    from .attend import Attend  # noqa: F401  pulled in transitively by .x_transformers
    from .autoregressive_wrapper import AutoregressiveWrapper  # noqa: F401  pulled in transitively by .x_transformers
    from .x_transformers import TransformerWrapper, Encoder, Decoder
except ImportError:
    # Dev mode: imported from the mammoth source tree, where x_transformers lives at mammoth.x_transformers.
    from mammoth.x_transformers import TransformerWrapper, Encoder, Decoder
from .configuration_mammoth import MammothConfig


def _build_encoder(config: MammothConfig) -> TransformerWrapper:
    encoder_attn = Encoder(
        dim=config.enc_model_dim,
        depth=config.enc_layers,
        heads=config.enc_heads,
        attn_dim_head=config.enc_attn_dim_head,
        attn_dropout=config.enc_attn_dropout,
        attn_flash=config.enc_attn_flash,
        attn_qkv_bias=config.enc_attn_qkv_bias,
        ff_mult=config.enc_ff_mult,
        ff_glu=config.enc_ff_glu,
        ff_no_bias=config.enc_ff_no_bias,
        ff_dropout=config.enc_ff_dropout,
        pre_norm=config.enc_pre_norm,
        use_rmsnorm=config.enc_use_rmsnorm,
        layernorm_bias=config.enc_layernorm_bias,
        norm_add_unit_offset=config.enc_norm_add_unit_offset,
        rotary_pos_emb=config.enc_rotary_pos_emb,
    )
    return TransformerWrapper(
        num_tokens=config.src_vocab_size,
        max_seq_len=config.enc_max_seq_len,
        attn_layers=encoder_attn,
        emb_dropout=config.enc_emb_dropout,
        post_emb_norm=config.enc_post_emb_norm,
        post_emb_norm_bias=config.enc_post_emb_norm_bias,
        scaled_embeddings=config.enc_scaled_embeddings,
        use_abs_pos_emb=config.enc_use_abs_pos_emb,
        return_only_embed=True,
    )


def _build_decoder(config: MammothConfig) -> TransformerWrapper:
    dec_kwargs = dict(
        dim=config.dec_model_dim,
        depth=config.dec_layers,
        heads=config.dec_heads,
        attn_dim_head=config.dec_attn_dim_head,
        attn_dropout=config.dec_attn_dropout,
        attn_flash=config.dec_attn_flash,
        attn_qkv_bias=config.dec_attn_qkv_bias,
        ff_mult=config.dec_ff_mult,
        ff_glu=config.dec_ff_glu,
        ff_no_bias=config.dec_ff_no_bias,
        ff_dropout=config.dec_ff_dropout,
        pre_norm=config.dec_pre_norm,
        use_rmsnorm=config.dec_use_rmsnorm,
        layernorm_bias=config.dec_layernorm_bias,
        norm_add_unit_offset=config.dec_norm_add_unit_offset,
        rotary_pos_emb=config.dec_rotary_pos_emb,
        cross_attend=True,
    )
    if config.dec_attn_kv_heads is not None:
        dec_kwargs['attn_kv_heads'] = config.dec_attn_kv_heads
    if config.dec_attn_qk_norm:
        dec_kwargs['attn_qk_norm'] = True
        dec_kwargs['attn_qk_norm_dim_scale'] = config.dec_attn_qk_norm_dim_scale
    if config.dec_cross_attn_dim_context is not None:
        dec_kwargs['cross_attn_dim_context'] = config.dec_cross_attn_dim_context
    if config.dec_sandwich_norm:
        dec_kwargs['sandwich_norm'] = True

    decoder_attn = Decoder(**dec_kwargs)
    return TransformerWrapper(
        num_tokens=config.tgt_vocab_size,
        max_seq_len=config.dec_max_seq_len,
        attn_layers=decoder_attn,
        emb_dropout=config.dec_emb_dropout,
        post_emb_norm=config.dec_post_emb_norm,
        post_emb_norm_bias=config.dec_post_emb_norm_bias,
        scaled_embeddings=config.dec_scaled_embeddings,
        use_abs_pos_emb=config.dec_use_abs_pos_emb,
        tie_embedding=config.tie_word_embeddings,
    )


class _HFEncoderWrapper(nn.Module):
    """Adapts x-transformers TransformerWrapper for HF generate (attention_mask → mask)."""

    def __init__(self, encoder: TransformerWrapper):
        super().__init__()
        self._encoder = encoder

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        mask = attention_mask.bool() if attention_mask is not None else None
        hidden = self._encoder(input_ids, mask=mask)
        return BaseModelOutput(last_hidden_state=hidden)


class MammothPreTrainedModel(PreTrainedModel):
    config_class = MammothConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        pass  # x-transformers handles its own initialization


class MammothModel(MammothPreTrainedModel):
    """Container holding encoder and decoder TransformerWrappers."""

    def __init__(self, config: MammothConfig):
        super().__init__(config)
        self.encoder = _build_encoder(config)
        self.decoder = _build_decoder(config)
        self.post_init()

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder


class MammothForConditionalGeneration(MammothPreTrainedModel, GenerationMixin):
    """Seq2seq LM wrapping Mammoth x-transformers encoder-decoder."""

    def __init__(self, config: MammothConfig):
        super().__init__(config)
        self.model = MammothModel(config)
        self.post_init()

    def get_encoder(self):
        return _HFEncoderWrapper(self.model.encoder)

    def get_decoder(self):
        return self.model.decoder

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
        src_mask = attention_mask.bool() if attention_mask is not None else None
        tgt_mask = decoder_attention_mask.bool() if decoder_attention_mask is not None else None

        if encoder_outputs is None:
            encoder_hidden = self.model.encoder(input_ids, mask=src_mask)
        elif isinstance(encoder_outputs, BaseModelOutput):
            encoder_hidden = encoder_outputs.last_hidden_state
        elif isinstance(encoder_outputs, (tuple, list)):
            encoder_hidden = encoder_outputs[0]
        else:
            encoder_hidden = encoder_outputs

        # decoder with return_only_embed=False (default) → applies to_logits → returns [B,T,V]
        logits = self.model.decoder(
            decoder_input_ids,
            context=encoder_hidden,
            context_mask=src_mask,
            mask=tgt_mask,
        )

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
