from collections import defaultdict
from torch import nn
from typing import Dict

from onmt.decoders.decoder import DecoderBase
from onmt.models.adapters import Adapter, AdaptedTransformerDecoder


class ModularDecoder(DecoderBase):
    def __init__(self, embeddings, decoders):
        super().__init__()

        self.embeddings = embeddings
        self.decoders = decoders
        self._adapter_to_decoder: Dict[str, str] = dict()

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        decoders = nn.ModuleList()
        for n_layers in opt.dec_layers:
            decoders.append(AdaptedTransformerDecoder(
                n_layers,
                opt.dec_rnn_size,
                opt.heads,
                opt.transformer_ff,
                opt.copy_attn,
                opt.self_attn_type,
                opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
                opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout,
                embeddings,
                opt.max_relative_positions,
                opt.aan_useffn,
                opt.full_context_alignment,
                opt.alignment_layer,
                alignment_heads=opt.alignment_heads,
                pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            ))
        return cls(embeddings, decoders)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        for decoder in self.decoders:
            decoder.init_state(src, memory_bank, enc_hidden)

    def detach_state(self):
        for decoder in self.decoders:
            decoder.detach_state()

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for decoder in self.decoders:
            decoder.update_dropout(dropout, attention_dropout)

    def forward(self, tgt, memory_bank=None, step=None, memory_lengths=None, **kwargs):
        # wrapper embeds tgt and creates tgt_pad_mask
        tgt_words = tgt[:, :, 0].transpose(0, 1)
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # len x batch x embedding_dim
        emb = emb.transpose(0, 1).contiguous()
        # memory bank transposed to batch-first order
        memory_bank = memory_bank.transpose(0, 1).contiguous()

        output = emb
        attns = defaultdict(list)
        for decoder in self.decoders:
            output, dec_attns = decoder.forward(
                output,
                memory_bank=memory_bank,
                step=step,
                memory_lengths=memory_lengths,
                tgt_pad_mask=tgt_pad_mask,
                skip_embedding=True,
            )
            for key, val in dec_attns.items():
                attns[key].append(val)

        # only at the end transpose back to timestep-first
        output = output.transpose(0, 1).contiguous()
        return output, attns

    def freeze_base_model(self, requires_grad=False):
        for decoder in self.decoders:
            decoder.freeze_base_model(requires_grad=requires_grad)

    def get_adapter(self, adapter_group: str, sub_id: str):
        dec_idx = Adapter._name(adapter_group, sub_id)
        return self.decoders[dec_idx].get_adapter(adapter_group, sub_id)

    def add_adapter(self, adapter_group: str, sub_id: str, adapter: Adapter, dec_idx: int):
        name = Adapter._name(adapter_group, sub_id)
        if name in self._adapter_to_decoder:
            raise ValueError(f'Duplicate Adapter "{name}"')
        self._adapter_to_decoder[name] = dec_idx
        self.decoders[dec_idx].add_adapter(adapter_group, sub_id, adapter)

    def deactivate_adapters(self):
        for decoder in self.decoders:
            decoder.deactivate_adapters()

    def activate_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        dec_idx = self._adapter_to_decoder[name]
        self.decoders[dec_idx].activate_adapter(adapter_group, sub_id)
