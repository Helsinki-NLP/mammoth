from torch import nn
from typing import Dict

from onmt.encoders.encoder import EncoderBase
from onmt.models.adapters import Adapter, AdaptedTransformerEncoder
from onmt.utils.misc import sequence_mask


class LayerStackEncoder(EncoderBase):
    def __init__(self, embeddings, encoders):
        super().__init__()

        self.embeddings = embeddings
        self.encoders = encoders
        self._adapter_to_encoder: Dict[str, str] = dict()

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        encoders = nn.ModuleList()
        for n_layers in opt.enc_layers:
            encoders.append(AdaptedTransformerEncoder(
                n_layers,
                opt.enc_rnn_size,
                opt.heads,
                opt.transformer_ff,
                opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
                opt.attention_dropout[0] if type(opt.attention_dropout) is list else opt.attention_dropout,
                None,  # embeddings,
                opt.max_relative_positions,
                pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
            ))
        return cls(embeddings, encoders)

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for encoder in self.encoders:
            encoder.update_dropout(dropout, attention_dropout)

    def forward(self, src, lengths=None, **kwargs):
        # wrapper embeds src and creates mask
        emb = self.embeddings(src)
        emb = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        output = emb
        for encoder in self.encoders:
            # Throw away emb, lengths, mask
            _, output, _, _ = encoder.forward(
                output,
                lengths=lengths,
                skip_embedding=True,
                mask=mask,
            )

        # only at the end transpose back to timestep-first
        output = output.transpose(0, 1).contiguous()
        return emb, output, lengths, mask

    def freeze_base_model(self, requires_grad=False):
        for encoder in self.encoders:
            encoder.freeze_base_model(requires_grad=requires_grad)

    def get_adapter(self, adapter_group: str, sub_id: str):
        layer_stack_index = Adapter._name(adapter_group, sub_id)
        return self.encoders[layer_stack_index].get_adapter(adapter_group, sub_id)

    def add_adapter(self, adapter_group: str, sub_id: str, adapter: Adapter, layer_stack_index: int):
        name = Adapter._name(adapter_group, sub_id)
        if name in self._adapter_to_encoder:
            raise ValueError(f'Duplicate Adapter "{name}"')
        self._adapter_to_encoder[name] = layer_stack_index
        self.encoders[layer_stack_index].add_adapter(adapter_group, sub_id, adapter)

    def deactivate_adapters(self):
        for encoder in self.encoders:
            encoder.deactivate_adapters()

    def activate_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index = self._adapter_to_encoder[name]
        self.encoders[layer_stack_index].activate_adapter(adapter_group, sub_id)
