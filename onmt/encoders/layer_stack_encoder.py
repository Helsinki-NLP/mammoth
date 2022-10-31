from torch import nn
from typing import Dict, Tuple, List

from onmt.encoders.encoder import EncoderBase
from onmt.models.adapters import Adapter, AdaptedTransformerEncoder
from onmt.utils.misc import sequence_mask
from onmt.utils.distributed import DatasetMetadata


class LayerStackEncoder(EncoderBase):
    def __init__(self, embeddings, encoders):
        super().__init__()

        self.embeddings = embeddings
        self.encoders: nn.ModuleList[nn.ModuleDict] = encoders
        self._adapter_to_stack: Dict[str, Tuple[int, str]] = dict()
        self._active: List[str] = []

    @classmethod
    def from_opt(cls, opt, embeddings, task_queue_manager):
        """Alternate constructor."""
        encoders = nn.ModuleList()
        for layer_stack_index, n_layers in enumerate(opt.enc_layers):
            stacks = nn.ModuleDict()
            for module_id in task_queue_manager.get_decoders(layer_stack_index):
                if module_id in stacks:
                    # several tasks using the same layer stack
                    continue
                stacks[module_id] = AdaptedTransformerEncoder(
                    n_layers,
                    opt.enc_rnn_size,
                    opt.heads,
                    opt.transformer_ff,
                    opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
                    (
                        opt.attention_dropout[0]
                        if type(opt.attention_dropout) is list
                        else opt.attention_dropout
                    ),
                    None,  # embeddings,
                    opt.max_relative_positions,
                    pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
                )
            encoders.append(stacks)
        return cls(embeddings, encoders)

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for stacks in self.encoders:
            for stack in stacks:
                stack.update_dropout(dropout, attention_dropout)

    def forward(self, src, lengths=None, **kwargs):
        # wrapper embeds src and creates mask
        emb = self.embeddings(src)
        emb = emb.transpose(0, 1).contiguous()
        mask = ~sequence_mask(lengths).unsqueeze(1)

        output = emb
        for active_id, stacks in zip(self._active, self.encoders):
            encoder = stacks[active_id]
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
        for stacks in self.encoders:
            for stack in stacks:
                stack.freeze_base_model(requires_grad=requires_grad)

    def get_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index, module_id = self._adapter_to_stack[name]
        return self.encoders[layer_stack_index][module_id].get_adapter(adapter_group, sub_id)

    def add_adapter(
        self,
        adapter_group: str,
        sub_id: str,
        adapter: Adapter,
        layer_stack_index: int,
        module_id: str,
    ):
        """Adds the specified adapter with the name (adapter_group, sub_id)
        into the module_id sharing group of the layer_stack_index'th stack"""
        name = Adapter._name(adapter_group, sub_id)
        if name in self._adapter_to_encoder:
            raise ValueError(f'Duplicate Adapter "{name}"')
        self._adapter_to_stack[name] = layer_stack_index
        self.encoders[layer_stack_index][module_id].add_adapter(adapter_group, sub_id, adapter)

    def deactivate_adapters(self):
        for stacks in self.encoders:
            for stack in stacks:
                stack.deactivate_adapters()

    def activate_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index, module_id = self._adapter_to_stack[name]
        self.encoders[layer_stack_index][module_id].activate_adapter(adapter_group, sub_id)

    def activate(self, metadata: DatasetMetadata):
        self._active = metadata.encoder_id
        if metadata.encoder_adapter_ids is not None:
            self.deactivate_adapters()
            for adapter_id in metadata.encoder_adapter_ids:
                self.activate_adapter(*adapter_id)
