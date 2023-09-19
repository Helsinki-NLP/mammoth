from torch import nn
from typing import Dict, List

from mammoth.encoders.encoder import EncoderBase
from mammoth.models.adapters import Adapter, AdaptedTransformerEncoder
from mammoth.utils.misc import sequence_mask
from mammoth.distributed import DatasetMetadata


class LayerStackEncoder(EncoderBase):
    def __init__(self, embeddings, encoders):
        super().__init__()

        self.embeddings = embeddings
        self.encoders: nn.ModuleList[nn.ModuleDict] = encoders
        self._adapter_to_stack: Dict[str, int] = dict()
        self._active: List[str] = []

    @classmethod
    def from_opt(cls, opt, embeddings, task_queue_manager):
        """Alternate constructor for use during training."""
        encoders = nn.ModuleList()
        for layer_stack_index, n_layers in enumerate(opt.enc_layers):
            stacks = nn.ModuleDict()
            is_on_top = layer_stack_index == len(opt.enc_layers) - 1
            for module_id in task_queue_manager.get_encoders(layer_stack_index):
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
                    layer_norm_module=(
                        nn.LayerNorm(opt.enc_rnn_size, eps=1e-6) if is_on_top
                        else nn.Identity()
                    )
                )
            encoders.append(stacks)
        return cls(embeddings, encoders)

    @classmethod
    def from_trans_opt(cls, model_opt, embeddings, opt_stack):
        """Alternate constructor for use during translation."""
        encoders = nn.ModuleList()
        for layer_stack_index, n_layers in enumerate(model_opt.enc_layers):
            stacks = nn.ModuleDict()
            module_opts = opt_stack['encoder'][layer_stack_index]
            module_id = module_opts['id']
            is_on_top = layer_stack_index == len(model_opt.enc_layers) - 1
            stacks[module_id] = AdaptedTransformerEncoder(
                n_layers,
                model_opt.enc_rnn_size,
                model_opt.heads,
                model_opt.transformer_ff,
                model_opt.dropout[0] if type(model_opt.dropout) is list else model_opt.dropout,
                (
                    model_opt.attention_dropout[0]
                    if type(model_opt.attention_dropout) is list
                    else model_opt.attention_dropout
                ),
                None,  # embeddings,
                model_opt.max_relative_positions,
                pos_ffn_activation_fn=model_opt.pos_ffn_activation_fn,
                layer_norm_module=(
                    nn.LayerNorm(model_opt.enc_rnn_size, eps=1e-6) if is_on_top
                    else nn.Identity()
                )
            )
            encoders.append(stacks)
        return cls(embeddings, encoders)

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for stacks in self.encoders:
            for stack in stacks.values():
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
            for stack in stacks.values():
                stack.freeze_base_model(requires_grad=requires_grad)

    @property
    def n_layer_stacks(self):
        return len(self.encoders)

    def get_submodule(self, layer_stack_index: int, module_id: str):
        return self.encoders[layer_stack_index][module_id]

    def get_adapter(self, module_id: str, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index = self._adapter_to_stack[name]
        return self.encoders[layer_stack_index][module_id].get_adapter(adapter_group, sub_id)

    def add_adapter(
        self,
        adapter_group: str,
        sub_id: str,
        adapter: Adapter,
        layer_stack_index: int,
        module_ids: List[str],
    ):
        """Adds the specified adapter with the name (adapter_group, sub_id)
        into the module_id sharing group of the layer_stack_index'th stack"""
        name = Adapter._name(adapter_group, sub_id)
        if name in self._adapter_to_stack:
            raise ValueError(f'Duplicate Adapter "{name}"')
        self._adapter_to_stack[name] = layer_stack_index
        if layer_stack_index >= len(self.encoders):
            raise ValueError(
                f'No layer stack with index {layer_stack_index}. There are {len(len(self.encoders))} layer stacks'
            )
        if len(module_ids) == 0:
            raise Exception(f'Adapter {adapter_group} {sub_id} has no module_ids')
        for module_id in module_ids:
            if module_id not in self.encoders[layer_stack_index]:
                raise ValueError(
                    f'No sharing group / module_id "{module_id}" in the selected index {layer_stack_index}. '
                    f'Expected one of {self.encoders[layer_stack_index].keys()}'
                )
            self.encoders[layer_stack_index][module_id].add_adapter(adapter_group, sub_id, adapter)

    def deactivate_adapters(self):
        for stacks in self.encoders:
            for stack in stacks.values():
                stack.deactivate_adapters()

    def activate_adapter(self, module_id: str, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        try:
            layer_stack_index = self._adapter_to_stack[name]
        except KeyError:
            raise KeyError(f'"{name}" not in {self._adapter_to_stack.keys()}')
        self.encoders[layer_stack_index][module_id].activate_adapter(adapter_group, sub_id)

    def activate(self, metadata: DatasetMetadata):
        self._active = metadata.encoder_id
        self.embeddings.activate(metadata.src_lang)
        if metadata.encoder_adapter_ids is not None:
            self.deactivate_adapters()
            for layer_stack_index, adapter_group, sub_id in metadata.encoder_adapter_ids:
                module_id = metadata.encoder_id[layer_stack_index]
                self.activate_adapter(module_id, adapter_group, sub_id)
