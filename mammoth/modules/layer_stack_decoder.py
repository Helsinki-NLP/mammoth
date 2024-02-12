from collections import defaultdict
from torch import nn
from typing import Dict, List

from mammoth.modules.decoder import DecoderBase
from mammoth.models.adapters import Adapter, AdaptedTransformerDecoder
from mammoth.distributed import DatasetMetadata


class LayerStackDecoder(DecoderBase):
    def __init__(self, embeddings, decoders):
        super().__init__()

        self.embeddings = embeddings
        self.decoders: nn.ModuleList[nn.ModuleDict] = decoders
        self._adapter_to_stack: Dict[str, int] = dict()
        self._active: List[str] = []

    @classmethod
    def from_opts(cls, opts, embeddings, task_queue_manager, is_on_top=False):
        """Alternate constructor for use during training."""
        decoders = nn.ModuleList()
        for layer_stack_index, n_layers in enumerate(opts.dec_layers):
            is_on_top = layer_stack_index == len(opts.dec_layers) - 1
            stacks = nn.ModuleDict()
            for module_id in task_queue_manager.get_my_decoders(layer_stack_index):
                if module_id in stacks:
                    # several tasks using the same layer stack
                    continue
                stacks[module_id] = AdaptedTransformerDecoder(
                    n_layers,
                    opts.model_dim,
                    opts.heads,
                    opts.transformer_ff,
                    opts.copy_attn,
                    opts.self_attn_type,
                    opts.dropout[0] if isinstance(opts.dropout, list) else opts.dropout,
                    (
                        opts.attention_dropout[0]
                        if isinstance(opts.attention_dropout, list)
                        else opts.attention_dropout
                    ),
                    None,  # embeddings,
                    opts.max_relative_positions,
                    opts.aan_useffn,
                    opts.full_context_alignment,
                    opts.alignment_layer,
                    alignment_heads=opts.alignment_heads,
                    pos_ffn_activation_fn=opts.pos_ffn_activation_fn,
                    layer_norm_module=(
                        nn.LayerNorm(opts.model_dim, eps=1e-6) if is_on_top
                        else nn.Identity()
                    ),
                    is_normformer=opts.normformer,
                )
            decoders.append(stacks)
        return cls(embeddings, decoders)

    @classmethod
    def from_trans_opt(cls, opts, embeddings, task, is_on_top=False):
        """Alternate constructor for use during training."""
        decoders = nn.ModuleList()
        for layer_stack_index, n_layers in enumerate(opts.dec_layers):
            is_on_top = layer_stack_index == len(opts.dec_layers) - 1
            stacks = nn.ModuleDict()
            module_id = task.decoder_id[layer_stack_index]
            stacks[module_id] = AdaptedTransformerDecoder(
                n_layers,
                opts.model_dim,
                opts.heads,
                opts.transformer_ff,
                opts.copy_attn,
                opts.self_attn_type,
                opts.dropout[0] if isinstance(opts.dropout, list) else opts.dropout,
                (
                    opts.attention_dropout[0]
                    if isinstance(opts.attention_dropout, list)
                    else opts.attention_dropout
                ),
                None,  # embeddings,
                opts.max_relative_positions,
                opts.aan_useffn,
                opts.full_context_alignment,
                opts.alignment_layer,
                alignment_heads=opts.alignment_heads,
                pos_ffn_activation_fn=opts.pos_ffn_activation_fn,
                layer_norm_module=(
                    nn.LayerNorm(opts.model_dim, eps=1e-6) if is_on_top
                    else nn.Identity()
                ),
                is_normformer=opts.normformer,
            )
            decoders.append(stacks)
        return cls(embeddings, decoders)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        for stacks in self.decoders:
            for stack in stacks.values():
                stack.init_state(src, memory_bank, enc_hidden)

    def detach_state(self):
        for stacks in self.decoders:
            for stack in stacks.values():
                stack.detach_state()

    def map_state(self, fn_map_state):
        for stacks in self.decoders:
            for stack in stacks.values():
                stack.map_state(fn_map_state)

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for stacks in self.decoders:
            for stack in stacks.values():
                stack.update_dropout(dropout, attention_dropout)

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
        for active_id, stacks in zip(self._active, self.decoders):
            decoder = stacks[active_id]
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
        for stacks in self.decoders:
            for stack in stacks.values():
                stack.freeze_base_model(requires_grad=requires_grad)

    @property
    def n_layer_stacks(self):
        return len(self.decoders)

    def get_submodule(self, layer_stack_index: int, module_id: str):
        return self.decoders[layer_stack_index][module_id]

    def get_adapter(self, module_id: str, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index = self._adapter_to_stack[name]
        return self.decoders[layer_stack_index][module_id].get_adapter(adapter_group, sub_id)

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
        if layer_stack_index >= len(self.decoders):
            raise ValueError(
                f'No layer stack with index {layer_stack_index}. There are {len(len(self.decoders))} layer stacks'
            )
        if len(module_ids) == 0:
            raise Exception(f'Adapter {adapter_group} {sub_id} has no module_ids')
        for module_id in module_ids:
            if module_id not in self.decoders[layer_stack_index]:
                raise ValueError(
                    f'No sharing group / module_id "{module_id}" in the selected index {layer_stack_index}. '
                    f'Expected one of {self.decoders[layer_stack_index].keys()}'
                )
            self.decoders[layer_stack_index][module_id].add_adapter(adapter_group, sub_id, adapter)

    def deactivate_adapters(self):
        for stacks in self.decoders:
            for stack in stacks.values():
                stack.deactivate_adapters()

    def activate_adapter(self, module_id: str, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        try:
            layer_stack_index = self._adapter_to_stack[name]
        except KeyError:
            raise KeyError(f'"{name}" not in {self._adapter_to_stack.keys()}')
        self.decoders[layer_stack_index][module_id].activate_adapter(adapter_group, sub_id)

    def activate(self, metadata: DatasetMetadata):
        self._active = metadata.decoder_id
        self.embeddings.activate(metadata.tgt_lang)
        if metadata.decoder_adapter_ids is not None:
            self.deactivate_adapters()
            for layer_stack_index, adapter_group, sub_id in metadata.decoder_adapter_ids:
                module_id = metadata.decoder_id[layer_stack_index]
                self.activate_adapter(module_id, adapter_group, sub_id)
