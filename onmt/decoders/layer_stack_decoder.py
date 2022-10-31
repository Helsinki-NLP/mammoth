from collections import defaultdict
from torch import nn
from typing import Dict, Tuple, List

from onmt.decoders.decoder import DecoderBase
from onmt.models.adapters import Adapter, AdaptedTransformerDecoder
from onmt.utils.distributed import DatasetMetadata


class LayerStackDecoder(DecoderBase):
    def __init__(self, embeddings, decoders):
        super().__init__()

        self.embeddings = embeddings
        self.decoders: nn.ModuleList[nn.ModuleDict] = decoders
        self._adapter_to_stack: Dict[str, Tuple[int, str]] = dict()
        self._active: List[str] = []

    @classmethod
    def from_opt(cls, opt, embeddings, task_queue_manager):
        """Alternate constructor."""
        decoders = nn.ModuleList()
        for layer_stack_index, n_layers in enumerate(opt.dec_layers):
            stacks = nn.ModuleDict()
            for module_id in task_queue_manager.get_decoders(layer_stack_index):
                if module_id in stacks:
                    # several tasks using the same layer stack
                    continue
                stacks[module_id] = AdaptedTransformerDecoder(
                    n_layers,
                    opt.dec_rnn_size,
                    opt.heads,
                    opt.transformer_ff,
                    opt.copy_attn,
                    opt.self_attn_type,
                    opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
                    (
                        opt.attention_dropout[0]
                        if type(opt.attention_dropout) is list
                        else opt.attention_dropout
                    ),
                    None,  # embeddings,
                    opt.max_relative_positions,
                    opt.aan_useffn,
                    opt.full_context_alignment,
                    opt.alignment_layer,
                    alignment_heads=opt.alignment_heads,
                    pos_ffn_activation_fn=opt.pos_ffn_activation_fn,
                )
            decoders.append(stacks)
        return cls(embeddings, decoders)

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        for stacks in self.decoders:
            for stack in stacks:
                stack.init_state(src, memory_bank, enc_hidden)

    def detach_state(self):
        for stacks in self.decoders:
            for stack in stacks:
                stack.detach_state()

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for stacks in self.decoders:
            for stack in stacks:
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
            for stack in stacks:
                stack.freeze_base_model(requires_grad=requires_grad)

    def get_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index, module_id = self._adapter_to_stack[name]
        return self.decoders[layer_stack_index][module_id].get_adapter(adapter_group, sub_id)

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
        if name in self._adapter_to_decoder:
            raise ValueError(f'Duplicate Adapter "{name}"')
        self._adapter_to_stack[name] = layer_stack_index
        self.decoders[layer_stack_index][module_id].add_adapter(adapter_group, sub_id, adapter)

    def deactivate_adapters(self):
        for stacks in self.decoders:
            for stack in stacks:
                stack.deactivate_adapters()

    def activate_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        layer_stack_index, module_id = self._adapter_to_stack[name]
        self.decoders[layer_stack_index][module_id].activate_adapter(adapter_group, sub_id)

    def activate(self, metadata: DatasetMetadata):
        self._active = metadata.decoder_id
        if metadata.decoder_adapter_ids is not None:
            self.deactivate_adapters()
            for adapter_id in metadata.decoder_adapter_ids:
                self.activate_adapter(*adapter_id)
