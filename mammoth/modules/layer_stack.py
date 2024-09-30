from torch import nn
from typing import List, Sequence, Optional, Tuple
from x_transformers.x_transformers import LayerIntermediates

from mammoth.modules.adapters import AdaptedAttentionLayers


class AdaptedAttentionLayersStack(nn.Module):
    """
    Wrapper that allows stacking multiple AdaptedAttentionLayers.
    Represents one particular stacking: does not allow switching out entire layers
    (but does delegate the switching out of adapters to its components)
    """
    def __init__(self, attention_layers_stack: Sequence[AdaptedAttentionLayers]):
        super().__init__()
        self.attention_layers_stack = nn.ModuleList(attention_layers_stack)
        assert len(set(attention_layers.dim for attention_layers in attention_layers_stack)) == 1, \
            'All AdaptedAttentionLayers must have the same dimension'

    def forward(self, x, return_hiddens=False, cache: Optional[List[LayerIntermediates]] = None, **kwargs):
        all_intermediates = []
        for i, attention_layers in enumerate(self.attention_layers_stack):
            if cache:
                cache_i = cache[i]
            else:
                cache_i = None
            if return_hiddens:
                x, intermediates = attention_layers.forward(x, return_hiddens=True, cache=cache_i, **kwargs)
                all_intermediates.append(intermediates)
            else:
                x = attention_layers.forward(x, return_hiddens=False, cache=cache_i, **kwargs)
        if return_hiddens:
            return x, all_intermediates
        else:
            return x

    def freeze_base_model(self, requires_grad=False):
        for attention_layers in self.attention_layers_stack:
            attention_layers.freeze_base_model(requires_grad=requires_grad)

    def deactivate_adapters(self):
        for attention_layers in self.attention_layers_stack:
            attention_layers.deactivate_adapters()

    def activate_adapter(self, layer_stack_index: int, adapter_group: str, sub_id: str):
        attention_layers = self.attention_layers_stack[layer_stack_index]
        attention_layers.activate_adapter(adapter_group, sub_id)

    @property
    def dim(self):
        return self.attention_layers_stack[0].dim

    @property
    def disable_abs_pos_emb(self):
        return self.attention_layers_stack[0].disable_abs_pos_emb


class StackXcoder(nn.ModuleDict):
    """
    Switches between different AdaptedAttentionLayersStacks depending on the task.
    """
    def __init__(self, *args, token_embs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.active_task = None
        self.token_embs = token_embs

    # TransformerWrapper wraps an AttentionLayers in embeddings and some other functionality.
    # We use one TransformerWrapper per task.
    def activate(self, task_id: str, adapter_ids: Optional[List[Tuple[int, str, str]]]):
        self.active_task = task_id
        transformer_wrapper = self[task_id]
        attention_layers_stack = transformer_wrapper.attn_layers
        if adapter_ids:
            attention_layers_stack.deactivate_adapters()
            for layer_stack_index, adapter_group, sub_id in adapter_ids:
                attention_layers_stack.activate_adapter(layer_stack_index, adapter_group, sub_id)
        return transformer_wrapper

    def get_attention_layers(self, task_id: str, layer_stack_index: int) -> AdaptedAttentionLayers:
        return self[task_id].attn_layers.attention_layers_stack[layer_stack_index]

    def get_embedding_by_task_id(self, task_id):
        transformer_wrapper = self[task_id]
        return transformer_wrapper.token_emb

    def get_embedding_by_lang(self, lang):
        return self.token_embs[lang]

    # Lack of forward is intentional: call forward on the return value of activate
