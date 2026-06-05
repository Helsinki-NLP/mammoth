from torch import nn
from typing import Optional, Dict

from mammoth.modules.transformer import NativeTransformerWrapper, TransformerStack


class StackXcoder(nn.ModuleDict):
    """
    Switches between different NativeTransformerWrappers depending on the task.
    """
    def __init__(
        self,
        transformer_wrappers: Dict[str, NativeTransformerWrapper],
        attention_layer_blocks: Dict[int, Dict[str, TransformerStack]],
        token_embs: Dict[str, nn.Embedding],
        adapters=None,  # no longer used; kept for call-site compatibility until Phase 3 rewrites model_builder
        per_component_post_emb_norms: Optional[Dict[tuple, nn.Module]] = None,
        per_component_pos_embs: Optional[Dict[tuple, nn.Module]] = None,
        per_component_project_embs: Optional[Dict[tuple, nn.Module]] = None,
        per_component_to_logits: Optional[Dict[tuple, nn.Module]] = None,
    ):
        super().__init__(transformer_wrappers)
        self.attention_layers_by_xcoder_id: Dict[int, Dict[str, TransformerStack]] = attention_layer_blocks
        self.token_embs: Dict[str, nn.Embedding] = token_embs
        self.active_task: Optional[str] = None
        self.per_component_post_emb_norms = per_component_post_emb_norms or {}
        self.per_component_pos_embs = per_component_pos_embs or {}
        self.per_component_project_embs = per_component_project_embs or {}
        self.per_component_to_logits = per_component_to_logits or {}

    def activate(self, task_id: str, adapter_ids=None) -> NativeTransformerWrapper:
        self.active_task = task_id
        return self[task_id]

    def get_attention_layers_by_task_id(self, task_id: str, layer_stack_index: int) -> TransformerStack:
        return self[task_id].stacks[layer_stack_index]

    def get_attention_layers_by_xcoder_id(self, layer_stack_index: int, xcoder_id: str) -> TransformerStack:
        return self.attention_layers_by_xcoder_id[layer_stack_index][xcoder_id]

    def get_embedding_by_task_id(self, task_id):
        return self[task_id].token_emb

    def get_embedding_by_lang(self, lang):
        return self.token_embs[lang]

    def get_post_emb_norm_by_component(self, component_key: tuple):
        return self.per_component_post_emb_norms[component_key]

    def get_pos_emb_by_component(self, component_key: tuple):
        return self.per_component_pos_embs.get(component_key)

    def get_project_emb_by_component(self, component_key: tuple):
        return self.per_component_project_embs.get(component_key)

    def get_to_logits_by_component(self, component_key: tuple):
        return self.per_component_to_logits.get(component_key)

    # Lack of forward is intentional: call forward on the return value of activate
