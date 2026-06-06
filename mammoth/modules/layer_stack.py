from torch import nn
from typing import Optional, Dict

from mammoth.modules.transformer import NativeTransformerWrapper, TransformerStack


class StackXcoder(nn.Module):
    """
    Switches between different NativeTransformerWrappers depending on the task.

    StackXcoder is the single registered owner of all shared parameters.
    NativeTransformerWrapper instances in task_wrappers hold plain Python
    references to the shared modules (not nn.Module children), so state_dict()
    contains exactly one copy of each shared parameter.
    """

    def __init__(
        self,
        task_wrappers: Dict[str, NativeTransformerWrapper],
        attention_layer_blocks: Dict[int, Dict[str, TransformerStack]],
        token_embs: Dict[str, nn.Embedding],
        shared_stacks: nn.ModuleDict,
        shared_token_embs: nn.ModuleDict,
        shared_post_emb_norms: nn.ModuleDict,
        shared_rotary_embs: nn.ModuleDict,
        shared_to_logits: nn.ModuleDict,
    ):
        super().__init__()
        # Single owners of all shared parameters.
        self.shared_stacks = shared_stacks
        self.shared_token_embs = shared_token_embs
        self.shared_post_emb_norms = shared_post_emb_norms
        self.shared_rotary_embs = shared_rotary_embs
        self.shared_to_logits = shared_to_logits
        # Lightweight routing views — no parameters of their own (only emb_dropout).
        self.task_wrappers = nn.ModuleDict(task_wrappers)

        # Plain-dict lookups for convenience (no ownership).
        self.attention_layers_by_xcoder_id: Dict[int, Dict[str, TransformerStack]] = attention_layer_blocks
        # token_embs is kept as a plain dict mirror of shared_token_embs for
        # legacy access (e.g. embedding sharing in build_model).
        self.token_embs: Dict[str, nn.Embedding] = token_embs
        self.active_task: Optional[str] = None

    def activate(self, task_id: str, adapter_ids=None) -> NativeTransformerWrapper:
        self.active_task = task_id
        return self.task_wrappers[task_id]

    def get_attention_layers_by_task_id(self, task_id: str, layer_stack_index: int) -> TransformerStack:
        return self.task_wrappers[task_id]._stacks[layer_stack_index]

    def get_attention_layers_by_xcoder_id(self, layer_stack_index: int, xcoder_id: str) -> TransformerStack:
        return self.attention_layers_by_xcoder_id[layer_stack_index][xcoder_id]

    def get_embedding_by_task_id(self, task_id):
        return self.task_wrappers[task_id]._token_emb

    def get_embedding_by_lang(self, lang):
        return self.token_embs[lang]

    def get_post_emb_norm_by_component(self, component_key: tuple):
        key = '__'.join(component_key)
        return self.shared_post_emb_norms[key]

    def get_pos_emb_by_component(self, component_key: tuple):
        key = '__'.join(component_key)
        return self.shared_rotary_embs[key] if key in self.shared_rotary_embs else None

    def get_to_logits_by_component(self, component_key: tuple):
        key = '__'.join(component_key)
        return self.shared_to_logits[key] if key in self.shared_to_logits else None

    # Lack of forward is intentional: call forward on the return value of activate
