import torch.nn as nn
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Set, Any, Optional, Dict

from mammoth.models import NMTModel


class DistributedComponentBuilder:
    """
    Deduplicates DistributedComponents, merging the data from each copy.
    """
    def __init__(self) -> None:
        self.components: Dict[str, DistributedComponent] = dict()

    def add(self, component):
        name = component.get_name()
        if name not in self.components:
            # new component
            self.components[name] = component
        else:
            # already seen component must be merged
            old_component = self.components[name]
            assert type(old_component) == type(component), \
                f'Unexpected type {name}: {old_component} != {component}'
            assert old_component.group is None
            assert component.group is None
            # Merge the sets of new component into the old component
            old_component.global_ranks.update(component.global_ranks)
            old_component.task_ids.update(component.task_ids)

    def __iter__(self):
        result = []
        for key in sorted(self.components.keys()):
            result.append(self.components[key])
        return iter(result)


class Side(Enum):
    encoder = auto()
    decoder = auto()


# mypy doesn't like abstract dataclasses
@dataclass      # type: ignore
class DistributedComponent(ABC):
    """
    Represents a model component that may be distributed across several
    devices according to some parameter sharing pattern.
    """
    # This was implemented as a separate dataclass instead of making it a mixin
    # of the nn.Module. The main reason is the need to create and use the
    # DistributedComponents also in contexts where an initialized model is not
    # (yet) available: 1) in the dataloader, 2) (after future refactoring) when
    # creating the Modules that the model consists of.

    global_ranks: Set[int]
    task_ids: Set[str]
    # distributed communication group object, or None if on a single device
    group: Optional[Any]

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_module(self, model: NMTModel) -> nn.Module:
        pass

    def named_parameters(self, model: NMTModel):
        module = self.get_module(model)
        yield from module.named_parameters()

    def state_dict(self, model: NMTModel, prefix='', keep_vars=False):
        module = self.get_module(model)
        return module.state_dict(prefix=prefix, keep_vars=keep_vars)

    def load_state_dict(self, model: NMTModel, state_dict: Dict[str, Any]):
        module = self.get_module(model)
        return module.load_state_dict(state_dict)

    @property
    def min_rank(self) -> int:
        return min(self.global_ranks)

    def needs_communication(self) -> bool:
        # if the component needs communication, a group must be set
        return self.group is not None


@dataclass  # type: ignore
class DistributedTransformerWrapper(DistributedComponent, ABC):
    """Represents a distributed TransformerWrapper object from x-transformers"""
    task_id: str
    side: Side

    def get_name(self) -> str:
        return f'{self.side.name}_{self.task_id}'

    def get_module(self, model: NMTModel) -> nn.Module:
        parent = model.encoder if self.side == Side.encoder else model.decoder
        transformer_wrapper = parent[self.task_id]
        return transformer_wrapper

    def named_parameters(self, model: NMTModel):
        module = self.get_module(model)
        for name, p in module.named_parameters():
            # TransformerWrapper contains the AttentionLayers and the embs.
            # however, we want to treat these as distinct DistributedComponents
            if name.startswith('attn_layers.'):
                continue
            if name.startswith('token_emb.'):
                continue
            yield name, p

    def state_dict(self, model: NMTModel, prefix='', keep_vars=False) -> Dict[str, Any]:
        module = self.get_module(model)
        destination: Dict[str, Any] = OrderedDict()
        for name, sub_module in module._modules.items():
            if name in {'attn_layers', 'token_emb'}:
                # stored separately
                continue
            sub_module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        return destination

    def load_state_dict(self, model: NMTModel, state_dict: Dict[str, Any]):
        module = self.get_module(model)
        mismatch = module.load_state_dict(state_dict, strict=False)
        missing_keys = [
            name for name in mismatch.missing_keys
            if not any(name.startswith(prefix) for prefix in ('attn_layers.', 'token_emb.'))
        ]
        return mismatch._replace(missing_keys=missing_keys)


@dataclass  # type: ignore
class DistributedAttentionLayersBlock(DistributedComponent, ABC):
    """Represents a distributed AdaptedAttentionLayers object"""
    layer_stack_index: int
    xcoder_id: str

    @property
    @abstractmethod
    def side(self) -> Side:
        pass

    def get_name(self) -> str:
        return f'{self.side.name}_{self.layer_stack_index}_{self.xcoder_id}'

    def named_parameters(self, model: NMTModel):
        module = self.get_module(model)
        for name, p in module.named_parameters():
            # encoders and decoders contain embeddings and adapters as submodules
            # however, we want to treat these as distinct DistributedComponents
            if 'adapter' in name:
                continue
            yield name, p

    def state_dict(self, model: NMTModel, prefix='', keep_vars=False) -> Dict[str, Any]:
        module = self.get_module(model)
        destination: Dict[str, Any] = OrderedDict()
        for name, sub_module in module.get_sub_modules().items():
            if name == 'adapters':
                # Adapters are stored separately
                continue
            sub_module.state_dict(destination=destination, prefix=prefix + name + '.', keep_vars=keep_vars)
        return destination

    def load_state_dict(self, model: NMTModel, state_dict: Dict[str, Any]):
        module = self.get_module(model)
        mismatch = module.load_state_dict(state_dict, strict=False)
        missing_keys = [
            name for name in mismatch.missing_keys
            if not name.startswith('layers.')
        ]
        return mismatch._replace(missing_keys=missing_keys)


@dataclass
class DistributedEncoderAttentionLayersBlock(DistributedAttentionLayersBlock):
    """Represents a distributed encoder-side AdaptedAttentionLayers"""
    @property
    def side(self) -> Side:
        return Side.encoder

    @property
    def encoder_id(self) -> str:
        return self.xcoder_id

    def get_module(self, model: NMTModel) -> nn.Module:
        return model.encoder.get_attention_layers_by_xcoder_id(self.layer_stack_index, self.xcoder_id)


@dataclass
class DistributedDecoderAttentionLayersBlock(DistributedAttentionLayersBlock):
    """Represents a distributed decoder-side AdaptedAttentionLayers"""
    @property
    def side(self) -> Side:
        return Side.decoder

    @property
    def decoder_id(self) -> str:
        return self.xcoder_id

    def get_module(self, model: NMTModel) -> nn.Module:
        return model.decoder.get_attention_layers_by_xcoder_id(self.layer_stack_index, self.xcoder_id)


@dataclass
class DistributedEmbedding(DistributedComponent):
    """Represents a distributed Embeddings object"""
    side: Side
    lang: str

    def get_name(self) -> str:
        side_str = 'src' if self.side == Side.encoder else 'tgt'
        return f'{side_str}_embeddings_{self.lang}'

    def get_module(self, model: NMTModel) -> nn.Module:
        if self.side == Side.encoder:
            return model.encoder.get_embedding_by_lang(self.lang)
        else:
            return model.decoder.get_embedding_by_lang(self.lang)


@dataclass
class DistributedAdapter(DistributedComponent):
    """Represents a distributed Adapter object"""
    # Can't use parent object of type DistributedAttentionLayersBlock: that refers to a
    # specific module, while the adapter is for the entire layerstack slot
    side: Side
    layer_stack_index: int
    adapter_group: str
    sub_id: str

    def get_name(self) -> str:
        return f'{self.side.name}_adapter_{self.layer_stack_index}_{self.adapter_group}_{self.sub_id}'

    def get_module(self, model: NMTModel) -> nn.Module:
        if self.side == Side.encoder:
            return model.encoder.get_adapter(f'adapter_{self.adapter_group}_{self.sub_id}')
        else:
            return model.decoder.get_adapter(f'adapter_{self.adapter_group}_{self.sub_id}')


@dataclass
class DistributedAttentionBridge(DistributedComponent):
    """
    Represents the attention bridge.
    Even though the sharing of the attention bridge is not configurable (it is always fully shared),
    representing it as a DistributedComponent allows for code reuse.
    """
    def get_name(self) -> str:
        return 'attention_bridge'

    def get_module(self, model: NMTModel) -> Optional[nn.Module]:
        return model.attention_bridge


@dataclass
class DistributedComponentGradientSync:
    """
    Represents a gradient communication action to be performed on a particular model component.
    Other actions (init broadcast, optimizer step, checkpoint saving) do not need additional metadata,
    and can be represented by just the DistributedComponent.
    """
    component: DistributedComponent
    # True: has a real gradient that needs to be communicated
    # False: send a zero dummy gradient, receive and apply gradient from others
    has_local_gradient: bool
    # Normalization denominator
    gradient_norm: int
