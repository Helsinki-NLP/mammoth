import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Set, Any, Optional, Dict

from mammoth.models import NMTModel


class DistributedComponentBuilder:
    def __init__(self):
        self.components: Dict[str, DistributedComponent] = dict()

    def add(self, component):
        name = component.get_name()
        if name not in self.components:
            # new component
            self.components[name] = component
        else:
            # already seen component must be merged
            old_component = self.components[name]
            assert type(old_component) == type(component)
            assert old_component.group is None
            assert component.group is None
            old_component.global_ranks.update(component.global_ranks)

    def __iter__(self):
        result = []
        for key in sorted(self.components.keys()):
            result.append(self.components[key])
        return iter(result)


class Side(Enum):
    encoder = auto()
    decoder = auto()


@dataclass
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

    def min_rank(self) -> int:
        return min(self.global_ranks)


@dataclass
class DistributedXCoder(DistributedComponent, ABC):
    layer_stack_index: int
    xcoder_id: str

    def get_name(self) -> str:
        return f'{self.side.name}_{self.layer_stack_index}_{self.xcoder_id}'

    def named_parameters(self, model: NMTModel):
        module = self.get_module(model)
        for name, p in module.named_parameters():
            # encoders and decoders contain embeddings and adapters as submodules
            # however, we want to treat these as distinct DistributedComponents
            if 'embeddings' not in name and 'adapter' not in name:
                yield name, p


@dataclass
class DistributedEncoder(DistributedXCoder):
    @property
    def side(self) -> Side:
        return Side.encoder

    @property
    def encoder_id(self) -> str:
        return self.xcoder_id

    def get_module(self, model: NMTModel) -> nn.Module:
        return model.encoder.get_submodule(self.layer_stack_index, self.xcoder_id)


@dataclass
class DistributedDecoder(DistributedXCoder):
    @property
    def side(self) -> Side:
        return Side.encoder

    @property
    def decoder_id(self) -> str:
        return self.xcoder_id

    def get_module(self, model: NMTModel) -> nn.Module:
        return model.decoder.get_submodule(self.layer_stack_index, self.xcoder_id)


@dataclass
class DistributedEmbedding(DistributedComponent):
    side: Side
    lang: str

    def get_name(self) -> str:
        side_str = 'src' if self.side == Side.encoder else 'tgt'
        return f'{side_str}_embeddings_{self.lang}'

    def get_module(self, model: NMTModel) -> nn.Module:
        if self.side == Side.encoder:
            return model.encoder.embeddings[f'embeddings_{self.lang}']
        else:
            return model.decoder.embeddings[f'embeddings_{self.lang}']


@dataclass
class DistributedGenerator(DistributedComponent):
    lang: str

    def get_name(self) -> str:
        return f'generator_{self.lang}'

    def get_module(self, model: NMTModel) -> nn.Module:
        return model.generator[f'generator_{self.lang}']


@dataclass
class DistributedAdapter(DistributedComponent):
    # Can't use parent object of type DistributedXCoder: that refers to a
    # specific module, while the adapter is for the entire layerstack slot
    side: Side
    layer_stack_index: int
    adapter_group: str
    sub_id: str

    def get_name(self) -> str:
        return f'{self.side.name}_adapter_{self.layer_stack_index}_{self.adapter_group}_{self.sub_id}'

    def get_module(self, model: NMTModel) -> nn.Module:
        if self.side == Side.encoder:
            model.encoder.get_adapter(self.adapter_group, self.sub_id)
        else:
            model.decoder.get_adapter(self.adapter_group, self.sub_id)


@dataclass
class DistributedAttentionBridge(DistributedComponent):
    def get_name(self) -> str:
        return 'attention_bridge'

    def get_module(self, model: NMTModel) -> Optional[nn.Module]:
        return self.model.attention_bridge


@dataclass
class DistributedComponentAction:
    """
    Represents an action to be performed on a particular model component.
    Actions include init broadcast, gradient communication, optimizer step, checkpoint saving.
    """
    component: DistributedComponent


@dataclass
class DistributedComponentActionWithGradient(DistributedComponentAction):
    # True: has a real gradient that needs to be communicated
    # False: send a zero dummy gradient, receive gradient from others
    has_local_gradient: bool
    gradient_norm: int
