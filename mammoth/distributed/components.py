import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Any, Optional


class Side(Enum):
    encoder = auto()
    decoder = auto()


@dataclass
class DistributedComponent(ABC):
    """
    Represents a model component that may be distributed across several
    devices according to some parameter sharing pattern.
    """
    module: nn.Module
    ranks: List[int]
    # distributed communication group object, or None if on a single device
    group: Optional[Any]

    @abstractmethod
    def get_name(self) -> str:
        pass

    def named_parameters(self):
        yield from self.module.named_parameters()

    def min_rank(self) -> int:
        return min(self.ranks)


@dataclass
class DistributedXCoder(DistributedComponent):
    side: Side
    layer_stack_index: int
    xcoder_id: str

    def get_name(self) -> str:
        return f'{self.side.name}_{self.layer_stack_index}_{self.xcoder_id}'

    def named_parameters(self):
        for name, p in self.module.named_parameters():
            # encoders and decoders contain embeddings and adapters as submodules
            # however, we want to treat these as distinct DistributedComponents
            if 'embeddings' not in name and 'adapter' not in name:
                yield name, p


@dataclass
class DistributedEmbedding(DistributedComponent):
    side: Side
    lang: str

    def get_name(self) -> str:
        side_str = 'src' if self.side == Side.encoder else 'tgt'
        return f'{side_str}_embeddings_{self.lang}'


@dataclass
class DistributedGenerator(DistributedComponent):
    lang: str

    def get_name(self) -> str:
        return f'generator_{self.lang}'


@dataclass
class DistributedAdapter(DistributedComponent):
    side: Side
    layer_stack_index: int
    adapter_group: str
    sub_id: str

    def get_name(self) -> str:
        return f'{self.side.name}_adapter_{self.layer_stack_index}_{self.adapter_group}_{self.sub_id}'


@dataclass
class DistributedAttentionBridge(DistributedComponent):
    def get_name(self) -> str:
        return 'attention_bridge'


@dataclass
class DistributedComponentAction:
    """
    Represents an action to be performed on a particular model component.
    Actions include init broadcast, gradient communication, optimizer step, checkpoint saving.
    """
    component: DistributedComponent


@dataclass
class DistributedComponentActionGradient(DistributedComponentAction):
    # True: has a real gradient that needs to be communicated
    # False: send a zero dummy gradient, receive gradient from others
    has_local_gradient: bool
    gradient_norm: int
