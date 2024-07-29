"""
Inject small bottleneck layers with residual connection.
Can be applied during main training,
or injected into an already trained network to adapt it for a new task.
"""

import torch
import torch.nn as nn
from collections import defaultdict
from typing import Union, Set
from functools import partial

from x_transformers.x_transformers import (
    AttentionLayers,
    FeedForward,
    Residual,
    SimpleRMSNorm,
)


class FeedForwardAdapterLayer(nn.Module):
    """A separate adapter layer injected after a FeedForward. Has its own norms."""
    def __init__(self, dim, pre_norm=True, sandwich_norm=False, **kwargs):
        super().__init__()
        norm_fn = partial(SimpleRMSNorm, dim)
        self.pre_branch_norm = norm_fn() if pre_norm else None
        self.post_branch_norm = norm_fn() if sandwich_norm else None
        self.post_main_norm = norm_fn() if not pre_norm else None
        self.ff = FeedForward(dim, **kwargs)
        self.residual = Residual(dim, scale_residual=False)

    @property
    def is_wrapper(self):
        return False

    def as_layer_struct(self):
        return nn.ModuleList([
            nn.ModuleList([
                self.pre_branch_norm,
                self.post_branch_norm,
                self.post_main_norm,
            ]),
            self.ff,
            self.residual,
        ])


class LoraAdapterLayer(nn.Module):
    """A LoRA adapter layer wrapping a FeedForward. No additional norms."""
    def __init__(
        self,
        dim,
        dim_out=None,
        r=8,
        alpha=None
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim
        alpha = alpha if alpha is not None else r
        self.scale = alpha / r

        self.A = nn.Parameter(torch.randn(dim, r))
        self.B = nn.Parameter(torch.zeros(r, dim_out))
        self.wrapped_base_layer = None

    @property
    def is_wrapper(self):
        return True

    def wrap(self, base_layer):
        self.wrapped_base_layer = base_layer
        return self

    @property
    def weight(self):
        return (self.A @ self.B) * self.scale

    def forward(self, x):
        return (x @ self.weight) + self.wrapped_base_layer.forward(x)


AdapterLayer = Union[FeedForwardAdapterLayer, LoraAdapterLayer]


class Adapter(nn.Module):
    """
    A container for one or several AdapterLayers,
    together with layer indices for injecting into the base network.
    """

    def __init__(self, adapter_group: str, sub_id: str):
        super().__init__()
        self.adapter_group = adapter_group
        self.sub_id = sub_id
        self.name = self._name(adapter_group, sub_id)
        # mapping layer_idx -> ModuleList of AdapterLayer to inject at that layer
        self.adapter_layers = nn.ModuleDict()
        self._adapted_layer_indices: Set[int] = set()

    @staticmethod
    def _name(adapter_group: str, sub_id: str) -> str:
        assert isinstance(adapter_group, str), f'Expecting str, not {adapter_group}'
        assert isinstance(sub_id, str), f'Expecting str, not {sub_id}'
        return f'adapter_{adapter_group}_{sub_id}'

    def add_layer(self, layer_idx: int, adapter_layer: AdapterLayer):
        self._adapted_layer_indices.add(layer_idx)
        layer_idx_str = f'layer{layer_idx}'
        if layer_idx_str not in self.adapter_layers:
            self.adapter_layers[layer_idx_str] = nn.ModuleList()
        self.adapter_layers[layer_idx_str].append(adapter_layer)

    def get_layers(self):
        return self.adapter_layers.items()

    def __repr__(self):
        return f'<Adapter {self.name} with layers {self.adapter_layers}>'


class AdaptedAttentionLayers(AttentionLayers):
    """
    Extends an x_transformers.AttentionLayers block
    with the ability to inject additional layers
    or dymically wrap layers in LoRA wrappers.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_layer_types = tuple(self.layer_types)
        self._base_layers = nn.ModuleList(self.layers)
        self._base_layer_dropouts = tuple(self.layer_dropouts)

    def freeze_base_model(self, requires_grad=False):
        for name, p in self._base_layers.named_parameters():
            p.requires_grad = requires_grad

    def get_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        return self.adapters[name]

    def add_adapter(self, adapter: Adapter):
        if adapter.name in self.adapters:
            raise ValueError(f'Duplicate Adapter "{adapter.name}"')
        max_layer_index = max(adapter._adapted_layer_indices)
        n_feedforwards = sum(layer_type == 'f' for layer_type in self._base_layer_types)
        if max_layer_index >= n_feedforwards:
            raise ValueError(
                f'Invalid layer number {max_layer_index} in Adapter "{adapter.name}". '
                f'There are ony {n_feedforwards} layers.'
            )
        self.adapters[adapter.name] = adapter

    def deactivate_adapters(self):
        self.active = set()

    def activate_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        if name in self.adapters:
            self.active.add(name)

    def _merge_active_adapters(self):
        """
        Returns a single mapping layer_idx -> list of AdapterLayer,
        containing the layers of all currently active adapters
        """
        active_adapters = [
            adapter for name, adapter in self.adapters.items()
            if name in self.active
        ]
        merged = defaultdict(list)
        for adapter in active_adapters:
            for layer_idx, layers in adapter.get_layers():
                merged[layer_idx].extend(layers)
        return merged

    def _inject_adapters(self):
        adapted_layer_types = []
        adapted_layers = nn.ModuleList()
        adapted_layer_dropouts = []
        i = 0
        for layer_type, layer_struct, layer_dropout in zip(
            self._base_layer_types,
            self._base_layers,
            self._base_layer_dropouts,
        ):
            adapter_layers_by_index = self._merge_active_adapters()
            if layer_type == 'f':
                # Adapters apply to feedforward layers
                adapter_layers = adapter_layers_by_index[i]
                for adapter_layer in adapter_layers:
                    if adapter_layer.is_wrapper:
                        # LoRA wraps the base ff
                        adapted_layer_types.append('f')
                        adapted_layers.append(adapter_layer.wrap(layer_struct))
                        adapted_layer_dropouts.append(layer_dropout)
                    else:
                        # FeedForwards are injected after the base ff
                        adapted_layer_types.append('f')
                        adapted_layer_types.append('f')
                        adapted_layers.append(layer_struct)
                        adapted_layers.append(adapter_layer.as_layer_struct())
                        adapted_layer_dropouts.append(layer_dropout)
                        adapted_layer_dropouts.append(layer_dropout)
                i += 1
            else:
                # Attetion layers are unmodified
                adapted_layer_types.append(layer_type)
                adapted_layers.append(layer_struct)
                adapted_layer_dropouts.append(layer_dropout)
        self.layer_types = adapted_layer_types
        self.layers = adapted_layers
        self.layer_dropouts = adapted_layer_dropouts

    def forward(self, *args, **kwargs):
        self._inject_adapters()
        return super().forward(*args, **kwargs)
