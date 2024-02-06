"""
Inject small bottleneck layers with residual connection
into an already trained network, to adapt it for a new task.
"""

import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from collections import defaultdict

from mammoth.modules import TransformerEncoder
from mammoth.modules import TransformerDecoder
from mammoth.rmsnorm_torch import RMSNorm


class AdapterLayer(ABC, nn.Module):
    """
    A single adapter layer module

    Implements Simple, Scalable Adaptation for Neural Machine Translation
    (https://arxiv.org/abs/1909.08478)
    See also fairseq implementation:
    https://github.com/ahmetustun/fairseq/blob/master/fairseq/modules/adapter_layer.py
    """

    def __init__(self, input_dim, hidden_dim, pfeiffer=False, init='small', layernorm='layernorm'):
        super().__init__()
        # Omit LayerCache
        self._does_not_need_cache = True

        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.pfeiffer = pfeiffer
        if not self.pfeiffer:
            if layernorm == 'rmsnorm':
                self.layer_norm = RMSNorm(input_dim, eps=1e-6)
            else:
                self.layer_norm = nn.LayerNorm(input_dim, eps=1e-6)

        if init == 'small' or 'init' == 'bert':
            if init == 'small':
                almost_zero = 1e-5
                delta = 1e-6

                def init_fn(tensor):
                    nn.init.uniform_(
                        tensor,
                        almost_zero - delta, almost_zero + delta
                    )
            elif init == 'bert':

                def init_fn(tensor):
                    nn.init.normal_(tensor, mean=0.0, std=0.02)

            # Init up.
            init_fn(self.up_proj.weight)
            init_fn(self.up_proj.bias)

            # Init down.
            init_fn(self.down_proj.weight)
            init_fn(self.down_proj.bias)

    def forward(self, x):
        if self.pfeiffer:
            y = self.down_proj(x)
            y = F.relu(y)
            y = self.up_proj(y)
        else:
            y = self.layer_norm(x)
            y = self.down_proj(y)
            y = F.relu(y)
            y = self.up_proj(y)
            y = x + y
        return y


class EncoderAdapterLayer(AdapterLayer):
    # same call signature as TransformerEncoderLayer
    def forward(self, inputs, mask):
        out = super().forward(inputs)
        return out


class DecoderAdapterLayer(AdapterLayer):
    # same call signature as TransformerDecoderLayer
    def forward(
        self,
        output,
        src_memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        step=None,
        with_align=False,
        future=False,
    ):
        output = super().forward(output)
        attn = None
        attn_align = None
        return output, attn, attn_align


class Adapter(nn.Module):
    """
    A container for one or several AdapterLayers,
    together with layer indices for injecting into the base network.
    """

    def __init__(self, adapter_group: str, sub_id: str):
        super().__init__()
        self.name = self._name(adapter_group, sub_id)
        # mapping layer_idx -> ModuleList of AdapterLayer to inject after that layer
        self.adapter_layers = nn.ModuleDict()
        self._adapted_layer_indices = set()

    @staticmethod
    def _name(adapter_group: str, sub_id: str) -> str:
        assert isinstance(adapter_group, str), f'Expecting str, not {adapter_group}'
        assert isinstance(sub_id, str), f'Expecting str, not {sub_id}'
        return f'adapter_{adapter_group}_{sub_id}'

    def add_layer(self, layer_idx, adapter_layer: AdapterLayer):
        self._adapted_layer_indices.add(layer_idx)
        layer_idx_str = f'layer{layer_idx}'
        if layer_idx_str not in self.adapter_layers:
            self.adapter_layers[layer_idx_str] = nn.ModuleList()
        self.adapter_layers[layer_idx_str].append(adapter_layer)

    def get_layers(self):
        return self.adapter_layers.items()

    def __repr__(self):
        return f'<Adapter {self.name} with layers {self.adapter_layers}>'


class TransformerAdapterMixin:
    """
    Mixin to manage one or several Adapters
    for a TransformerEncoder or TransformerDecoder.
    """

    def __init__(self, *args, **kwargs):
        # run init of next parallel inheritance class
        super(TransformerAdapterMixin, self).__init__(*args, **kwargs)
        self.adapters = nn.ModuleDict()
        self.active = set()

    def freeze_base_model(self, requires_grad=False):
        adapter_parameters = {name for name, p in self.adapters.named_parameters()}
        for name, p in self.named_parameters():
            if name not in adapter_parameters:
                # freeze everything except the adapter parameters
                p.requires_grad = requires_grad

    def get_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        return self.adapters[name]

    def add_adapter(self, adapter_group: str, sub_id: str, adapter: Adapter):
        name = Adapter._name(adapter_group, sub_id)
        if name in self.adapters:
            raise ValueError(f'Duplicate Adapter "{name}"')
        max_layer_index = max(adapter._adapted_layer_indices)
        if not self._check_n_layers(max_layer_index):
            raise ValueError(f'Invalid number of layers {max_layer_index} in Adapter "{name}"')
        self.adapters[name] = adapter

    def _check_n_layers(self, max_layer_index):
        """Override this"""
        return True

    def deactivate_adapters(self):
        self.active = set()

    def activate_adapter(self, adapter_group: str, sub_id: str):
        name = Adapter._name(adapter_group, sub_id)
        if name not in self.adapters:
            raise ValueError(
                f'Nonexistent Adapter "{name}". '
                f'Should be one of: {" ".join(self.adapters.keys())}'
            )
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

    def _inject_adapters(self, base_layers):
        active_layers = self._merge_active_adapters()
        result = []
        for layer_idx, base_layer in enumerate(base_layers):
            layer_idx = f'layer{layer_idx}'
            result.append(base_layer)
            if layer_idx in active_layers:
                result.extend(active_layers[layer_idx])
        return result


class AdaptedTransformerEncoder(TransformerAdapterMixin, TransformerEncoder):
    def _forward_loop(self, out, mask):
        injected = self._inject_adapters(self.transformer)
        for layer in injected:
            out = layer(out, mask)
        return out

    def _check_n_layers(self, max_layer_index):
        return max_layer_index <= len(self.transformer)

    def state_dict(self, *args, include_adapters=False, **kwargs):
        if not include_adapters:
            # hide adapters
            omitted_adapters = self.adapters
            self.adapters = None

        result = TransformerEncoder.state_dict(self, *args, **kwargs)

        if not include_adapters:
            # unhide adapters
            self.adapters = omitted_adapters

        return result


class AdaptedTransformerDecoder(TransformerAdapterMixin, TransformerDecoder):
    def forward(self, *args, **kwargs):
        self._injected = self._inject_adapters(self.transformer_layers)
        return super().forward(*args, **kwargs)

    def _get_layers(self):
        return self._injected

    def _check_n_layers(self, max_layer_index):
        return max_layer_index <= len(self.transformer_layers)

    def state_dict(self, *args, include_adapters=False, **kwargs):
        if not include_adapters:
            # hide adapters
            omitted_adapters = self.adapters
            self.adapters = None

        result = TransformerDecoder.state_dict(self, *args, **kwargs)

        if not include_adapters:
            # unhide adapters
            self.adapters = omitted_adapters

        return result
