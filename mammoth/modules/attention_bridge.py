from __future__ import division

import warnings

import torch
import torch.nn as nn
from mammoth.rmsnorm_torch import RMSNorm
from mammoth.modules.transformer_encoder import TransformerEncoderLayer

from mammoth.modules.multi_headed_attn import MultiHeadedAttention


class BaseAttentionBridgeLayer(nn.Module):
    """
    Base class for attention bridge layers
    """

    @property
    def is_fixed_length(self) -> bool:
        """whether this layer always produce output of the same size"""
        raise NotImplementedError


class AttentionBridgeNorm(nn.Module):
    """Norm component (shared implementation across architectures)."""

    def __init__(self, normalized_shape, ab_layer_norm_type):
        super().__init__()
        if ab_layer_norm_type == 'rmsnorm':
            self.norm = RMSNorm(normalized_shape, eps=1e-6)
        elif ab_layer_norm_type == 'layernorm':
            self.norm = nn.LayerNorm(normalized_shape, eps=1e-6)
        elif ab_layer_norm_type == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError(f"ab_layer_norm_type `{ab_layer_norm_type}` is not recognized.")

    def forward(self, input):
        return self.norm(input)  # possibly an nn.Identity


class PerceiverAttentionBridgeLayer(BaseAttentionBridgeLayer):
    def __init__(
        self, latent_size, ff_size, fixed_seqlen, n_heads, attention_dropout, max_relative_positions, norm_type
    ):
        super().__init__()
        self.latent_array = nn.Parameter(torch.zeros(fixed_seqlen, latent_size))
        self.cross_attention_block = MultiHeadedAttention(
            n_heads,
            latent_size,
            attention_dropout,
            max_relative_positions
        )
        self.cross_attention_norm = AttentionBridgeNorm(latent_size, norm_type)
        self.cross_ff_block = nn.Sequential(
            nn.Linear(latent_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, latent_size),
        )
        self.cross_ff_norm = AttentionBridgeNorm(latent_size, norm_type)
        self.self_attention_block = MultiHeadedAttention(
            n_heads,
            latent_size,
            attention_dropout,
            max_relative_positions
        )
        self.self_attention_norm = AttentionBridgeNorm(latent_size, norm_type)
        self.self_ff_block = nn.Sequential(
            nn.Linear(latent_size, ff_size),
            nn.ReLU(),
            nn.Linear(ff_size, latent_size),
        )
        self.self_ff_norm = AttentionBridgeNorm(latent_size, norm_type)

    @classmethod
    def from_opts(cls, opts):
        return cls(
            opts.model_dim,
            opts.hidden_ab_size,
            opts.ab_fixed_length,
            opts.heads,
            opts.attention_dropout[0],
            opts.max_relative_positions,
            opts.ab_layer_norm,
        )

    @property
    def is_fixed_length(self):
        return True

    def forward(self, intermediate_output, encoder_output, mask=None):
        """
        intermediate_output: ``(batch, seq_len, hidden_dim)``
        enc_output: original input to the attention bridge
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        S, B, F = encoder_output.shape
        if intermediate_output is not None:
            cross_query = intermediate_output
        else:
            cross_query = self.latent_array.unsqueeze(0).expand(B, -1, -1)
        encoder_output = encoder_output.transpose(0, 1)

        # sublayer 1: projects to fixed size
        cross_attention_output, alphas = self.cross_attention_block(
            encoder_output, encoder_output, cross_query, mask=mask, attn_type='context'
        )
        cross_attention_output = self.cross_attention_norm(cross_attention_output + cross_query)
        cross_attention_output = self.cross_ff_block(cross_attention_output) + cross_attention_output
        cross_attention_output = self.cross_ff_norm(cross_attention_output)

        # sublayer 2: performs self-attention
        self_attention_output, _ = self.self_attention_block(
            cross_attention_output, cross_attention_output, cross_attention_output, mask=None, attn_type='self'
        )
        self_attention_output = self.self_attention_norm(self_attention_output + cross_attention_output)
        self_attention_output = self.self_ff_block(self_attention_output) + self_attention_output
        self_attention_output = self.self_ff_norm(self_attention_output)

        return alphas, self_attention_output


class LinAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """
    Multi-headed attention. Bridge between encoders->decoders.
    Based on Lin et al. (2017) A structured self-attentive sentence embedding
    """

    def __init__(
        self,
        hidden_size,
        attention_heads,
        hidden_ab_size,
        model_type,
        model_dim,
        ab_layer_norm=None,
    ):
        """Attention Heads Layer:"""
        super(LinAttentionBridgeLayer, self).__init__()
        d = hidden_size
        u = hidden_ab_size
        r = attention_heads
        self.dd = u
        self.model_type = model_type
        if self.model_type != "text":
            d = model_dim
        self.ws1 = nn.Linear(d, u, bias=True)
        self.ws2 = nn.Linear(u, r, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = r
        self.M = None  # TODO : remove
        self.norm = AttentionBridgeNorm(d, ab_layer_norm)

    @classmethod
    def from_opts(cls, opts):
        """Alternate constructor."""
        return cls(
            opts.model_dim,
            opts.ab_fixed_length,
            opts.hidden_ab_size,
            opts.model_type,
            opts.model_dim,
            opts.ab_layer_norm,
        )

    def forward(self, intermediate_output, encoder_output, mask=None):
        """
        intermediate_output: ``(batch, seq_len, hidden_dim)``
        enc_output: unused
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        B, L, H = intermediate_output.size()  # [bsz, len, nhid]
        compressed_embeddings = intermediate_output.reshape(-1, H)  # [bsz*len, nhid*2]
        hbar = self.ws1(compressed_embeddings)  # [bsz*len, attention-unit]

        hbar = self.relu(hbar)

        alphas = self.ws2(hbar).view(B, L, -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        # previously, we would penalize alphas if "text" and only the 1st AB-layer
        # but we should only consider penalizing if there's something to mask
        if mask is not None:
            mask_reshaped = mask.view(B, 1, L).expand(B, self.attention_hops, L)  # [bsz, hop, len]
            penalized_alphas = alphas + (-10000 * (mask_reshaped == 1).float())  # [bsz, hop, len] + [bsz, hop, len]
            alphas = penalized_alphas

        alphas = self.softmax(alphas.view(-1, L))  # [bsz*hop, len]
        alphas = alphas.view(B, self.attention_hops, L)  # [bsz, hop, len]
        output = torch.bmm(alphas, intermediate_output)

        output = self.norm(output)
        # TODO: why cache? not sure what else is looking at layer.M
        self.M = torch.transpose(
            output, 0, 1
        ).contiguous()  # [r,bsz,nhid]       torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        return alphas, output

    @property
    def is_fixed_length(self):
        return True


class SimpleAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """Simple attention based bridge layer using a fixed query matrix.
    This matrix is a learned parameter: the model should learn to probe the
    latent key space to produce coherent mixtures of value vectors.
    """

    def __init__(self, input_size, hidden_size, fixed_seqlen, ab_layer_norm):
        super().__init__()
        self.query_matrix = nn.Parameter(torch.zeros(fixed_seqlen, hidden_size))
        self.keys_proj = nn.Linear(input_size, hidden_size)
        self.values_proj = nn.Linear(input_size, input_size)
        self.d_sqrt = hidden_size**0.5
        self.R = fixed_seqlen
        self.softmax = nn.Softmax(dim=-1)
        self.norm = AttentionBridgeNorm(input_size, ab_layer_norm)

    @property
    def is_fixed_length(self):
        return True

    def forward(self, intermediate_output, encoder_output, mask=None):
        """
        intermediate_output: ``(batch, seq_len, hidden_dim)``
        encoder_output: unused
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        B, L, H = intermediate_output.size()
        R = self.R
        keys = self.keys_proj(intermediate_output)
        values = self.values_proj(intermediate_output)
        raw_scores = (self.query_matrix @ torch.flatten(keys, end_dim=-2).T).view(B, R, L)
        if mask is not None:
            mask_reshaped = mask.view(B, 1, L)
            raw_scores = raw_scores.masked_fill(mask_reshaped, -float('inf'))
        attention_weights = self.softmax(raw_scores / self.d_sqrt)
        output = self.norm(attention_weights @ values)
        return attention_weights, output

    @classmethod
    def from_opts(cls, opts):
        return cls(
            opts.model_dim,
            opts.hidden_ab_size,
            opts.ab_fixed_length,
            opts.ab_layer_norm,
        )


# TODO: for now I've used the basic implementation of TransformerEncoderLayer;
# we could consider an attention-bridge-specific implementation that would allow
# us to control the norm and return alphas if necessary.
class TransformerAttentionBridgeLayer(BaseAttentionBridgeLayer, TransformerEncoderLayer):
    """Using a Transformer encoder layer as a shared component in the attention bridge"""

    def __init__(self, *args, **kwargs):
        TransformerEncoderLayer.__init__(self, *args, **kwargs)

    @property
    def is_fixed_length(self):
        return False

    def forward(self, intermediate_output, encoder_output, mask=None):
        """
        mask: binary mask 1/0 indicating which keys have
        enc_output: ``(batch, seq_len, hidden_dim)``
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        outp = TransformerEncoderLayer.forward(self, intermediate_output, mask)
        return None, outp

    @classmethod
    def from_opts(cls, opts):
        return cls(
            opts.model_dim,
            opts.heads,
            opts.hidden_ab_size,  # d_ff
            # TODO: that list indexing things seems suspicious to me...
            opts.dropout[0],
            opts.attention_dropout[0],
            max_relative_positions=opts.max_relative_positions,
        )


class FeedForwardAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """Simple feedforward bridge component"""

    def __init__(self, input_size, hidden_size, ab_layer_norm):
        super().__init__()
        self.module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            AttentionBridgeNorm(input_size, ab_layer_norm),
        )

    @property
    def is_fixed_length(self):
        return False

    def forward(self, intermediate_output, encoder_output, mask=None):
        """
        intermediate_output: ``(batch, seq_len, hidden_dim)``
        encoder_output: unused
        mask: unused
        """
        return None, self.module(intermediate_output)

    @classmethod
    def from_opts(cls, opts):
        return cls(
            opts.model_dim,
            opts.hidden_ab_size,
            opts.ab_layer_norm,
        )


class AttentionBridge(nn.Module):
    """
    N-layered attention-bridge between encoders->decoders
    """

    def __init__(self, layers):
        """Attention Heads Layer"""
        super(AttentionBridge, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.is_fixed_length = any(x.is_fixed_length for x in layers)

    @classmethod
    def from_opts(cls, opts):
        """Alternate constructor."""
        # convert opts specifications to architectures
        layer_type_to_cls = {
            'lin': LinAttentionBridgeLayer,
            'perceiver': PerceiverAttentionBridgeLayer,
            'simple': SimpleAttentionBridgeLayer,
            'transformer': TransformerAttentionBridgeLayer,
            'feedforward': FeedForwardAttentionBridgeLayer,
        }

        # preconstruct layers using .from_opts(...)
        layers = [layer_type_to_cls[layer_type].from_opts(opts) for layer_type in opts.ab_layers]

        # FIXME: locking-in edge case behavior
        if any(layer == 'perceiver' for layer in opts.ab_layers):
            first_perceiver_index = next(idx for idx, layer in enumerate(opts.ab_layers) if layer == 'perceiver')
            if first_perceiver_index != 0:
                assert any(layer.is_fixed_length for layer in layers[:first_perceiver_index]), \
                    'Unsupported bridge configuration: at least one layer must be fixed-size before perceiver'
            if not all(layer == 'perceiver' for layer in opts.ab_layers):
                warnings.warn('Architecture-mixing not fully supported with perceiver.')
            # FIXME: deleting unused params manually
            for perceiver_layer in layers[1:]:
                perceiver_layer.latent_array = None

        return cls(layers)

    def forward(self, enc_output, mask):
        """Forward pass for the bridge layers"""
        out = enc_output.transpose(0, 1)
        if self.layers and isinstance(self.layers[0], PerceiverAttentionBridgeLayer):
            out = None
        alphas = None
        orig_mask = mask
        for layer in self.layers:
            mask_ = orig_mask if isinstance(layer, PerceiverAttentionBridgeLayer) else mask
            alphas, out = layer(out, enc_output, mask_)
            if layer.is_fixed_length:
                # In this case, we've ensured all batch items have a constant
                # sequence length, so the mask is no longer required.
                mask = None
        out = torch.transpose(out, 0, 1).contiguous()
        return out, alphas  # [hop, bsz, nhid], [bsz, hop, srcseqlen]
