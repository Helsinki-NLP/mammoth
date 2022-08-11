"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
from onmt.rmsnorm_torch import RMSNorm
from onmt.encoders.transformer import TransformerEncoderLayer
#from onmt.modules.position_ffn import ActivationFunction
#import math


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
        self.norm = nn.Identity()
        if ab_layer_norm_type == 'rmsnorm':
            self.norm = RMSNorm(normalized_shape, eps=1e-6)
        elif ab_layer_norm_type == 'layernorm':
            self.norm = nn.LayerNorm(normalized_shape, eps=1e-6)

    def forward(self, input):
        return self.norm(input) # possibly an nn.Identity


class LinAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """
    Multi-headed attention. Bridge between encoders->decoders. Based on Lin et al. (2017) A structured self-attentive sentence embedding
    """
    def __init__(self,
                 hidden_size,
                 attention_heads,
                 hidden_ab_size,
                 model_type,
                 dec_rnn_size,
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
            d = dec_rnn_size
        self.ws1 = nn.Linear(d, u, bias=True)
        self.ws2 = nn.Linear(u, r, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = r
        self.M = None # TODO : remove
        self.norm = AttentionBridgeNorm(d, ab_layer_norm)

    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(
            opt.rnn_size,
            opt.ab_fixed_length,
            opt.hidden_ab_size,
            opt.model_type,
            opt.dec_rnn_size,
            opt.ab_layer_norm,
        )


    def forward(self, enc_output, mask):
        """
        enc_output: ``(batch, seq_len, hidden_dim)``
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        enc_output = enc_output.transpose(0, 1)
        output, alphas = self.mixAtt(enc_output, mask)
        output = self.norm(output)
        # TODO: why cache? not sure what else is looking at layer.M
        self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]       torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        return alphas, output

    @property
    def is_fixed_length(self):
        return True

    # TODO: why is this a helper function?
    def mixAtt(self, outp, mask):
        """Notation based on Lin et al. (2017)"""
        outp = torch.transpose(outp, 0, 1).contiguous()
        B, L, H = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, H)  # [bsz*len, nhid*2]
        hbar = self.ws1(compressed_embeddings)  # [bsz*len, attention-unit]

        alphas = self.ws2(hbar).view(B, L, -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        # previously, we would penalize alphas if "text" and only the 1st AB-layer
        # but we should only consider penalizing if there's something to mask
        if mask is not None:
            mask_reshaped = mask.view(B, 1, L).expand(B, self.attention_hops, L) # [bsz, hop, len]
            penalized_alphas = alphas + (-10000 * (mask_reshaped == 1).float()) # [bsz, hop, len] + [bsz, hop, len]
            alphas = penalized_alphas

        alphas = self.softmax(alphas.view(-1, L))  # [bsz*hop, len]
        alphas = alphas.view(B, self.attention_hops, L)  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas


class SimpleAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """Simple attention based bridge layer using a fixed query matrix.
    This matrix is a learned parameter, so the model should learn to
    """
    def __init__(self, input_size, hidden_size, fixed_seqlen, ab_layer_norm):
        super().__init__()
        self.query_matrix = nn.Parameter(torch.zeros(fixed_seqlen, hidden_size))
        self.keys_proj = nn.Linear(input_size, hidden_size)
        self.values_proj = nn.Linear(input_size, input_size)
        self.d_sqrt = hidden_size ** 0.5
        self.R = fixed_seqlen
        self.softmax = nn.Softmax(dim=-1)
        self.norm = AttentionBridgeNorm(input_size, ab_layer_norm)

    @property
    def is_fixed_length(self):
        return True

    def forward(self, outp, mask):
        """
        enc_output: ``(batch, seq_len, hidden_dim)``
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        B, L, H = outp.size()
        R = self.R
        keys = self.keys_proj(outp)
        values = self.values_proj(outp)
        raw_scores = (self.query_matrix @ torch.flatten(keys, end_dim=-2).T).view(B, R, L)
        if mask is not None:
            mask_reshaped = mask.view(B, 1, L)
            raw_scores = raw_scores.masked_fill(mask_reshaped, -float('inf'))
        attention_weights =  self.softmax(raw_scores / self.d_sqrt)
        output = self.norm(attention_weights @ values)
        return attention_weights, output

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.enc_rnn_size,
            opt.hidden_ab_size,
            opt.ab_fixed_length,
            opt.ab_layer_norm,
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

    def forward(self, outp, mask):
        """
        enc_output: ``(batch, seq_len, hidden_dim)``
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        outp = TransformerEncoderLayer.forward(self, outp, mask)
        return None, outp

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.enc_rnn_size,
            opt.heads,
            opt.hidden_ab_size, # d_ff
            # TODO: that list indexing things seems suspicious to me...
            opt.dropout[0],
            opt.attention_dropout[0],
            max_relative_positions=opt.max_relative_positions,
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

    def forward(self, outp, _):
        """
        enc_output: ``(batch, seq_len, hidden_dim)``
        mask: the second parameter is ignored
        """
        return None, self.module(outp)

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.enc_rnn_size,
            opt.hidden_ab_size,
            opt.ab_layer_norm,
        )


class AttentionBridge(nn.Module):
    """
    N-layered attention-bridge between encoders->decoders
    """
    def __init__(self, layers):
        """Attention Heads Layer"""
        super(AttentionBridge, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.is_fixed_length = any(l.is_fixed_length for l in layers)

    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        # convert opts specifications to architectures
        layer_type_to_cls = {
            'lin': LinAttentionBridgeLayer,
            'simple': SimpleAttentionBridgeLayer,
            'transformer': TransformerAttentionBridgeLayer,
            'feedforward': FeedForwardAttentionBridgeLayer,
        }

        # preconstruct layers using .from_opt(...)
        layers = [
            layer_type_to_cls[layer_type].from_opt(opt)
            for layer_type in opt.ab_layers
        ]

        return cls(layers)

    def forward(self, enc_output, mask):
        """Forward pass for the bridge layers"""
        out = enc_output.transpose(0, 1)
        alphas = None
        for layer in self.layers:
            alphas, out  = layer(out, mask)
            if layer.is_fixed_length:
                # In this case, we've padded all examples to a constant size,
                # so the mask is no longer required
                mask = None
        out =  torch.transpose(out, 0, 1).contiguous() # [hop, bsz, nhid]
        return out, alphas # [hop, bsz, nhid], [bsz, hop, srcseqlen]
