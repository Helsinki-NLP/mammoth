"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
#from onmt.rmsnorm_torch import RMSNorm
from onmt.encoders.transformer import TransformerEncoderLayer
#from onmt.modules.position_ffn import ActivationFunction
#import math


class BaseAttentionBridgeLayer(nn.Module):
    """
    Base class for attention bridge layers
    """
    @property
    def is_fixed_length(self):
        raise NotImplementedError


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
                 ):
        """Attention Heads Layer:"""
        super(AttentionBridgeLayer, self).__init__()
        d = hidden_size
        u = hidden_ab_size
        r = attention_heads
        self.dd = u
        #TEST
        self.model_type = model_type
        if self.model_type != "text":
            d = dec_rnn_size
        self.ws1 = nn.Linear(d, u, bias=True)
        self.ws2 = nn.Linear(u, r, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.attention_hops = r
        #self.layer_norm = nn.LayerNorm(d, eps=1e-6)
        self.M = None

        #ADDONS - alessandro's (?)
        #self.transformer = nn.ModuleList(
        #    [TransformerEncoderLayer(
        #        d, 16, 4096, 0.1, 0.1,
        #        max_relative_positions=0,
        #        pos_ffn_activation_fn=ActivationFunction.relu)
        #     for i in range(num_layers)])#

        #self.layer_norm = RMSNorm(d, eps=1e-6)  #nn.LayerNorm(d, eps=1e-6) #RMSNorm(d, eps=1e-6) #nn.LayerNorm(d, eps=1e-6)
        #self.layer_norm_init = RMSNorm(d, eps=1e-6)
        #self.ws3 = nn.LayerNorm(d, eps=1e-6) #RMSNorm(d) #nn.LayerNorm(d, eps=1e-6)
        #self.ws3 = nn.Linear(d, d, bias=True)

    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(
            opt.rnn_size,
            opt.ab_fixed_length,
            opt.hidden_ab_size,
            opt.model_type,
            opt.dec_rnn_size,
        )


    def forward(self, enc_output, mask):
        """
        mask: binary mask 1/0 indicating which keys have
        zero/non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        """
        output, alphas = self.mixAtt(enc_output, mask)
        #take transpose to match dimensions s.t. r=new_seq_len:
        #output = self.layer_norm(output)
        #output = self.ws3(output)
        self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        #self.M = self.layer_norm(self.M)
        #h_avrg = (self.M).mean(dim=0, keepdim=True)
        return alphas, self.M

    @property
    def is_fixed_length(self):
        return True

    def mixAtt(self, outp, mask):
        """Notation based on Lin et al. (2017)"""
        outp = torch.transpose(outp, 0, 1).contiguous()
        #outp = self.ws3(outp)
        #outp = self.layer_norm_init(outp)
        #outp = outp / math.sqrt(self.dd)
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        hbar = self.ws1(compressed_embeddings)  # [bsz*len, attention-unit]
#        hbar = hbar / math.sqrt(self.dd) # [bsz*len, attention-unit]
        hbar = self.relu(hbar)
#        hbar = self.relu(self.ws1(compressed_embeddings))  # [bsz*len, attention-unit]

        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        # previously, we sould penalize alphas if "text" and only the 1st AB-layer
        # but we should only consider penalizing if there's something to mask
        if mask is not None:
            #transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
            #transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
            concatenated_inp = [mask for i in range(self.attention_hops)]
            concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

            penalized_alphas = alphas + (-10000 * (concatenated_inp == 1).float()) # [bsz, hop, len] + [bsz, hop, len]
            alphas = penalized_alphas

        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas


class SimpleAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """Simple attention based bridge layer using a fixed query matrix"""
    def __init__(self, input_size, hidden_size, fixed_seqlen):
        super().__init__()
        self.query_matrix = nn.Parameter(torch.zeros(fixed_seqlen, hidden_size))
        self.keys_proj = nn.Linear(input_size, hidden_size)
        self.values_proj = nn.Linear(input_size, input_size)
        self.d_sqrt = hidden_size ** 0.5
        self.softmax = nn.Softmax(dim=-1)

    @property
    def is_fixed_length(self):
        return True

    def forward(self, outp, mask):
        B, L, H = outp.size()
        R = self.query_matrix.weight.size(0)
        keys = self.keys_proj(outp)
        values = self.values_proj(outp)
        raw_scores = (self.query_matrix @ torch.flatten(keys, end_dim=-2).T).view(B, R, L)
        if mask is not None:
            raw_scores = raw_scores.masked_fill(mask, -float('inf'))
        attention_weights =  self.softmax(raw_scores / self.d_sqrt)
        return attention_weights,attention_weights @ values

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.enc_rnn_size,
            opt.hidden_ab_size,
            opt.ab_fixed_length,
        )


class TransformerAttentionBridgeLayer(BaseAttentionBridgeLayer, TransformerEncoderLayer):
    """Using a Transformer encoder layer as a shared component in the attention bridge"""
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)

    @property
    def is_fixed_length(self):
        return False

    def forward(self, outp, mask):
        outp = outp.transpose(0,1).contiguous()
        outp = super(TransformerEncoderLayer, self).forward(outp, mask)
        return None, outp.transpose(0, 1).contiguous()

    @classmethod
    def from_opt(cls, opt):
        return cls(
            d_model=opt.enc_rnn_size,
            heads=opt.heads,
            d_ff=opt.hidden_ab_size,
            dropout=opt.dropout,
            attention_dropout=opt.attention_dropout[0],
            max_relative_positions=opt.smax_relative_positions
        )


class FeedForwardAttentionBridgeLayer(BaseAttentionBridgeLayer):
    """Simple feedforward bridge component"""
    def __init__(self, input_size, hidden_size):
        self.module = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    @property
    def is_fixed_length(self):
        return False

    def forward(self, outp, mask):
        return self.module(outp)

    @classmethod
    def from_opt(cls, opt):
        return cls(
            opt.enc_rnn_size,
            opt.hidden_ab_size,
        )


class AttentionBridge(nn.Module):
    """
    N-layered attention-bridge between encoders->decoders
    """
    def __init__(self, layers):
        """Attention Heads Layer"""
        super(AttentionBridge, self).__init__()
        self.layers = layers
        self.is_fixed_length = any(l.is_fixed_length for l in layers)
    #
    #
    # self.ab_nlayers = n_layers_attbrg
    # self.ab_layertype = layer_type_attbrg
    # self.word_padding_idx = word_padding_idx
    # if self.ab_layertype == 'fixed-size':
    #     self.attbrg = nn.ModuleList([
    #         AttentionBridgeLayer.from_opt(model_opt) for i in range(self.ab_nlayers)
    #         ])
    # elif self.ab_layertype == 'transformer':
    #     self.attbrg = nn.ModuleList(
    #         [TransformerEncoderLayer(
    #             d_model=enc_rnn_size,
    #             heads=heads,
    #             d_ff=transformer_ff,
    #             dropout=dropout,
    #             attention_dropout=model_opt.attention_dropout[0],
    #             max_relative_positions=max_relative_positions)
    #         for i in range(self.ab_nlayers-1)])
    #     self.attbrg.append(AttentionBridgeLayer.from_opt(model_opt))

    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        if opt.ab_layers:
            layer_type_to_cls = {
                'lin': LinAttentionBridgeLayer,
                'simple': SimpleAttentionBridgeLayer,
                'transformer': TransformerAttentionBridgeLayer,
                'feedforward': FeedForwardAttentionBridgeLayer,
            }
            layers = nn.ModuleList([
                layer_type_to_cls[layer_type].from_opt(opt)
                for layer_type in opt.ab_layers
            ])
            return cls(layers)
            # return cls(
            #     opt.n_layers_attbrg,
            #     opt.layer_type_attbrg,
            #     opt.word_padding_idx,
            #     opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            #     opt.enc_rnn_size,
            #     opt.heads,
            #     opt.transformer_ff,
            #     opt.max_relative_positions,
            #     opt,
            #     )
        else:
            return nn.Identity()

    def forward(self, enc_output, mask):
        """Forward pass for the bridge layers"""
        out = enc_output
        for layer in self.layers:
            alphas, out  = layer(out, mask)
            if layer.is_fixed_length:
                # In this case, we've padded all examples to a constant size,
                # so the mask is no longer required
                mask = None
        return out, alphas # [hop, bsz, nhid], [bsz, hop, srcseqlen]
