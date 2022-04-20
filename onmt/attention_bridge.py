"""Multi-headed attention"""

from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt
from onmt.rmsnorm_torch import RMSNorm
#from onmt.modules.position_ffn import ActivationFunction
#from onmt.encoders.transformer import TransformerEncoderLayer
import math


from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt

class AttentionBridgeLayer(nn.Module):
    """
    Multi-headed attention. Bridge between encoders->decoders
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
            opt.hidden_size,
            opt.attention_heads,
            opt.hidden_ab_size,
            opt.model_type,
            opt.dec_rnn_size,
        )

    def v1forward(self, enc_in_out: tuple):
        enc_input, enc_output = enc_in_out
        self.M, alphas = self.mixAtt(enc_output, enc_input)

        #output, alphas = self.mixAtt(enc_output, enc_input)
        #take transpose to match dimensions s.t. r=new_seq_len:
        #self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        #self.M = self.layer_norm(self.M)
        #h_avrg = (self.M).mean(dim=0, keepdim=True)
        return alphas, self.M

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


    def v1mixAtt(self, outp, inp):
        """Notation based on Lin et al. (2017) A structured self-attentive sentence embedding"""
        #outp = torch.transpose(outp, 0, 1).contiguous() # <- passed to the AttentionBridge
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]

        hbar = self.relu(self.ws1(compressed_embeddings))  # [bsz*len, attention-unit]

        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]

        #Penalize alphas if "text"
        if self.model_type == "text":
            transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
            transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
            concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
            concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

            penalized_alphas = alphas + (-10000 * (concatenated_inp == 1).float()) # [bsz, hop, len] + [bsz, hop, len]
            alphas = penalized_alphas

        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def mixAtt(self, outp, mask):
        """Notation based on Lin et al. (2017) A structured self-attentive sentence embedding"""
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

        #Penalize alphas if "text"
        if self.model_type == "text":
            #transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
            #transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
            concatenated_inp = [mask for i in range(self.attention_hops)]
            concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

            penalized_alphas = alphas + (-10000 * (concatenated_inp == 1).float()) # [bsz, hop, len] + [bsz, hop, len]
            alphas = penalized_alphas

        alphas = self.softmax(alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas




class AttentionBridge(nn.Module):
    """
    N-layered attention-bridge between encoders->decoders
    """
    def __init__(self, 
            n_layers_attbrg,
            layer_type_attbrg,
            word_padding_idx,
            dropout,
            enc_rnn_size, 
            heads, 
            transformer_ff,
            max_relative_positions,
            model_opt,
            ):
        """Attention Heads Layer:"""
        super(AttentionBridge, self).__init__()
        self.ab_nlayers = n_layers_attbrg 
        self.ab_layertype = layer_type_attbrg 
        self.word_padding_idx = word_padding_idx
        if self.ab_layertype == 'fixed-size':
            self.attbrg = nn.ModuleList([
                AttentionBridgeLayer.from_opt(model_opt) for i in range(self.ab_nlayers)
                ])
        elif self.ab_layertype == 'transformer':
            from  onmt.encoders.transformer import TransformerEncoderLayer
            self.attbrg = nn.ModuleList(
                [TransformerEncoderLayer(
                    d_model=enc_rnn_size, 
                    heads=heads, 
                    d_ff=transformer_ff, 
                    dropout=dropout,
                    max_relative_positions=max_relative_positions)
                for i in range(self.ab_nlayers-1)])
            self.attbrg.append(AttentionBridgeLayer.from_opt(model_opt))
    
    @classmethod
    def from_opt(cls, opt):
        """Alternate constructor."""
        return cls(
            opt.n_layers_attbrg
            opt.layer_type_attbrg,
            opt.word_padding_idx,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            opt.enc_rnn_size, 
            opt.heads, 
            opt.transformer_ff,
            opt.max_relative_positions,
            model_opt,
            )

    def forward(self, enc_in_out):
        """Forward pass for the bridge layers"""
        
        src, out = enc_in_out
        out = torch.transpose(out, 0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        mask = words.data.eq(self.word_padding_idx).unsqueeze(1)  # [B, 1, T]
        for layer in self.attbrg:
            if isinstance(layer, AttentionBridgeLayer):
                alphas, out = layer((src,out)) 
            else:
                out = layer(out, mask)
        
        return alphas, out.transpose(0, 1).contiguous()