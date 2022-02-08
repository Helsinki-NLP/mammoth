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

class AttentionBridge(nn.Module):
    """
    Multi-headed attention. Bridge between encoders->decoders
    """
    def __init__(self, hidden_size, attention_heads, model_opt):
        """Attention Heads Layer:"""
        super(AttentionBridge, self).__init__()
        d = hidden_size
        u = model_opt.hidden_ab_size
        r = attention_heads
        self.dd = u
        #TEST
        self.model_type = model_opt.model_type
        if self.model_type != "text":
            d = model_opt.dec_rnn_size
        self.ws1 = nn.Linear(d, u, bias=True)
        self.ws2 = nn.Linear(u, r, bias=True)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        #self.softmax = nn.ReLU() # nn.Softmax(dim=1)
        #num_layers=1
        self.attention_hops = r
        #ADDONS
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

        self.M = None


    def forward(self, enc_output, mask):
##           mask: binary mask 1/0 indicating which keys have
##           zero / non-zero attention ``(batch, query_len, key_len)`` -> # [bsz, 1, len]
        #enc_output = torch.transpose(enc_output, 0, 1).contiguous()
        #for layer in self.transformer:
        #    enc_output = layer(enc_output, mask)
        output, alphas = self.mixAtt(enc_output, mask)
        #take transpose to match dimensions s.t. r=new_seq_len:
        #output = self.layer_norm(output)
        #output = self.ws3(output)
        self.M = torch.transpose(output, 0, 1).contiguous() #[r,bsz,nhid]
        #self.M = self.layer_norm(self.M)
        #h_avrg = (self.M).mean(dim=0, keepdim=True)
        return alphas, self.M


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
