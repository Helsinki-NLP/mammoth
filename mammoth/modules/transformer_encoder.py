"""
Implementation of "Attention is All You Need"
"""

import torch.nn as nn

from mammoth.modules.encoder import EncoderBase
from mammoth.modules import MultiHeadedAttention
from mammoth.modules.position_ffn import PositionwiseFeedForward, ActivationFunction
from mammoth.utils.misc import sequence_mask


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        is_normformer (bool):
            whether to apply normformer-style normalization
    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        max_relative_positions=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        is_normformer=False,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=attention_dropout, max_relative_positions=max_relative_positions
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            is_normformer=is_normformer,
        )
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        if is_normformer:
            self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm_1(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm, mask=mask, attn_type="self")
        out = self.dropout(self.layer_norm_2(context)) + inputs
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class TransformerEncoder(EncoderBase):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (mammoth.modules.Embeddings):
          embeddings to use, should have positional encodings
        pos_ffn_activation_fn (ActivationFunction):
            activation function choice for PositionwiseFeedForward layer
        is_normformer (bool):
            whether to apply normformer-style normalization

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        pos_ffn_activation_fn=ActivationFunction.relu,
        layer_norm_module=None,
        is_normformer=False,
    ):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    max_relative_positions=max_relative_positions,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    is_normformer=is_normformer,
                )
                for i in range(num_layers)
            ]
        )
        self.layer_norm = layer_norm_module

    @classmethod
    def from_opts(cls, opts, embeddings, is_on_top=False):
        """Alternate constructor."""
        return cls(
            opts.enc_layers,
            opts.model_dim,
            opts.ab_heads,
            opts.transformer_ff,
            opts.dropout[0] if isinstance(opts.dropout, list) else opts.dropout,
            opts.attention_dropout[0] if isinstance(opts.attention_dropout, list) else opts.attention_dropout,
            embeddings,
            opts.max_relative_positions,
            pos_ffn_activation_fn=opts.pos_ffn_activation_fn,
            layer_norm_module=(
                nn.LayerNorm(opts.model_dim, eps=1e-6) if is_on_top
                else nn.Identity()
            ),
            is_normformer=opts.normformer,
        )

    def forward(self, src, lengths=None, skip_embedding=False, mask=None):
        """See :func:`EncoderBase.forward()`"""

        if skip_embedding:
            out = src
            emb = None
        else:
            self._check_args(src, lengths)
            emb = self.embeddings(src)
            out = emb.transpose(0, 1).contiguous()
            if mask is None:
                mask = ~sequence_mask(lengths).unsqueeze(1)

        # Run the forward pass of every layer of the tranformer.
        out = self._forward_loop(out, mask)
        out = self.layer_norm(out)

        # caller should call transpose and contiguous if they need it
        return emb, out, lengths, mask

    def _forward_loop(self, out, mask):
        """ Run the forward pass of every layer of the transformer. """
        for layer in self.transformer:
            out = layer(out, mask)

    def update_dropout(self, dropout, attention_dropout):
        self.embeddings.update_dropout(dropout)
        for layer in self.transformer:
            layer.update_dropout(dropout, attention_dropout)
