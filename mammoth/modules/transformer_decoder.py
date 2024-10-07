"""
Implementation of "Attention is All You Need" and of
subsequent transformer based architectures
"""

import torch
import torch.nn as nn

from mammoth.modules.decoder import DecoderBase
from mammoth.modules import MultiHeadedAttention, AverageAttention
from mammoth.modules.position_ffn import PositionwiseFeedForward, ActivationFunction
from mammoth.utils.misc import sequence_mask


# TODO: this abstract base class that's not even declared as one should be merged with
# its sole child class
class TransformerDecoderLayerBase(nn.Module):
    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        is_normformer=False,
    ):
        """
        Args:
            d_model (int): the dimension of keys/values/queries in
                :class:`MultiHeadedAttention`, also the input size of
                the first-layer of the :class:`PositionwiseFeedForward`.
            heads (int): the number of heads for MultiHeadedAttention.
            d_ff (int): the second-layer of the
                :class:`PositionwiseFeedForward`.
            dropout (float): dropout in residual, self-attn(dot) and
                feed-forward
            attention_dropout (float): dropout in context_attn  (and
                self-attn(avg))
            self_attn_type (string): type of self-attention scaled-dot,
                average
            max_relative_positions (int):
                Max distance between inputs in relative positions
                representations
            aan_useffn (bool): Turn on the FFN layer in the AAN decoder
            full_context_alignment (bool):
                whether enable an extra full context decoder forward for
                alignment
            alignment_heads (int):
                N. of cross attention heads to use for alignment guiding
            pos_ffn_activation_fn (ActivationFunction):
                activation function choice for PositionwiseFeedForward layer
            is_normformer (bool):
                whether to apply normformer-style normalization

        """
        super(TransformerDecoderLayerBase, self).__init__()
        assert not full_context_alignment, 'alignment is obsolete'
        assert alignment_heads == 0, 'alignment is obsolete'

        if self_attn_type == "scaled-dot":
            self.self_attn = MultiHeadedAttention(
                heads,
                d_model,
                dropout=attention_dropout,
                max_relative_positions=max_relative_positions,
            )
        elif self_attn_type == "average":
            assert not is_normformer, 'normformer only supported with scaled-dot attn'
            self.self_attn = AverageAttention(d_model, dropout=attention_dropout, aan_useffn=aan_useffn)

        self.feed_forward = PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout,
            pos_ffn_activation_fn,
            is_normformer=is_normformer,
        )
        self.layer_norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm_3 = nn.Identity()  # normformer's
        self.layer_norm_4 = nn.Identity()  # normformer's
        if is_normformer:
            self.layer_norm_3 = nn.LayerNorm(d_model, eps=1e-6)
            self.layer_norm_4 = nn.LayerNorm(d_model, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, *args, **kwargs):
        """Extend `_forward` for (possibly) multiple decoder pass:
        Always a default (future masked) decoder forward pass,
        Possibly a second future aware decoder pass for joint learn
        full context alignement, :cite:`garg2019jointly`.

        Args:
            * All arguments of _forward.

        Returns:
            (FloatTensor, FloatTensor, FloatTensor or None):

            * output ``(batch_size, T, model_dim)``
            * top_attn ``(batch_size, T, src_len)``
            * attn_align ``(batch_size, T, src_len)`` or None
        """
        output, attns = self._forward(*args, **kwargs)
        top_attn = attns[:, 0, :, :].contiguous()
        attn_align = None
        return output, top_attn, attn_align

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.drop.p = dropout

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_dec_mask(self, tgt_pad_mask, future):
        tgt_len = tgt_pad_mask.size(-1)
        if not future:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len],
                device=tgt_pad_mask.device,
                dtype=torch.uint8,
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:  # only mask padding, result mask in (B, 1, T)
            dec_mask = tgt_pad_mask
        return dec_mask

    def _forward_self_attn(self, inputs_norm, dec_mask, layer_cache, step):
        if isinstance(self.self_attn, MultiHeadedAttention):
            return self.self_attn(
                inputs_norm,
                inputs_norm,
                inputs_norm,
                mask=dec_mask,
                layer_cache=layer_cache,
                attn_type="self",
            )
        elif isinstance(self.self_attn, AverageAttention):
            return self.self_attn(inputs_norm, mask=dec_mask, layer_cache=layer_cache, step=step)
        else:
            raise ValueError(f"self attention {type(self.self_attn)} not supported")


class TransformerDecoderLayer(TransformerDecoderLayerBase):
    """Transformer Decoder layer block in Pre-Norm style.
    Pre-Norm style is an improvement w.r.t. Original paper's Post-Norm style,
    providing better converge speed and performance. This is also the actual
    implementation in tensor2tensor and also avalable in fairseq.
    See https://tunz.kr/post/4 and :cite:`DeeperTransformer`.

    .. mermaid::

        graph LR
        %% "*SubLayer" can be self-attn, src-attn or feed forward block
            A(input) --> B[Norm]
            B --> C["*SubLayer"]
            C --> D[Drop]
            D --> E((+))
            A --> E
            E --> F(out)

    """

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout,
        attention_dropout,
        self_attn_type="scaled-dot",
        max_relative_positions=0,
        aan_useffn=False,
        full_context_alignment=False,
        alignment_heads=0,
        pos_ffn_activation_fn=ActivationFunction.relu,
        is_normformer=False,
    ):
        """
        Args:
            See TransformerDecoderLayerBase
        """
        super(TransformerDecoderLayer, self).__init__(
            d_model,
            heads,
            d_ff,
            dropout,
            attention_dropout,
            self_attn_type,
            max_relative_positions,
            aan_useffn,
            full_context_alignment,
            alignment_heads,
            pos_ffn_activation_fn=pos_ffn_activation_fn,
            is_normformer=is_normformer,
        )
        self.context_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model, eps=1e-6)

    def update_dropout(self, dropout, attention_dropout):
        super(TransformerDecoderLayer, self).update_dropout(dropout, attention_dropout)
        self.context_attn.update_dropout(attention_dropout)

    def _forward(
        self,
        inputs,
        memory_bank,
        src_pad_mask,
        tgt_pad_mask,
        layer_cache=None,
        step=None,
        future=False,
    ):
        """A naive forward pass for transformer decoder.

        # T: could be 1 in the case of stepwise decoding or tgt_len

        Args:
            inputs (FloatTensor): ``(batch_size, T, model_dim)``
            memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
            src_pad_mask (bool): ``(batch_size, 1, src_len)``
            tgt_pad_mask (bool): ``(batch_size, 1, T)``
            layer_cache (dict or None): cached layer info when stepwise decode
            step (int or None): stepwise decoding counter
            future (bool): If set True, do not apply future_mask.

        Returns:
            (FloatTensor, FloatTensor):

            * output ``(batch_size, T, model_dim)``
            * attns ``(batch_size, head, T, src_len)``

        """
        dec_mask = None

        if inputs.size(1) > 1:
            # masking is necessary when sequence length is greater than one
            dec_mask = self._compute_dec_mask(tgt_pad_mask, future)

        inputs_norm = self.layer_norm_1(inputs)

        query, _ = self._forward_self_attn(inputs_norm, dec_mask, layer_cache, step)
        query = self.layer_norm_3(query)

        query = self.drop(query) + inputs

        query_norm = self.layer_norm_2(query)
        mid, attns = self.context_attn(
            memory_bank,
            memory_bank,
            query_norm,
            mask=src_pad_mask,
            layer_cache=layer_cache,
            attn_type="context",
        )
        mid = self.layer_norm_4(mid)
        output = self.feed_forward(self.drop(mid) + query)

        return output, attns


class TransformerDecoderBase(DecoderBase):
    def __init__(self, d_model, copy_attn, embeddings, alignment_layer, layer_norm_module, **_):
        super(TransformerDecoderBase, self).__init__()

        self.embeddings = embeddings

        # Decoder State
        self.state = {}

        # previously, there was a GlobalAttention module here for copy
        # attention. But it was never actually used -- the "copy" attention
        # just reuses the context attention.
        self._copy = copy_attn
        self.layer_norm = layer_norm_module

        self.alignment_layer = alignment_layer

    @classmethod
    def from_opts(cls, opts, embeddings, is_on_top=False):
        """Alternate constructor."""
        return cls(
            opts.dec_layers,
            opts.model_dim,
            opts.heads,
            opts.transformer_ff,
            opts.copy_attn,
            opts.self_attn_type,
            opts.dropout[0] if isinstance(opts.dropout, list) else opts.dropout,
            opts.attention_dropout[0] if isinstance(opts.attention_dropout, list) else opts.attention_dropout,
            embeddings,
            opts.max_relative_positions,
            opts.aan_useffn,
            False,
            None,
            alignment_heads=0.,
            pos_ffn_activation_fn=opts.pos_ffn_activation_fn,
            layer_norm_module=(
                nn.LayerNorm(opts.model_dim, eps=1e-6) if is_on_top
                else nn.Identity()
            ),
            is_normformer=opts.normformer,
        )

    def init_state(self, src, memory_bank, enc_hidden):
        """Initialize decoder state."""
        self.state["src"] = src
        self.state["cache"] = None

    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)

        if self.state["src"] is not None:
            self.state["src"] = fn(self.state["src"], 1)
        if self.state["cache"] is not None:
            _recursive_map(self.state["cache"])

    def detach_state(self):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def update_dropout(self, dropout, attention_dropout):
        if self.embeddings:
            self.embeddings.update_dropout(dropout)
        for layer in self.transformer_layers:
            layer.update_dropout(dropout, attention_dropout)


class TransformerDecoder(TransformerDecoderBase):
    """The Transformer decoder from "Attention is All You Need".
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          BB[multi-head src-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> BB
          BB --> C
          C --> O


    Args:
        num_layers (int): number of decoder layers.
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        copy_attn (bool): if using a separate copy attention
        self_attn_type (str): type of self-attention scaled-dot, average
        dropout (float): dropout in residual, self-attn(dot) and feed-forward
        attention_dropout (float): dropout in context_attn (and self-attn(avg))
        embeddings (mammoth.modules.Embeddings):
            embeddings to use, should have positional encodings
        max_relative_positions (int):
            Max distance between inputs in relative positions representations
        aan_useffn (bool): Turn on the FFN layer in the AAN decoder
        full_context_alignment (bool):
            whether enable an extra full context decoder forward for alignment
        alignment_layer (int): N° Layer to supervise with for alignment guiding
        alignment_heads (int):
            N. of cross attention heads to use for alignment guiding
        is_normformer (bool):
            whether to apply normformer-style normalization
    """

    def __init__(
        self,
        num_layers,
        d_model,
        heads,
        d_ff,
        copy_attn,
        self_attn_type,
        dropout,
        attention_dropout,
        embeddings,
        max_relative_positions,
        aan_useffn,
        full_context_alignment,
        alignment_layer,
        alignment_heads,
        pos_ffn_activation_fn=ActivationFunction.relu,
        layer_norm_module=None,
        is_normformer=False,
    ):
        super(TransformerDecoder, self).__init__(d_model, copy_attn, embeddings, alignment_layer, layer_norm_module)

        self.transformer_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model,
                    heads,
                    d_ff,
                    dropout,
                    attention_dropout,
                    self_attn_type=self_attn_type,
                    max_relative_positions=max_relative_positions,
                    aan_useffn=aan_useffn,
                    full_context_alignment=full_context_alignment,
                    alignment_heads=alignment_heads,
                    pos_ffn_activation_fn=pos_ffn_activation_fn,
                    is_normformer=is_normformer,
                )
                for i in range(num_layers)
            ]
        )

    def detach_state(self):
        self.state["src"] = self.state["src"].detach()

    def _get_layers(self):
        """ Allow subclasses to modify layer stack on-the-fly """
        return self.transformer_layers

    def forward(
        self,
        tgt,
        memory_bank=None,
        step=None,
        memory_lengths=None,
        tgt_pad_mask=None,
        skip_embedding=False,
        **kwargs
    ):
        """Decode, possibly stepwise."""
        if memory_bank is None:
            memory_bank = self.embeddings(tgt)
            src_memory_bank = memory_bank.transpose(0, 1).contiguous()
        if step == 0:
            self._init_cache(memory_bank)

        if skip_embedding:
            # tgt and memory_bank are already in batch-first order
            output = tgt
            src_memory_bank = memory_bank
        else:
            tgt_words = tgt[:, :, 0].transpose(0, 1)

            pad_idx = self.embeddings.word_padding_idx
            tgt_pad_mask = tgt_words.data.eq(pad_idx).unsqueeze(1)  # [B, 1, T_tgt]

            emb = self.embeddings(tgt, step=step)
            assert emb.dim() == 3  # len x batch x embedding_dim

            output = emb.transpose(0, 1).contiguous()

        src_pad_mask = None
        if memory_lengths is not None:
            # either if the attention bridge contains no fixed-length component
            # or lengths were provided for a DecodeStrategy in translation
            src_max_len = memory_bank.size(1)
            src_pad_mask = ~sequence_mask(memory_lengths, src_max_len).unsqueeze(1)

        attn_aligns = []

        for i, layer in enumerate(self._get_layers()):
            layer_cache = (
                self.state["cache"]["layer_{}".format(i)]
                if step is not None
                else None
            )
            output, attn, attn_align = layer(
                output,
                src_memory_bank,
                src_pad_mask,
                tgt_pad_mask,
                layer_cache=layer_cache,
                step=step,
            )
            if attn_align is not None:
                attn_aligns.append(attn_align)

        output = self.layer_norm(output)
        # caller should call transpose and contiguous if they need it
        dec_outs = output

        attns = {"std": attn}
        if self._copy:
            attns["copy"] = attn

        # TODO change the way attns is returned dict => list or tuple (onnx)
        return dec_outs, attns

    def _init_cache(self, memory_bank):
        self.state["cache"] = {}
        # memory_bank is now batch-first
        batch_size = memory_bank.size(0)
        depth = memory_bank.size(-1)

        for i, layer in enumerate(self._get_layers()):
            try:
                if layer._does_not_need_cache:
                    self.state["cache"]["layer_{}".format(i)] = None
                    continue
            except AttributeError:
                # needs the cache
                pass
            layer_cache = {"memory_keys": None, "memory_values": None}
            if isinstance(layer.self_attn, AverageAttention):
                layer_cache["prev_g"] = torch.zeros((batch_size, 1, depth), device=memory_bank.device)
            else:
                layer_cache["self_keys"] = None
                layer_cache["self_values"] = None
            self.state["cache"]["layer_{}".format(i)] = layer_cache
