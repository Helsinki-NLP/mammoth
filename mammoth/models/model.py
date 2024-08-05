""" Onmt NMT Model base class definition """
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder, attention_bridge):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, lengths, return_attention=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            return_attention (Boolean): A flag indicating whether output attention,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def count_parameters(self, log=print):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (mammoth.encoders.EncoderBase): an encoder object
      decoder (mammoth.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder, attention_bridge):
        super(NMTModel, self).__init__(encoder, decoder, attention_bridge)
        self.encoder = encoder
        self.decoder = decoder
        self.attention_bridge = attention_bridge

    def forward(self, src, decoder_input, src_mask, return_attention=False, metadata=None):
        # Activate the correct pluggable embeddings and modules
        active_encoder = self.encoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.encoder_adapter_ids,
        )
        active_decoder = self.decoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.decoder_adapter_ids,
        )

        # # QUI logging for batch shapes
        # def quishape(name, val):
        #     print(f'{name} {val.shape}  {val.shape[0] * val.shape[1]}')
        # quishape('src', src)
        # quishape('src_mask', src_mask)

        encoder_output = active_encoder(
            x=src,
            mask=src_mask,
            return_embeddings=True,
        )

        encoder_output, alphas = self.attention_bridge(encoder_output, src_mask)
        if self.attention_bridge.is_fixed_length:
            # turn off masking in the transformer decoder
            src_mask = None

        retval = active_decoder(
            decoder_input,
            context=encoder_output,
            context_mask=src_mask,
            return_attn=return_attention,
            return_logits_and_embeddings=True,
        )
        if return_attention:
            (logits, decoder_output), attentions = retval
        else:
            logits, decoder_output = retval
            attentions = None
        return logits, decoder_output, attentions

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self, log=print):
        """Count number of parameters in model (& print with `log` callback).

        Returns:
            (int, int):
            * encoder side parameter count
            * decoder side parameter count
        """

        enc, dec = 0, 0

        for name, param in self.named_parameters():
            if 'encoder' in name:
                enc += param.nelement()
            else:
                dec += param.nelement()
        if callable(log):
            log('encoder: {}'.format(enc))
            log('decoder: {}'.format(dec))
            log('* number of parameters: {}'.format(enc + dec))
        return enc, dec
