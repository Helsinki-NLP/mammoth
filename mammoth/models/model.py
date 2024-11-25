""" Onmt NMT Model base class definition """
import torch.nn as nn

from mammoth.utils.logging import logger


class BaseModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder / decoder or decoder only model.
    """

    def __init__(self, encoder, decoder, attention_bridge):
        super(BaseModel, self).__init__()

    def forward(self, src, tgt, lengths):
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

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        raise NotImplementedError

    def update_dropout(self, dropout):
        raise NotImplementedError

    def count_parameters(self):
        raise NotImplementedError


class NMTModel(BaseModel):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.
    Args:
      encoder (mammoth.modules.layer_stack.StackXcoder): an encoder object
      decoder (mammoth.modules.layer_stack.StackXcoder): a decoder object
    """

    def __init__(self, encoder, decoder, attention_bridge):
        super(NMTModel, self).__init__(encoder, decoder, attention_bridge)
        self.encoder = encoder
        self.decoder = decoder
        self.attention_bridge = attention_bridge

    def forward(self, src, decoder_input, src_mask, metadata=None):
        # Activate the correct pluggable embeddings and modules
        active_encoder = self.encoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.encoder_adapter_ids,
        )
        active_decoder = self.decoder.activate(
            task_id=metadata.corpus_id,
            adapter_ids=metadata.decoder_adapter_ids,
        )

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
            return_attn=False,
            return_logits_and_embeddings=True,
        )
        # not that if return_attn were to be used, the return signature would be:
        # (logits, decoder_output), attentions = retval
        logits, decoder_output = retval
        return logits, decoder_output

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)

    def count_parameters(self):
        """Count and log number of parameters in model.

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
        logger.debug('encoder: {}'.format(enc))
        logger.debug('decoder: {}'.format(dec))
        logger.debug('* number of parameters: {}'.format(enc + dec))
        return enc, dec
