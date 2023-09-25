import torch.nn as nn


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opts(cls, opts, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError
