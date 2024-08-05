"""
Just a thin helper.
Use standard torch.nn.*Loss classes, or custom losses that quack like them.
"""
import torch.nn as nn

from mammoth.constants import DefaultTokens


def build_loss_function(tgt_vocab, label_smoothing=0.0):
    """Helper for building torch.nn.CrossEntropyLoss"""
    padding_idx = tgt_vocab.stoi[DefaultTokens.PAD]
    loss_function = nn.CrossEntropyLoss(
        ignore_index=padding_idx,
        reduction='sum',
        label_smoothing=label_smoothing,
    )
    return loss_function
