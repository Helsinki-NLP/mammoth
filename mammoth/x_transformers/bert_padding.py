# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

# Adapted from https://github.com/HazyResearch/flash-attention/blob/main/flash_attn/bert_padding.py
# Which was adapted from https://github.com/mlcommons/training_results_v1.1/blob/main/NVIDIA/benchmarks/bert/implementations/pytorch/padding.py

"""Helper functions for padding and unpadding batches.

These functions are used to improve performance by removing padding tokens
from input sequences before processing, then adding them back afterwards.
This is particularly useful with variable-length sequences where padding
can significantly slow down computation.
"""

from typing import Tuple, cast

import torch
import torch.nn.functional as F
from einops import rearrange, repeat


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Get just the values of `input` which are at `indices`.

        Arguments:
            ctx: the autograd context object
            input: (b, ...) 2+ dimensional tensor
            indices: (num_idx) 1D tensor
        """
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]  # type: ignore
        second_dim = other_shape.numel()  # product of sizes of all but first dimension (total number of elements in the other dimensions)
        # TD [2022-03-04] For some reason torch.gather is a bit faster than indexing.
        return torch.gather(
            rearrange(input, "b ... -> b (...)"),  # (b, ...) -> (b, second_dim)
            0,
            repeat(indices, "z -> z d", d=second_dim),  # (indices,) -> (indices, second_dim)
        ).reshape(-1, *other_shape)  # (num_idx, ...)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]], device=grad_output.device, dtype=grad_output.dtype
        )
        # TD [2022-03-04] For some reason torch.scatter is a bit faster than indexing.
        # grad_input[indices] = grad_output
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values: torch.Tensor, indices: torch.Tensor, first_axis_dim) -> torch.Tensor:
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove padding from input sequences.

    This function takes a batch of padded sequences and removes all padding tokens,
    creating a single contiguous tensor of valid tokens. It also returns metadata
    needed to reconstruct the original batch structure.

    Arguments:
        hidden_states: (batch, seqlen, ...) - Input tensor with padding
        attention_mask: (batch, seqlen) - Binary mask where 1=valid token, 0=padding

    Returns:
        hidden_states: (total_nnz, ...) - Unpadded tokens, where total_nnz = sum of valid tokens
        indices: (total_nnz,) - Original positions of valid tokens in flattened input
        cu_seqlens: (batch + 1,) - Cumulative sequence lengths for each sequence in batch
        max_seqlen_in_batch: int - Maximum sequence length (excluding padding) in the batch

    Example:
        If batch has sequences of actual lengths [5, 3, 4] padded to length 6:
        - hidden_states shape: (3, 6, d) -> (12, d)  # 5+3+4 = 12 valid tokens
        - cu_seqlens: [0, 5, 8, 12]  # cumulative indices
        - max_seqlen_in_batch: 5
    """
    # get the lengths of each sequence in the batch (e,g., [5, 3, 4] for a batch of 3 sequences)
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # get the indices of the valid tokens from the entire batch
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 'item()' converts the tensor to a scalar
    max_seqlen_in_batch = int(seqlens_in_batch.max().item())
    # 'F.pad' pads the tensor with zeros to the left and right
    # 'torch.cumsum' computes the cumulative sum of the tensor along the specified dimension
    # so it gives us the starting index of each sequence in the batch
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    # TD [2022-03-04] We don't want to index with a bool mask, because Pytorch will expand the
    # bool mask, then call nonzero to get the indices, then index with those. The indices is @dim
    # times larger than it needs to be, wasting memory. It's faster and more memory-efficient to
    # index with integer indices. Moreover, torch's index is a bit slower than it needs to be,
    # so we write custom forward and backward to make it a bit faster.

    # Handle both 2D (token indices) and 3D+ (embeddings) inputs, matching ModernBERT's approach
    if hidden_states.dim() == 2:
        # For 2D input (batch, seqlen) - token indices before embedding
        hidden_states = hidden_states.flatten()[indices]
    else:
        # For 3D+ input (batch, seqlen, ...) - embeddings or higher-dimensional tensors
        hidden_states = cast(torch.Tensor, index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices))

    return hidden_states, indices, cu_seqlens, max_seqlen_in_batch


def unpad_input_only(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Like unpad_input, but only return the unpadded first tensor.

    This is a lightweight version that saves overhead when you only need
    the unpadded hidden states and not the metadata.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Returns:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
    """
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    rearranged = rearrange(hidden_states, "b s ... -> (b s) ...")
    return index_first_axis(rearranged, indices)  # type: ignore


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    """Add padding back to unpadded sequences.

    This reconstructs the original padded tensor shape from the unpadded
    representation returned by unpad_input().

    Arguments:
        hidden_states: (total_nnz, ...) - Unpadded tokens from unpad_input()
        indices: (total_nnz,) - Token position indices from unpad_input()
        batch: int - Original batch size
        seqlen: int - Maximum sequence length (with padding)

    Returns:
        hidden_states: (batch, seqlen, ...) - Reconstructed padded tensor

    Example:
        If we unpacked sequences of lengths [5, 3, 4] from shape (3, 6, d):
        pad_input(unpadded_tokens, indices, batch=3, seqlen=6) -> (3, 6, d)
        where positions corresponding to padding contain zeros
    """
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)  # type: ignore


def pad_output(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    batch: int,
    seqlen: int
) -> torch.Tensor:
    """Add padding back to unpadded sequences (ModernBERT-style).

    This function matches ModernBERT's _pad_modernbert_output() behavior.
    It reconstructs the original padded tensor shape from the unpadded
    representation returned by unpad_input().

    Arguments:
        hidden_states: (total_nnz, ...) - Unpadded tokens from unpad_input()
        indices: (total_nnz,) - Token position indices from unpad_input()
        batch: int - Original batch size
        seqlen: int - Maximum sequence length (with padding)

    Returns:
        hidden_states: (batch, seqlen, ...) - Reconstructed padded tensor

    Example:
        If we unpadded sequences of lengths [5, 3, 4] from shape (3, 6, d):
        pad_output(unpadded_tokens, indices, batch=3, seqlen=6) -> (3, 6, d)
        where positions corresponding to padding contain zeros
    """
    if hidden_states.dim() == 1:
        # For 1D tensors
        output = torch.zeros(batch * seqlen, dtype=hidden_states.dtype, device=hidden_states.device)
        output[indices] = hidden_states
        return output.view(batch, seqlen)
    else:
        # For 2D+ tensors
        _, *rest = hidden_states.shape
        output = torch.zeros(batch * seqlen, *rest, dtype=hidden_states.dtype, device=hidden_states.device)
        output[indices] = hidden_states
        return output.view(batch, seqlen, *rest)
