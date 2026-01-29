"""
Indexed corpus reader for pre-tokenized data.

This module provides a dataset class for reading pre-tokenized data from
indexed binary files during training.
"""

import itertools
import torch
from torch.utils.data import IterableDataset

from mammoth.constants import DefaultTokens
from mammoth.inputters.indexed_dataset import IndexedDataset
from mammoth.utils.logging import logger


class IndexedCorpus(IterableDataset):
    """
    Dataset for reading pre-tokenized data from indexed binary files.

    This is similar to ParallelCorpus but reads from pre-tokenized binary files
    instead of text files, eliminating tokenization overhead during training.
    """

    def __init__(
        self,
        src_path,
        tgt_path,
        src_vocab,
        tgt_vocab,
        transforms,
        device='cpu',
        stride=None,
        offset=None,
        is_train=False,
        task=None,
        max_length=None,
        line_idx_restore=None,
        model_max_seq_len=None,
        verbose_dataloader=False,
        device_rank=0,
    ):
        """
        Initialize indexed corpus.

        Args:
            src_path: Path prefix for source indexed dataset (without .bin/.idx)
            tgt_path: Path prefix for target indexed dataset (without .bin/.idx)
            src_vocab: Source vocabulary
            tgt_vocab: Target vocabulary
            transforms: Transform pipeline
            device: Device for tensors
            stride: Stride for distributed training
            offset: Offset for distributed training
            is_train: Whether this is training data
            task: Task configuration
            max_length: Maximum sequence length for padding
            line_idx_restore: Line index to restore from checkpoint
            model_max_seq_len: Model's maximum sequence length
            verbose_dataloader: Whether to log verbose dataloader info
            device_rank: Rank of the current device
        """
        self.src_path = src_path
        self.tgt_path = tgt_path
        self.transforms = transforms
        self.device = device
        self.vocabs = {
            'src': src_vocab,
            'tgt': tgt_vocab,
        }
        self.stride = stride
        self.offset = offset
        self.is_train = is_train
        self.corpus_id = task.corpus_id
        self.max_length = max_length
        self.model_max_seq_len = model_max_seq_len
        self._line_idx_restore = line_idx_restore
        self.verbose_dataloader = verbose_dataloader
        self.device_rank = device_rank

        # Load indexed datasets
        logger.info(f"Loading indexed dataset: {src_path}")
        self.src_dataset = IndexedDataset(src_path)

        if tgt_path is not None:
            logger.info(f"Loading indexed dataset: {tgt_path}")
            self.tgt_dataset = IndexedDataset(tgt_path)
        else:
            self.tgt_dataset = None

        # Validate dataset sizes match
        if self.tgt_dataset is not None:
            if len(self.src_dataset) != len(self.tgt_dataset):
                raise ValueError(
                    f"Source and target datasets have different sizes: "
                    f"{len(self.src_dataset)} vs {len(self.tgt_dataset)}"
                )

        self.num_examples = len(self.src_dataset)
        logger.info(f"Loaded {self.num_examples} examples from indexed dataset")

    def _maybe_add_special_tokens(self, token_ids, side):
        """
        Add BOS/EOS tokens to token IDs if needed.

        Args:
            token_ids: Tensor of token IDs
            side: 'src' or 'tgt'

        Returns:
            Tensor with special tokens added
        """
        vocab = self.vocabs[side]
        bos = vocab.specials.get(DefaultTokens.BOS)
        eos = vocab.specials.get(DefaultTokens.EOS)

        # Convert to tensor if needed
        if not isinstance(token_ids, torch.Tensor):
            token_ids = torch.tensor(token_ids, dtype=torch.long)

        # Check if special tokens are already present
        has_bos = len(token_ids) > 0 and token_ids[0] == bos
        has_eos = len(token_ids) > 0 and token_ids[-1] == eos

        # Add special tokens if not present
        tokens_to_add = []
        if not has_bos and bos is not None:
            tokens_to_add.append(bos)

        if len(tokens_to_add) > 0:
            # Add BOS at beginning
            token_ids = torch.cat([
                torch.tensor(tokens_to_add, dtype=torch.long),
                token_ids
            ])

        if not has_eos and eos is not None:
            # Add EOS at end
            token_ids = torch.cat([
                token_ids,
                torch.tensor([eos], dtype=torch.long)
            ])

        return token_ids

    def _make_example_dict(self, idx):
        """
        Create example dictionary from indexed dataset.

        Args:
            idx: Example index

        Returns:
            Dictionary with 'src', 'tgt', and 'line_idx' keys
        """
        # Read pre-tokenized token IDs
        src_ids = self.src_dataset[idx]
        tgt_ids = self.tgt_dataset[idx] if self.tgt_dataset is not None else None

        # Convert to tensors and add special tokens
        src_tensor = self._maybe_add_special_tokens(src_ids, 'src')
        tgt_tensor = self._maybe_add_special_tokens(tgt_ids, 'tgt') if tgt_ids is not None else None

        # Log first few examples for debugging
        start_line = self.offset if self.offset is not None else 0
        if self.is_train and idx < start_line + 5 and (self.verbose_dataloader or self.device_rank == 0):
            logger.info(
                f"[IndexedCorpus Rank {self.device_rank}] Example {idx}: "
                f"SRC_LEN={len(src_tensor)} TGT_LEN={len(tgt_tensor) if tgt_tensor is not None else 0}"
            )

        return {
            'src': src_tensor,
            'tgt': tgt_tensor,
            'line_idx': idx,
        }

    def __iter__(self):
        """Iterate over examples."""
        # Determine start index
        if self._line_idx_restore is not None:
            start_idx = self._line_idx_restore
            logger.info(
                f"Restoring from checkpoint: starting at example {start_idx}"
            )
        elif self.offset is not None:
            start_idx = self.offset
        else:
            start_idx = 0

        # Generate indices
        if self.stride is not None:
            # Distributed training: every stride-th example
            indices = range(start_idx, self.num_examples, self.stride)
        else:
            # Single process: all examples from start
            indices = range(start_idx, self.num_examples)

        # Create examples
        for idx in indices:
            example = self._make_example_dict(idx)

            # NOTE: Transforms are NOT applied for indexed datasets!
            # Indexed datasets contain pre-tokenized token IDs, while transforms
            # expect token strings. Any preprocessing (tokenization, filtering, etc.)
            # should be done in the preprocessing script, not during training.
            #
            # If you need to filter or transform data, do it during preprocessing:
            #   python -m mammoth.scripts.preprocess_indexed ...

            yield example

    def __len__(self):
        """Return approximate length (for progress tracking)."""
        if self.stride is not None:
            # Distributed: approximate number of examples for this worker
            start_idx = self.offset if self.offset is not None else 0
            return (self.num_examples - start_idx + self.stride - 1) // self.stride
        else:
            # Single process
            start_idx = self.offset if self.offset is not None else 0
            return self.num_examples - start_idx
