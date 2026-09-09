"""Tests for validation-dataset gating in DynamicDatasetIter.

A task should get a validation dataset iterator whenever it defines
`path_valid_src`, *regardless of its training weight*. weight=0 tasks are
eval-only (never sampled for training batches) but must still be validated
if the user gave them validation data.
"""

from mammoth.inputters.dataloader import task_needs_validation_dataset


def test_task_with_validation_paths_is_included_even_with_zero_weight():
    assert task_needs_validation_dataset(is_train=False, path_valid_src='some/path.dev') is True


def test_task_without_validation_paths_is_excluded():
    assert task_needs_validation_dataset(is_train=False, path_valid_src=None) is False


def test_training_mode_always_needs_a_dataset_regardless_of_validation_paths():
    # Training-time dataset construction doesn't depend on path_valid_src at all.
    assert task_needs_validation_dataset(is_train=True, path_valid_src=None) is True
    assert task_needs_validation_dataset(is_train=True, path_valid_src='some/path.dev') is True
