"""
Tests for indexed dataset implementation.
"""

import os
import tempfile
import shutil
import numpy as np

from mammoth.inputters.indexed_dataset import (
    IndexedDatasetBuilder,
    IndexedDataset,
    optimal_dtype,
    exists,
)


def test_optimal_dtype():
    """Test optimal dtype selection."""
    assert optimal_dtype(100) == np.uint8
    assert optimal_dtype(256) == np.uint16
    assert optimal_dtype(70000) == np.int32


def test_builder_and_reader():
    """Test writing and reading indexed dataset."""
    # Create temporary directory
    tmpdir = tempfile.mkdtemp()
    try:
        prefix = os.path.join(tmpdir, 'test')
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'

        # Build dataset
        builder = IndexedDatasetBuilder(bin_path, dtype=np.int32)

        # Add some documents
        doc1 = [1, 2, 3, 4, 5]
        doc2 = [10, 20, 30]
        doc3 = [100, 200, 300, 400]

        builder.add_document(doc1)
        builder.add_document(doc2)
        builder.add_document(doc3)

        builder.finalize(idx_path)

        # Verify files exist
        assert exists(prefix)
        assert os.path.exists(bin_path)
        assert os.path.exists(idx_path)

        # Read dataset
        dataset = IndexedDataset(prefix)

        # Verify metadata
        assert len(dataset) == 3
        assert dataset.total_tokens == len(doc1) + len(doc2) + len(doc3)

        # Verify documents
        assert list(dataset[0]) == doc1
        assert list(dataset[1]) == doc2
        assert list(dataset[2]) == doc3

        # Verify sentence lengths
        assert dataset.get_sentence_lengths(0) == [len(doc1)]
        assert dataset.get_sentence_lengths(1) == [len(doc2)]
        assert dataset.get_sentence_lengths(2) == [len(doc3)]

        print("✓ All indexed dataset tests passed!")

    finally:
        # Cleanup
        shutil.rmtree(tmpdir)


def test_empty_document():
    """Test handling of empty documents."""
    tmpdir = tempfile.mkdtemp()
    try:
        prefix = os.path.join(tmpdir, 'test_empty')
        bin_path = prefix + '.bin'
        idx_path = prefix + '.idx'

        builder = IndexedDatasetBuilder(bin_path, dtype=np.int32)

        # Add document with content
        doc1 = [1, 2, 3]
        builder.add_document(doc1)

        # Try to add empty document (should be skipped)
        builder.add_document([])

        # Add another document with content
        doc2 = [4, 5, 6]
        builder.add_document(doc2)

        builder.finalize(idx_path)

        # Read dataset
        dataset = IndexedDataset(prefix)

        # Should only have 2 documents (empty one skipped)
        assert len(dataset) == 2
        assert list(dataset[0]) == doc1
        assert list(dataset[1]) == doc2

        print("✓ Empty document test passed!")

    finally:
        shutil.rmtree(tmpdir)


if __name__ == '__main__':
    test_optimal_dtype()
    test_builder_and_reader()
    test_empty_document()
    print("\n✓ All tests passed successfully!")
