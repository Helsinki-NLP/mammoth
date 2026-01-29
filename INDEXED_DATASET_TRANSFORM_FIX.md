# Indexed Dataset Transform Compatibility Fix

## Problem Identified

You correctly identified that `TransformPipe(opts, transforms_to_apply)` **does not work** with indexed datasets.

### Root Cause

**Data format mismatch:**
- **Text datasets (ParallelCorpus):** Returns `example['src']` as **list of token strings** (e.g., `['▁Hello', '▁world']`)
- **Indexed datasets (IndexedCorpus):** Returns `example['src']` as **torch.Tensor of token IDs** (e.g., `tensor([4, 5, 6])`)

**Transform expectations:**
- All MAMMOTH transforms expect token strings, not token IDs
- Example: `PrefixTransform._prepend()` does `side_prefix.split() + example[side]`
- This tries to concatenate a list with a tensor → **TypeError**

### Why It Broke

The original implementation naively copied the transform application from ParallelCorpus:

```python
# This doesn't work!
for idx in indices:
    example = self._make_example_dict(idx)  # Returns tensors
    if self.transforms is not None:
        example = self.transforms(example)   # Transforms expect strings!
    yield example
```

## Solution

### Code Changes

1. **`mammoth/inputters/indexed_corpus.py`**
   - **Removed transform application** from `__iter__`
   - Added clear comment explaining why transforms are not supported
   - Indexed datasets now yield examples directly without transform processing

2. **`mammoth/inputters/dataset.py`**
   - Added **warning** when transforms are specified for indexed datasets
   - Warning explains that transforms will be ignored
   - Guides users to preprocess data with desired transforms instead

3. **Documentation Updates**
   - Added prominent warning in `docs/INDEXED_DATASET_GUIDE.md`
   - Updated example config `examples/indexed_dataset_train.yaml`
   - Added troubleshooting section for transform-related issues

### Design Rationale

**Why not support transforms for indexed datasets?**

1. **Data is already preprocessed:** Indexed datasets contain final token IDs
2. **Format incompatibility:** Converting IDs → strings → IDs is wasteful
3. **Clear separation of concerns:** Preprocessing happens once, training uses prepared data
4. **Consistent with design:** Indexed datasets are for performance, not flexibility

**The correct workflow:**

```bash
# 1. Preprocess with all transforms applied
python -m mammoth.scripts.preprocess_indexed \
    --input data.txt \
    --output_prefix data/preprocessed \
    --vocab vocab.txt \
    --workers 8
    # Add custom preprocessing here if needed

# 2. Train with indexed data (NO transforms)
tasks:
  corpus:
    data_type: indexed
    path_src: data/preprocessed
    # transforms: []  # Omit or leave empty!
```

## Testing

Created test demonstrating:
- ✓ Indexed datasets work correctly without transforms
- ✓ Examples are yielded with proper BOS/EOS tokens
- ✓ No crashes or type errors

## User Impact

**Users must:**
1. Remove all transforms from task configs when using indexed datasets
2. Apply any desired preprocessing during the indexing step
3. Accept warning if transforms are mistakenly specified

**Benefits:**
- Clear error messages
- Prevents silent failures
- Guides users to correct usage

## Files Modified

- `mammoth/inputters/indexed_corpus.py` - Removed transform application
- `mammoth/inputters/dataset.py` - Added warning for misconfiguration
- `docs/INDEXED_DATASET_GUIDE.md` - Added limitation documentation
- `examples/indexed_dataset_train.yaml` - Removed misleading transform examples

## Key Takeaway

**Indexed datasets trade flexibility for performance.** All preprocessing must happen during indexing, not during training. This is consistent with similar systems like Megatron-LM's indexed datasets.