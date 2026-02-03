# ROCTx Profiling Guide for Mammoth on LUMI

## Overview

Mammoth uses AMD ROCTx markers for GPU profiling on LUMI supercomputer. ROCTx provides low-overhead performance markers that integrate with AMD profiling tools like rocprofv3 and Omniperf.

This guide covers:
- Prerequisites and setup
- Basic usage for single-node and multi-node training
- Configuration options
- Analyzing profiling results
- Troubleshooting common issues

## Prerequisites

### 1. Load ROCm Module on LUMI

```bash
module load LUMI/23.09
module load rocm/6.2.4  # or newer
```

### 2. Set PYTHONPATH for ROCTx Python Bindings

ROCTx Python bindings are installed with ROCm but need to be added to `PYTHONPATH`:

```bash
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH
```

Add this to your job scripts or `.bashrc` for convenience.

### 3. Verify ROCTx Availability

```bash
python -c "import roctx; print('ROCTx available')"
```

If this command succeeds, you're ready to use ROCTx profiling.

## Basic Usage

### Single-Node Training

For single-node profiling, wrap your training command with `rocprofv3`:

```bash
rocprofv3 --marker-trace --output-format csv --output-dir ./profiling_output -- \
    python train.py -config config.yaml --enable_profiling
```

**Important**: The `--enable_profiling` flag must be passed to `train.py` to activate profiling markers in the code.

### Multi-Node Training (SLURM)

For multi-node distributed training, each rank needs its own profiling wrapper:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH --partition=standard-g

# Setup environment
module load LUMI/23.09
module load rocm/6.2.4
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH

# Create profiling output directory
mkdir -p ./profiling

# Run training with profiling on all ranks
srun bash -c "
    rocprofv3 --marker-trace --output-format csv \
        --output-dir ./profiling/rank_\${SLURM_PROCID} \
        -- python train.py \
            -config config.yaml \
            --enable_profiling \
            --node_rank \${SLURM_PROCID}
"
```

This creates separate profiling directories for each rank:
- `./profiling/rank_0/marker_api_trace.csv`
- `./profiling/rank_1/marker_api_trace.csv`
- ...

## Configuration

### YAML Configuration

Add profiling settings to your YAML config:

```yaml
# ROCTx Profiling Configuration
enable_profiling: true
profiling_output_prefix: my_experiment  # Optional: prefix for trace files
```

### Command-Line Options

Or use command-line flags:

```bash
python train.py \
    -config config.yaml \
    --enable_profiling \
    --profiling_output_prefix my_experiment
```

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `enable_profiling` | `false` | Enable/disable ROCTx profiling markers |
| `profiling_output_prefix` | `mammoth_trace` | Prefix for profiling output files |

**Note**: Output directory is controlled by `rocprofv3 --output-dir`, not by Mammoth options.

## Profiling Markers

Mammoth includes 14 fine-grained profiling markers throughout the training pipeline:

### Data Loading Markers

- **`batch_queue_get`**: Time spent retrieving batches from the queue
- **`batch_tensor_reattach_from_cpu`**: Converting NumPy arrays back to tensors
- **`batch_tensor_detach_to_cpu`**: Converting tensors to NumPy for IPC (batch producer process)
- **`data_transfer_to_device`**: Moving data from CPU to GPU

### Training Loop Markers

- **`data_preparation_step_{N}`**: Preparing batch data for training step N
- **`gradient_accumulation_step_{N}`**: All forward/backward passes for step N
- **`forward_pass_batch_{K}`**: Forward pass for individual batch K within a step
- **`loss_computation_batch_{K}`**: Loss calculation for batch K
- **`backward_pass_batch_{K}`**: Backward pass for batch K

### Distributed Training Markers

- **`gradient_sync_step_{N}`**: Gradient synchronization across ranks for step N
- **`allreduce_{component_name}`**: Per-component allreduce operation (e.g., `allreduce_shared_encoder`)
- **`barrier_post_validation`**: Synchronization barrier after validation

### Optimization Markers

- **`optimizer_step_{N}`**: Optimizer parameter updates for step N

### Validation Markers

- **`validation_forward_pass`**: Forward passes during validation

## Viewing Results

### CSV Trace Files

After profiling completes, check the output directory:

```bash
ls profiling_output/
# marker_api_trace.csv  - ROCTx marker timings (main file)
# hip_api_trace.csv     - HIP API calls (if --hip-trace enabled)
# hsa_api_trace.csv     - HSA API calls (if --hsa-trace enabled)
```

The `marker_api_trace.csv` file contains ROCTx marker data with columns:
- `Name`: Marker name
- `Start`: Start timestamp (ns)
- `End`: End timestamp (ns)
- `Duration_ns`: Duration in nanoseconds

### Analyzing Marker Timings with Python

```python
import pandas as pd

# Load marker trace
df = pd.read_csv('profiling_output/marker_api_trace.csv')

# Group by marker name and compute statistics
stats = df.groupby('Name')['Duration_ns'].agg(['count', 'mean', 'std', 'sum'])

# Convert to milliseconds for readability
stats['mean_ms'] = stats['mean'] / 1e6
stats['sum_ms'] = stats['sum'] / 1e6
stats['pct'] = 100 * stats['sum'] / stats['sum'].sum()

# Sort by total time spent
print(stats.sort_values('sum_ms', ascending=False))
```

Example output:

```
                                    count      mean         std        sum    mean_ms   sum_ms       pct
gradient_accumulation_step_*          100  45231234  1234567  4523123400  45.23    4523.12    65.2
forward_pass_batch_*                  400  8123456   234567   3249382400  8.12     3249.38    46.8
allreduce_shared_encoder              100  2134567   123456    213456700  2.13      213.46     3.1
...
```

### Using AMD Profiling Tools

#### Omniperf

For comprehensive GPU profiling and analysis:

```bash
# Collect detailed profile
omniperf profile -n my_profile -- \
    rocprofv3 --marker-trace -- python train.py -config config.yaml --enable_profiling

# Analyze results
omniperf analyze -p workloads/my_profile
```

#### rocprof Command-Line Analysis

```bash
# View summary statistics
rocprofv3 --stats -- python train.py -config config.yaml --enable_profiling
```

#### CSV Analysis with Spreadsheets

Open `marker_api_trace.csv` in Excel, LibreOffice, or Google Sheets for manual analysis and visualization.

## Performance Impact

### Profiling Disabled

When profiling is disabled (`enable_profiling: false`), there is **zero overhead**. The profiling markers become no-op context managers that compile to essentially nothing.

### Profiling Enabled

When profiling is enabled with ROCTx markers only:
- **Overhead**: < 2% (ROCTx markers are very lightweight)
- **Memory**: ~5-10 MB per rank for marker storage

When enabling additional traces (HIP/HSA):
- **Overhead**: ~20-30% (significantly higher)
- **Memory**: ~100-500 MB per rank
- **Use case**: Detailed kernel-level debugging only

## Advanced Usage

### Profile Specific Training Phases

You can add custom profiling markers in your code:

```python
from mammoth.utils.profiling import get_roctx_range

# In your training code
roctx_range = get_roctx_range(enable_profiling=True)

with roctx_range("my_custom_operation"):
    # Your code here
    result = expensive_computation()
```

### Combine with HIP/HSA Tracing

For detailed GPU kernel profiling:

```bash
rocprofv3 \
    --marker-trace \
    --hip-trace \
    --hsa-trace \
    --output-format csv \
    --output-dir ./detailed_profiling \
    -- python train.py -config config.yaml --enable_profiling
```

**Warning**: HIP/HSA tracing has significant overhead (~20-30%). Use only for debugging specific issues.

### Profile Only Selected Steps

Modify your YAML config to run fewer steps when profiling:

```yaml
train_steps: 100         # Just enough steps to capture patterns
valid_steps: 50
save_checkpoint_steps: 100
```

## Comparison to PyTorch Profiler (Previous Implementation)

Mammoth previously used PyTorch profiler. Here's how ROCTx compares:

| Feature | PyTorch Profiler (Old) | ROCTx (New) |
|---------|------------------------|-------------|
| **Platform** | Cross-platform (AMD/NVIDIA/CPU) | AMD GPUs only |
| **Overhead** | 5-10% | <2% |
| **Output Format** | TensorBoard JSON traces | CSV traces |
| **Visualization** | TensorBoard | CSV analysis, Omniperf |
| **Multi-node Support** | Requires careful setup | Native per-rank support |
| **Scheduling** | wait/warmup/active cycles | External control via rocprofv3 |
| **Zero-overhead Mode** | ✓ (noop) | ✓ (noop) |
| **AMD Tool Integration** | Limited | Excellent (Omniperf, rocprof) |

## Troubleshooting

### ROCTx Import Error

**Problem**:
```
ModuleNotFoundError: No module named 'roctx'
```

**Solution**:
Set PYTHONPATH to ROCm installation:
```bash
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH

# Verify
python -c "import roctx; print('Success')"
```

**Note**: Python version in path (3.12) must match your environment. Check with:
```bash
ls /opt/rocm/lib/
# Look for python3.X directories
```

### Empty Trace Files

**Problem**: `marker_api_trace.csv` is empty or missing.

**Solution**: Ensure you're running with `rocprofv3` wrapper:

```bash
# Correct (profiling enabled):
rocprofv3 --marker-trace -- python train.py --enable_profiling

# Incorrect (no traces generated):
python train.py --enable_profiling  # Missing rocprofv3 wrapper
```

### Warning: "ROCTx profiling requested but ROCTx not available"

**Problem**: Training runs but no profiling markers are created.

**Solution**:
1. Check PYTHONPATH is set correctly
2. Verify ROCm module is loaded
3. Check Python version matches ROCTx installation

```bash
module load rocm/6.2.4
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH
python -c "import roctx"
```

### Permission Denied for Trace Files

**Problem**: `rocprofv3` fails with permission errors.

**Solution**: Ensure output directory is writable:
```bash
mkdir -p ./profiling_output
chmod 755 ./profiling_output
```

On LUMI, use `/scratch` filesystem, not `/home` or `/tmp`:
```bash
mkdir -p /scratch/project_XXXXXX/profiling
rocprofv3 --output-dir /scratch/project_XXXXXX/profiling/rank_${SLURM_PROCID} -- ...
```

### "rocprofv3 command not found"

**Problem**: `rocprofv3` is not available.

**Solution**: Load ROCm 6.0+ module:
```bash
module load rocm/6.2.4  # or newer

# Verify
which rocprofv3
rocprofv3 --version
```

**Note**: `rocprofv3` requires ROCm 6.0+. Older versions only have `rocprof` (legacy tool).

### High Memory Usage During Profiling

**Problem**: Profiling runs out of memory.

**Solution**:
1. Reduce training steps: `train_steps: 100`
2. Disable HIP/HSA traces (if enabled): Use only `--marker-trace`
3. Profile fewer ranks: Profile just rank 0 to understand patterns

### Profiling Slows Down Training Significantly

**Problem**: Profiling has >10% overhead.

**Solution**:
- If using only `--marker-trace`: Overhead should be <2%. Check for other issues.
- If using `--hip-trace` or `--hsa-trace`: This is expected (~20-30%). Use only for debugging.
- Remove detailed traces and keep only markers:
  ```bash
  rocprofv3 --marker-trace -- ...  # Fast
  # vs
  rocprofv3 --marker-trace --hip-trace --hsa-trace -- ...  # Slow
  ```

## Example Workflows

### Workflow 1: Initial Performance Analysis

**Goal**: Understand where training time is spent.

```bash
# Step 1: Profile first 500 steps
rocprofv3 --marker-trace --output-format csv --output-dir ./profiling -- \
    python train.py -config config.yaml \
    --enable_profiling \
    --train_steps 500

# Step 2: Analyze marker timings
python analyze_traces.py profiling/marker_api_trace.csv

# Step 3: Identify bottlenecks (e.g., gradient_sync takes 40% of time)
```

### Workflow 2: Communication Bottleneck Analysis

**Goal**: Optimize distributed training communication.

```bash
# Profile with focus on allreduce operations
rocprofv3 --marker-trace --output-format csv -- \
    python train.py -config config.yaml \
    --enable_profiling \
    --train_steps 200

# Analyze CSV for allreduce timings
import pandas as pd
df = pd.read_csv('marker_api_trace.csv')
allreduce = df[df['Name'].str.contains('allreduce')]
print(allreduce.groupby('Name')['Duration_ns'].agg(['count', 'mean', 'sum']))
```

Look for:
- Long allreduce times → Network bottleneck
- Unbalanced component times → Load imbalance

### Workflow 3: Data Loading Optimization

**Goal**: Ensure data loading doesn't bottleneck GPU compute.

```bash
# Profile data pipeline markers
rocprofv3 --marker-trace -- python train.py --enable_profiling

# Analyze data loading markers
import pandas as pd
df = pd.read_csv('marker_api_trace.csv')
data_markers = df[df['Name'].str.contains('batch_queue|tensor_reattach|data_transfer')]
print(data_markers.groupby('Name')['Duration_ns'].describe())
```

If `batch_queue_get` is slow:
- Increase `num_workers` in dataloader configuration
- Check disk I/O on `/scratch`

If `data_transfer_to_device` is slow:
- Batch size may be too large
- Check PCIe bandwidth

### Workflow 4: Before/After Optimization Comparison

**Goal**: Quantify optimization impact.

```bash
# Baseline
rocprofv3 --marker-trace --output-dir ./baseline -- \
    python train.py -config baseline.yaml --enable_profiling --train_steps 200

# After optimization
rocprofv3 --marker-trace --output-dir ./optimized -- \
    python train.py -config optimized.yaml --enable_profiling --train_steps 200

# Compare
python compare_traces.py baseline/marker_api_trace.csv optimized/marker_api_trace.csv
```

## Example Analysis Script

Save this as `analyze_traces.py`:

```python
#!/usr/bin/env python3
"""Analyze ROCTx profiling traces from Mammoth training."""

import pandas as pd
import sys

def analyze_trace(csv_path):
    """Analyze ROCTx marker trace and print summary statistics."""
    df = pd.read_csv(csv_path)

    print(f"Analyzing: {csv_path}")
    print(f"Total markers: {len(df)}")
    print(f"Unique marker types: {df['Name'].nunique()}")
    print()

    # Group by marker name
    stats = df.groupby('Name')['Duration_ns'].agg(['count', 'mean', 'std', 'sum'])
    stats['mean_ms'] = stats['mean'] / 1e6
    stats['sum_ms'] = stats['sum'] / 1e6
    stats['pct'] = 100 * stats['sum'] / stats['sum'].sum()

    # Sort by total time
    stats = stats.sort_values('pct', ascending=False)

    print("Top 10 time-consuming operations:")
    print(stats.head(10).to_string())
    print()

    # Category analysis
    categories = {
        'Data Loading': ['batch_queue', 'tensor_reattach', 'tensor_detach', 'data_transfer'],
        'Forward Pass': ['forward_pass'],
        'Loss & Backward': ['loss_computation', 'backward_pass'],
        'Communication': ['allreduce', 'gradient_sync', 'barrier'],
        'Optimization': ['optimizer_step'],
        'Validation': ['validation'],
    }

    print("Time by category:")
    for category, keywords in categories.items():
        mask = df['Name'].str.contains('|'.join(keywords), case=False)
        category_time = df.loc[mask, 'Duration_ns'].sum() / 1e6  # ms
        category_pct = 100 * category_time / (df['Duration_ns'].sum() / 1e6)
        print(f"  {category:20s}: {category_time:10.2f} ms ({category_pct:5.1f}%)")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <marker_api_trace.csv>")
        sys.exit(1)

    analyze_trace(sys.argv[1])
```

Run with:
```bash
python analyze_traces.py profiling_output/marker_api_trace.csv
```

## References

- [ROCTx Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html)
- [rocprofv3 User Guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/rocprofv3.html)
- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)
- [Omniperf User Guide](https://rocm.docs.amd.com/projects/omniperf/en/latest/)

## Getting Help

If you encounter issues not covered in this guide:

1. Check the [LUMI documentation](https://docs.lumi-supercomputer.eu/)
2. Verify your ROCm version: `module list | grep rocm`
3. Test ROCTx availability: `python -c "import roctx"`
4. Check for known issues in the [Mammoth GitHub repository](https://github.com/Helsinki-NLP/mammoth)

For LUMI-specific support, contact LUMI support desk with:
- Your job ID
- Full error message
- ROCm version (`cat /opt/rocm/.info/version`)
- Python version (`python --version`)
