# GPU Profiling Guide for Mammoth

## Overview

Mammoth provides unified GPU profiling with automatic detection of the available profiler:
- **NVTX markers** on NVIDIA GPUs (Puhti V100, etc.)
- **ROCTx markers** on AMD GPUs (LUMI MI250X, etc.)
- **Silent no-op fallback** when neither is available (e.g., development laptops)

The profiling system uses a **zero-configuration approach**: markers are always present in the code with minimal overhead (~0.1%), but traces are only collected when you run under an external profiler wrapper (`nsys` for NVIDIA or `rocprofv3` for AMD).

**Key Benefits**:
- Same code works seamlessly on both NVIDIA and AMD systems
- No code changes needed when switching platforms
- Minimal overhead when not profiling
- 14 fine-grained markers throughout the training pipeline

This guide covers:
- Platform-specific setup (NVIDIA/AMD)
- Basic usage for single-node and multi-node training
- Analyzing profiling results
- Troubleshooting common issues

## Auto-Detection

Mammoth automatically detects which profiler to use based on available libraries:

1. **NVTX** (highest priority): Checks for `nvtx` package or `torch.cuda.nvtx`
2. **ROCTx** (fallback): Checks for `roctx` module from ROCm
3. **No-op** (final fallback): Silent no-op when neither is available

You'll see one of these messages at startup:
```
INFO - NVTX markers ACTIVE - traces collected via nsys/nvprof
INFO - ROCTx markers ACTIVE - traces collected via rocprofv3
```

If neither message appears, profiling gracefully falls back to no-op (no warnings, no errors).

## Platform-Specific Setup

### NVIDIA Systems (Puhti)

#### Prerequisites

**1. Install NVTX Python Bindings**

```bash
# Option 1: Standalone nvtx package (recommended)
pip install nvtx

# Option 2: Use PyTorch's built-in NVTX (usually already available)
python -c "import torch.cuda.nvtx; print('NVTX available')"
```

**2. Verify Installation**

```bash
python -c "import nvtx; print('NVTX available')"
# OR
python -c "import torch.cuda.nvtx; print('NVTX available')"
```

**3. Load NVIDIA Profiling Tools** (if using nsys)

```bash
# On Puhti or other HPC systems
module load nvidia-nsight-systems  # or similar module

# Verify
which nsys
nsys --version
```

#### Basic Usage on NVIDIA

**Single-Node Training**:

```bash
# Regular training (no profiling, ~0.1% overhead)
python train.py -config config.yaml

# With profiling (traces collected)
nsys profile -t nvtx,cuda,cublas,cudnn \
    --output profiling/trace \
    --stats=true \
    python train.py -config config.yaml
```

**Multi-Node Training (SLURM)**:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:v100:4
#SBATCH --partition=gputest

module load pytorch
module load nvidia-nsight-systems  # or equivalent

# Create profiling output directory
mkdir -p ./profiling

# Run training with profiling on all ranks
srun bash -c "
    nsys profile \
        -t nvtx,cuda,cublas,cudnn,osrt \
        -o ./profiling/rank_\${SLURM_PROCID} \
        --stats=true \
        --force-overwrite true \
        --delay 120 \
        --duration 30 \
        python train.py \
            -config config.yaml \
            --node_rank \${SLURM_PROCID}
"
```

This creates profiling traces for each rank:
- `./profiling/rank_0.nsys-rep`
- `./profiling/rank_1.nsys-rep`
- ...

**Viewing NVIDIA Traces**:

```bash
# GUI viewer (requires X11 forwarding or local copy)
nsys-ui profiling/rank_0.nsys-rep

# Command-line stats
nsys stats profiling/rank_0.nsys-rep
```

### AMD Systems (LUMI)

#### Prerequisites

**1. Load ROCm Module**

```bash
module load LUMI/23.09
module load rocm/6.2.4  # or newer (requires ROCm 6.0+ for rocprofv3)
```

**2. Set PYTHONPATH for ROCTx Python Bindings**

```bash
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH
```

Add this to your job scripts or `.bashrc` for convenience.

**3. Verify Installation**

```bash
python -c "import roctx; print('ROCTx available')"
which rocprofv3
rocprofv3 --version
```

**Note**: Python version in path (e.g., `python3.12`) must match your environment:
```bash
ls /opt/rocm/lib/  # Look for python3.X directories
```

#### Basic Usage on AMD

**Single-Node Training**:

```bash
# Regular training (no profiling, ~0.1% overhead)
python train.py -config config.yaml

# With profiling (CSV traces collected)
rocprofv3 --marker-trace --output-format csv --output-dir ./profiling -- \
    python train.py -config config.yaml
```

**Multi-Node Training (SLURM)**:

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
            --node_rank \${SLURM_PROCID}
"
```

This creates separate profiling directories for each rank:
- `./profiling/rank_0/marker_api_trace.csv`
- `./profiling/rank_1/marker_api_trace.csv`
- ...

**Viewing AMD Traces**:

See the "Analyzing AMD CSV Traces" section below.

## Profiling Markers

Mammoth includes **14 fine-grained profiling markers** throughout the training pipeline:

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

## Analyzing Profiling Results

### NVIDIA: Viewing nsys Traces

**GUI Analysis** (recommended):

```bash
# Copy .nsys-rep file to local machine with GUI
scp puhti:/path/to/profiling/rank_0.nsys-rep .

# Open in Nsight Systems GUI
nsys-ui rank_0.nsys-rep
```

In the GUI:
1. Timeline View: Shows NVTX markers as colored ranges
2. Look for the 14 Mammoth markers in the timeline
3. Check marker durations and nesting (e.g., `forward_pass_batch_0` inside `gradient_accumulation_step_1`)
4. Correlate markers with CUDA kernel launches

**Command-Line Analysis**:

```bash
# Print summary statistics
nsys stats rank_0.nsys-rep

# Export to SQLite for custom queries
nsys export --type sqlite rank_0.nsys-rep
```

### AMD: Analyzing CSV Traces

After profiling with `rocprofv3`, check the output directory:

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

**Python Analysis Script**:

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

**CSV Analysis with Spreadsheets**:

Open `marker_api_trace.csv` in Excel, LibreOffice, or Google Sheets for manual analysis and visualization.

**Using Omniperf** (comprehensive GPU analysis):

```bash
# Collect detailed profile
omniperf profile -n my_profile -- \
    rocprofv3 --marker-trace -- python train.py -config config.yaml

# Analyze results
omniperf analyze -p workloads/my_profile
```

## Performance Impact

### Without Profiler Wrapper (Default)

When running without `nsys`/`rocprofv3` wrapper:
- **Overhead**: ~0.1% (marker function call overhead only)
- **Memory**: Negligible
- **Use case**: Regular training

### With Profiler Wrapper - Markers Only

NVIDIA:
```bash
nsys profile -t nvtx,cuda python train.py -config config.yaml
```

AMD:
```bash
rocprofv3 --marker-trace python train.py -config config.yaml
```

- **Overhead**: <2% (markers are very lightweight)
- **Memory**: ~5-10 MB per rank for marker storage
- **Use case**: Performance analysis and optimization

### With Profiler Wrapper - Full Tracing

NVIDIA (with full CUDA/cuBLAS/cuDNN traces):
```bash
nsys profile -t nvtx,cuda,cublas,cudnn,osrt python train.py -config config.yaml
```

AMD (with HIP/HSA kernel traces):
```bash
rocprofv3 --marker-trace --hip-trace --hsa-trace python train.py -config config.yaml
```

- **Overhead**: ~20-30% (significantly higher)
- **Memory**: ~100-500 MB per rank
- **Use case**: Detailed kernel-level debugging only

## Adding Custom Profiling Markers

You can add custom profiling markers in your code:

```python
from mammoth.utils.profiling import get_profiler_range

# Get the appropriate profiler (NVTX/ROCTx/noop)
profiler_range = get_profiler_range()

# In your training code
with profiler_range("my_custom_operation"):
    # Your code here
    result = expensive_computation()
```

This works identically on both NVIDIA and AMD systems.

**Legacy API** (still works but less clear):

```python
from mammoth.utils.profiling import get_roctx_range

roctx_range = get_roctx_range()  # Actually returns NVTX on NVIDIA!
with roctx_range("operation"):
    pass
```

## Platform Comparison Table

| Feature | NVIDIA (nsys) | AMD (rocprofv3) |
|---------|---------------|-----------------|
| **Profiler Backend** | NVTX | ROCTx |
| **Wrapper Command** | `nsys profile -t nvtx` | `rocprofv3 --marker-trace` |
| **Output Format** | `.nsys-rep` (binary) | `.csv` (text) |
| **Visualization** | Nsight Systems GUI | CSV analysis, Omniperf |
| **Multi-node Support** | ✓ Per-rank traces | ✓ Per-rank traces |
| **Marker Overhead** | <2% | <2% |
| **Full Trace Overhead** | ~20-30% | ~20-30% |
| **Zero-overhead Mode** | ✓ (no wrapper) | ✓ (no wrapper) |
| **Command-line Analysis** | `nsys stats` | CSV tools, pandas |
| **Advanced Tool Integration** | Nsight Compute | Omniperf, rocprof |

## Troubleshooting

### NVTX Issues (NVIDIA)

**Problem**: `ModuleNotFoundError: No module named 'nvtx'`

**Solution**:
```bash
# Install nvtx package
pip install nvtx

# OR verify torch.cuda.nvtx is available
python -c "import torch.cuda.nvtx; print('Available')"
```

**Problem**: `nsys: command not found`

**Solution**:
```bash
# Load appropriate module (HPC systems)
module load nvidia-nsight-systems

# OR install locally
# Download from: https://developer.nvidia.com/nsight-systems
```

**Problem**: Empty NVTX ranges in nsys-ui

**Solution**: Ensure you're using `-t nvtx` flag:
```bash
# Correct:
nsys profile -t nvtx,cuda python train.py

# Incorrect (no NVTX traces):
nsys profile python train.py  # Missing -t nvtx
```

### ROCTx Issues (AMD)

**Problem**: `ModuleNotFoundError: No module named 'roctx'`

**Solution**:
```bash
# Set PYTHONPATH to ROCm installation
export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH

# Verify
python -c "import roctx; print('Success')"
```

**Note**: Python version in path (3.12) must match your environment:
```bash
ls /opt/rocm/lib/  # Look for python3.X directories
```

**Problem**: `rocprofv3: command not found`

**Solution**:
```bash
# Load ROCm 6.0+ module (rocprofv3 requires ROCm 6.0+)
module load rocm/6.2.4

# Verify
which rocprofv3
rocprofv3 --version
```

**Note**: Older ROCm versions (<6.0) only have `rocprof` (legacy tool), not `rocprofv3`.

**Problem**: Empty `marker_api_trace.csv`

**Solution**: Ensure you're running with `rocprofv3` wrapper:
```bash
# Correct (profiling enabled):
rocprofv3 --marker-trace -- python train.py

# Incorrect (no traces generated):
python train.py  # Missing rocprofv3 wrapper
```

### General Issues

**Problem**: No profiler detection message at startup

**Expected**: You should see either "NVTX markers ACTIVE" or "ROCTx markers ACTIVE"

**Solution**:
1. Check that nvtx or roctx is importable:
   ```bash
   python -c "import nvtx; print('NVTX OK')"
   python -c "import roctx; print('ROCTx OK')"
   ```
2. Verify PYTHONPATH is set (for ROCTx on AMD)
3. Check module loads (on HPC systems)

**Problem**: High memory usage during profiling

**Solution**:
1. Reduce training steps: `--train_steps 100` (in YAML or CLI)
2. Disable full kernel traces: Use only `-t nvtx` or `--marker-trace`
3. Profile fewer ranks: Profile just rank 0 to understand patterns

**Problem**: Profiling slows down training significantly

**Solution**:
- If using only markers (`-t nvtx` or `--marker-trace`): Overhead should be <2%
- If using full traces: This is expected (~20-30%). Reduce to markers only for regular profiling

## Example Workflows

### Workflow 1: Initial Performance Analysis

**Goal**: Understand where training time is spent.

**NVIDIA**:
```bash
nsys profile -t nvtx,cuda --output profiling/trace python train.py -config config.yaml
# View in nsys-ui to see marker timeline
```

**AMD**:
```bash
rocprofv3 --marker-trace --output-format csv --output-dir ./profiling -- \
    python train.py -config config.yaml

# Analyze with Python
python analyze_traces.py profiling/marker_api_trace.csv
```

### Workflow 2: Communication Bottleneck Analysis

**Goal**: Optimize distributed training communication.

Look for:
- Long `allreduce_*` times → Network bottleneck
- Unbalanced `gradient_sync` times → Load imbalance

**AMD CSV analysis**:
```python
import pandas as pd
df = pd.read_csv('profiling/marker_api_trace.csv')
allreduce = df[df['Name'].str.contains('allreduce')]
print(allreduce.groupby('Name')['Duration_ns'].agg(['count', 'mean', 'sum']))
```

### Workflow 3: Data Loading Optimization

**Goal**: Ensure data loading doesn't bottleneck GPU compute.

Look for:
- If `batch_queue_get` is slow: Increase `num_workers` in dataloader
- If `data_transfer_to_device` is slow: Check batch size or PCIe bandwidth

### Workflow 4: Before/After Optimization Comparison

**Goal**: Quantify optimization impact.

```bash
# Baseline
nsys profile -t nvtx -o baseline python train.py -config baseline.yaml
# OR: rocprofv3 --marker-trace --output-dir baseline -- python train.py -config baseline.yaml

# After optimization
nsys profile -t nvtx -o optimized python train.py -config optimized.yaml
# OR: rocprofv3 --marker-trace --output-dir optimized -- python train.py -config optimized.yaml

# Compare timelines/CSVs
```

## Example AMD Analysis Script

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
python analyze_traces.py profiling/marker_api_trace.csv
```

## Comparison to PyTorch Profiler (Previous Implementation)

Mammoth previously used PyTorch profiler. Here's how the current GPU-specific profilers compare:

| Feature | PyTorch Profiler (Old) | NVTX/ROCTx (New) |
|---------|------------------------|------------------|
| **Platform** | Cross-platform (AMD/NVIDIA/CPU) | GPU-specific, auto-detected |
| **Overhead** | 5-10% | <2% |
| **Output Format** | TensorBoard JSON traces | nsys-rep (NVIDIA), CSV (AMD) |
| **Visualization** | TensorBoard | Nsight Systems (NVIDIA), CSV/Omniperf (AMD) |
| **Multi-node Support** | Requires careful setup | Native per-rank support |
| **Configuration** | Requires flags/YAML | Zero-config (external wrapper) |
| **Zero-overhead Mode** | ✓ (noop) | ✓ (noop) |
| **Vendor Tool Integration** | Limited | Excellent (Nsight/Omniperf) |

## References

### NVIDIA
- [NVTX Documentation](https://nvidia.github.io/nvtx/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/)
- [Nsight Compute User Guide](https://docs.nvidia.com/nsight-compute/)

### AMD
- [ROCTx Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofiler-sdk-roctx.html)
- [rocprofv3 User Guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/rocprofv3.html)
- [Omniperf User Guide](https://rocm.docs.amd.com/projects/omniperf/en/latest/)
- [LUMI Documentation](https://docs.lumi-supercomputer.eu/)

## Getting Help

If you encounter issues not covered in this guide:

**NVIDIA (Puhti)**:
1. Check NVTX availability: `python -c "import nvtx"`
2. Verify nsys installation: `which nsys; nsys --version`
3. Check Puhti documentation for profiling tools

**AMD (LUMI)**:
1. Check the [LUMI documentation](https://docs.lumi-supercomputer.eu/)
2. Verify your ROCm version: `module list | grep rocm`
3. Test ROCTx availability: `python -c "import roctx"`
4. Contact LUMI support with job ID and error messages

**General**:
- Check for known issues in the [Mammoth GitHub repository](https://github.com/Helsinki-NLP/mammoth)
- Verify auto-detection is working: look for profiler activation message at startup
