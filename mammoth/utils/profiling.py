"""
GPU profiling utilities with automatic NVTX/ROCTx detection.

This module provides profiling markers throughout the training pipeline with
automatic detection of the available profiler:
- NVTX markers on NVIDIA GPUs (Puhti V100, etc.)
- ROCTx markers on AMD GPUs (LUMI MI250X, etc.)
- Silent no-op fallback when neither is available

Profiling is controlled by external wrapper commands - markers are always active
in code but traces are only collected when running under the profiler wrapper.

Usage:
    from mammoth.utils.profiling import get_profiler_range

    # In your code (works on both NVIDIA and AMD):
    profiler_range = get_profiler_range()
    with profiler_range("operation_name"):
        # Your code here
        pass

Prerequisites:
    NVIDIA (Puhti):
        - Install nvtx: pip install nvtx
        - Or use PyTorch's built-in: torch.cuda.nvtx (usually available)

    AMD (LUMI):
        - ROCm 6.0+ installed on the system
        - PYTHONPATH includes ROCTx Python bindings:
          export PYTHONPATH=/opt/rocm/lib/python3.12/site-packages:$PYTHONPATH

Profiling Control:
    # Without profiler wrapper (no traces collected, ~0.1% overhead):
    python train.py -config config.yaml

    # NVIDIA profiling with nsys:
    nsys profile -t nvtx,cuda --output profiling/trace python train.py -config config.yaml

    # AMD profiling with rocprofv3:
    rocprofv3 --marker-trace --output-dir ./profiling -- python train.py -config config.yaml

See docs/GPU_PROFILING.md for detailed platform-specific instructions.
"""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Global flags for runtime check (cached to avoid repeated imports)
_nvtx_available = None
_roctx_available = None
_profiler_backend = None


def is_nvtx_available():
    """Check if NVTX is available (result is cached).

    Tries to import NVTX from multiple sources:
    1. nvtx package (pip install nvtx)
    2. torch.cuda.nvtx (usually available with PyTorch+CUDA)

    Returns:
        bool: True if NVTX can be imported, False otherwise
    """
    global _nvtx_available
    if _nvtx_available is None:
        try:
            # Try standalone nvtx package first
            import nvtx
            _nvtx_available = True
            logger.debug("NVTX profiling available (nvtx package)")
        except ImportError:
            try:
                # Fallback to PyTorch's built-in NVTX
                import torch.cuda.nvtx
                _nvtx_available = True
                logger.debug("NVTX profiling available (torch.cuda.nvtx)")
            except (ImportError, AttributeError):
                _nvtx_available = False
                logger.debug("NVTX not available")
    return _nvtx_available


def is_roctx_available():
    """Check if ROCTx is available (result is cached).

    Returns:
        bool: True if ROCTx Python module can be imported, False otherwise
    """
    global _roctx_available
    if _roctx_available is None:
        try:
            import roctx
            _roctx_available = True
            logger.debug("ROCTx profiling available")
        except ImportError:
            _roctx_available = False
            logger.debug("ROCTx not available")
    return _roctx_available


def detect_profiler_backend():
    """Auto-detect which profiler backend to use (result is cached).

    Detection priority:
    1. NVTX (NVIDIA GPUs) - most common in deep learning
    2. ROCTx (AMD GPUs) - for LUMI and other AMD systems
    3. None - graceful fallback with no-op markers

    Returns:
        str: "nvtx", "roctx", or None
    """
    global _profiler_backend
    if _profiler_backend is None:
        if is_nvtx_available():
            _profiler_backend = "nvtx"
            logger.info("NVTX markers ACTIVE - traces collected via nsys/nvprof")
        elif is_roctx_available():
            _profiler_backend = "roctx"
            logger.info("ROCTx markers ACTIVE - traces collected via rocprofv3")
        else:
            _profiler_backend = None
            logger.debug("No profiler available - markers will be no-ops")
    return _profiler_backend


@contextmanager
def _noop_profiler_range(name):
    """Zero-overhead no-op context manager when no profiler is available.

    Args:
        name: Range name (ignored)

    Yields:
        None
    """
    yield


class NvtxRange:
    """NVTX range context manager with graceful fallback.

    Creates a named profiling range using NVIDIA NVTX markers. If NVTX is not
    available, this becomes a no-op.

    Example:
        with NvtxRange("my_operation"):
            # code to profile
            pass

    Attributes:
        name (str): Name of the profiling range
    """

    def __init__(self, name: str):
        """Initialize NVTX range.

        Args:
            name: Name of the profiling range
        """
        self.name = name

    def __enter__(self):
        """Start the NVTX range.

        Returns:
            self: This context manager instance
        """
        if is_nvtx_available():
            try:
                import nvtx
                nvtx.push_range(self.name)
            except (ImportError, AttributeError):
                # Fallback to torch.cuda.nvtx
                import torch.cuda.nvtx
                torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the NVTX range.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            bool: False (does not suppress exceptions)
        """
        if is_nvtx_available():
            try:
                import nvtx
                nvtx.pop_range()
            except (ImportError, AttributeError):
                # Fallback to torch.cuda.nvtx
                import torch.cuda.nvtx
                torch.cuda.nvtx.range_pop()
        return False


class RoctxRange:
    """ROCTx range context manager with graceful fallback.

    Creates a named profiling range using AMD ROCTx markers. If ROCTx is not
    available, this becomes a no-op.

    Example:
        with RoctxRange("my_operation"):
            # code to profile
            pass

    Attributes:
        name (str): Name of the profiling range
        range_id (int or None): ROCTx range ID returned by rangeStart()
    """

    def __init__(self, name: str):
        """Initialize ROCTx range.

        Args:
            name: Name of the profiling range
        """
        self.name = name
        self.range_id = None

    def __enter__(self):
        """Start the ROCTx range.

        Returns:
            self: This context manager instance
        """
        if is_roctx_available():
            import roctx
            self.range_id = roctx.rangeStart(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End the ROCTx range.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)

        Returns:
            bool: False (does not suppress exceptions)
        """
        if self.range_id is not None:
            import roctx
            roctx.rangeStop(self.range_id)
        return False


# Global profiler range class/function - determined once at module import
# This avoids repeated availability checks during training
_profiler_range_impl = None


def _get_profiler_impl():
    """Get profiler range implementation (cached).

    Returns appropriate profiler based on auto-detection:
    - NvtxRange if NVTX is available
    - RoctxRange if ROCTx is available
    - _noop_profiler_range if neither is available

    Returns:
        callable: Either NvtxRange, RoctxRange, or _noop_profiler_range
    """
    global _profiler_range_impl
    if _profiler_range_impl is None:
        backend = detect_profiler_backend()
        if backend == "nvtx":
            _profiler_range_impl = NvtxRange
        elif backend == "roctx":
            _profiler_range_impl = RoctxRange
        else:
            _profiler_range_impl = _noop_profiler_range
    return _profiler_range_impl


def get_profiler_range():
    """Get profiler range context manager (NVTX/ROCTx/noop).

    Returns the appropriate profiler based on auto-detection:
    - NVTX on NVIDIA GPUs (Puhti V100, etc.)
    - ROCTx on AMD GPUs (LUMI MI250X, etc.)
    - No-op fallback when neither is available

    Profiling is controlled by external wrapper commands (nsys/rocprofv3),
    not this function.

    Returns:
        callable: Either NvtxRange, RoctxRange, or _noop_profiler_range

    Example:
        from mammoth.utils.profiling import get_profiler_range

        profiler_range = get_profiler_range()
        with profiler_range("operation"):
            # code here
            pass
    """
    return _get_profiler_impl()


def get_roctx_range():
    """Backward compatibility alias for get_profiler_range().

    DEPRECATED: Use get_profiler_range() instead for clarity.
    This function now returns NVTX, ROCTx, or no-op based on auto-detection.

    Returns:
        callable: Either NvtxRange, RoctxRange, or _noop_profiler_range

    Example:
        from mammoth.utils.profiling import get_roctx_range

        roctx_range = get_roctx_range()  # Works but misleading name
        with roctx_range("operation"):
            # code here
            pass
    """
    return get_profiler_range()
