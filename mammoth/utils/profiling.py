"""
GPU profiling utilities controlled via MAMMOTH_PROFILER_BACKEND.

Profiling backend is selected exclusively via the MAMMOTH_PROFILER_BACKEND
environment variable:
  - "roctx"  → AMD ROCTx markers (LUMI MI250X, etc.)
  - "nvtx"   → NVIDIA NVTX markers (Puhti V100, etc.)
  - not set  → no profiling (no-op markers)

Usage:
    from mammoth.utils.profiling import get_profiler_range

    profiler_range = get_profiler_range()
    with profiler_range("operation_name"):
        # Your code here
        pass

Profiling Control:
    # AMD profiling with rocprofv3:
    MAMMOTH_PROFILER_BACKEND=roctx rocprofv3 --marker-trace --output-dir ./profiling -- python train.py -config config.yaml

    # NVIDIA profiling with nsys:
    MAMMOTH_PROFILER_BACKEND=nvtx nsys profile -t nvtx,cuda --output profiling/trace python train.py -config config.yaml

See docs/GPU_PROFILING.md for detailed platform-specific instructions.
"""

from contextlib import contextmanager
import logging
import os

logger = logging.getLogger(__name__)

_profiler_backend = None
_profiler_range_impl = None


def detect_profiler_backend():
    """Read MAMMOTH_PROFILER_BACKEND and return the selected backend.

    Returns:
        str: "nvtx", "roctx", or None (no profiling)
    """
    global _profiler_backend
    if _profiler_backend is None:
        value = os.environ.get("MAMMOTH_PROFILER_BACKEND", "").lower()
        if value == "roctx":
            _profiler_backend = "roctx"
            logger.info("ROCTx markers ACTIVE - traces collected via rocprofv3")
        elif value == "nvtx":
            _profiler_backend = "nvtx"
            logger.info("NVTX markers ACTIVE - traces collected via nsys/nvprof")
        else:
            _profiler_backend = "none"
            logger.debug("MAMMOTH_PROFILER_BACKEND not set - profiling disabled")
    return _profiler_backend


@contextmanager
def _noop_profiler_range(name):
    yield


class NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        try:
            import nvtx
            nvtx.push_range(self.name)
        except ImportError:
            import torch.cuda.nvtx
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import nvtx
            nvtx.pop_range()
        except ImportError:
            import torch.cuda.nvtx
            torch.cuda.nvtx.range_pop()
        return False


class RoctxRange:
    def __init__(self, name: str):
        self.name = name
        self.range_id = None

    def __enter__(self):
        import roctx
        self.range_id = roctx.rangeStart(self.name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import roctx
        roctx.rangeStop(self.range_id)
        return False


def get_profiler_range():
    """Get profiler range context manager based on MAMMOTH_PROFILER_BACKEND.

    Returns:
        callable: NvtxRange, RoctxRange, or _noop_profiler_range

    Example:
        profiler_range = get_profiler_range()
        with profiler_range("operation"):
            pass
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
