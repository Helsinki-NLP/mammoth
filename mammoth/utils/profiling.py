"""
GPU profiling utilities controlled via MAMMOTH_PLATFORM.

The platform is selected via the MAMMOTH_PLATFORM environment variable:
  - "lumi"   → CPU-GPU binding only (no profiler markers); safe when roctx bindings are absent
  - "roctx"  → AMD ROCTx markers + CPU-GPU binding (LUMI MI250X with roctx installed)
  - "nvidia" → NVIDIA NVTX markers (Puhti V100, etc.)
  - not set  → no-op markers, no CPU-GPU binding

Markers are always emitted but only collected when launched under the profiler.

Usage:
    from mammoth.utils.profiling import get_profiler_range

    profiler_range = get_profiler_range()
    with profiler_range("operation_name"):
        # Your code here
        pass

Profiling Control:
    # AMD profiling with rocprofv3 (requires roctx bindings):
    MAMMOTH_PLATFORM=roctx rocprofv3 --marker-trace --output-dir ./profiling -- python train.py -config config.yaml

    # LUMI without roctx (CPU-GPU binding only):
    MAMMOTH_PLATFORM=lumi python train.py -config config.yaml

    # NVIDIA profiling with nsys:
    MAMMOTH_PLATFORM=nvidia nsys profile -t nvtx,cuda --output profiling/trace python train.py -config config.yaml
"""

from contextlib import contextmanager
import logging
import os

logger = logging.getLogger(__name__)

_profiler_backend = None
_profiler_range_impl = None


def detect_profiler_backend():
    """Derive profiler backend from MAMMOTH_PLATFORM.

    Returns:
        str: "nvtx", "roctx", or "none"
    """
    global _profiler_backend
    if _profiler_backend is None:
        platform = os.environ.get("MAMMOTH_PLATFORM", "").lower()
        if platform == "roctx":
            _profiler_backend = "roctx"
            logger.info("ROCTx markers ACTIVE (MAMMOTH_PLATFORM=roctx) - collect via rocprofv3")
        elif platform == "nvidia":
            _profiler_backend = "nvtx"
            logger.info("NVTX markers ACTIVE (MAMMOTH_PLATFORM=nvidia) - collect via nsys/nvprof")
        else:
            _profiler_backend = "none"
            if platform == "lumi":
                logger.debug("MAMMOTH_PLATFORM=lumi - CPU-GPU binding enabled, profiling markers disabled")
            else:
                logger.debug("MAMMOTH_PLATFORM not set - profiling markers disabled")
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
