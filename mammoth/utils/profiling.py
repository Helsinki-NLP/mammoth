"""
ROCTx profiling utilities for AMD GPU profiling on LUMI.

This module provides ROCTx markers throughout the training pipeline. Profiling
is controlled by the rocprofv3 wrapper - markers are always active in code but
traces are only collected when running under rocprofv3.

Usage:
    from mammoth.utils.profiling import RoctxRange

    # In your code:
    with RoctxRange("operation_name"):
        # Your code here
        pass

Prerequisites:
    - ROCm 6.0+ installed on the system
    - PYTHONPATH includes ROCTx Python bindings:
      export PYTHONPATH=/opt/rocm/lib/python3.10/site-packages:$PYTHONPATH

Profiling Control:
    # Without rocprofv3 wrapper (no traces collected, ~0.1% overhead):
    python train.py -config config.yaml

    # With rocprofv3 wrapper (traces collected to CSV):
    rocprofv3 --marker-trace --output-dir ./profiling -- python train.py -config config.yaml
"""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

# Global flag for runtime check (cached to avoid repeated imports)
_roctx_available = None


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
            logger.info("ROCTx profiling available")
        except ImportError:
            _roctx_available = False
            logger.warning(
                "ROCTx not available - profiling will be disabled. "
                "Set PYTHONPATH=/opt/rocm/lib/python3.10/site-packages if needed."
            )
    return _roctx_available


@contextmanager
def _noop_roctx_range(name):
    """Zero-overhead no-op context manager when profiling is disabled.

    Args:
        name: Range name (ignored)

    Yields:
        None
    """
    yield


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


# Global ROCTx range class/function - determined once at module import
# This avoids repeated availability checks during training
_roctx_range_impl = None


def _get_roctx_impl():
    """Get ROCTx range implementation (cached)."""
    global _roctx_range_impl
    if _roctx_range_impl is None:
        if is_roctx_available():
            logger.info("ROCTx markers ACTIVE - traces will be collected if running under rocprofv3")
            _roctx_range_impl = RoctxRange
        else:
            logger.warning(
                "ROCTx not available - markers will be no-ops. "
                "Set PYTHONPATH=/opt/rocm/lib/python3.12/site-packages if needed."
            )
            _roctx_range_impl = _noop_roctx_range
    return _roctx_range_impl


# Convenience function for backward compatibility
def get_roctx_range():
    """Get ROCTx range context manager.

    Returns RoctxRange if available, otherwise noop context manager.
    Profiling is controlled by rocprofv3 wrapper, not this function.

    Returns:
        callable: Either RoctxRange class or _noop_roctx_range function

    Example:
        from mammoth.utils.profiling import get_roctx_range

        roctx_range = get_roctx_range()
        with roctx_range("operation"):
            # code here
            pass
    """
    return _get_roctx_impl()
