"""
GPU delegate patches for LiteRT conversion.

Annotates nn.RMSNorm and F.scaled_dot_product_attention with
StableHLOCompositeBuilder markers so the LiteRT GPU delegate can fuse them.
Both helpers are no-ops when litert_torch is unavailable (macOS / export-only).
"""

from __future__ import annotations

import contextlib
from typing import Generator

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from litert_torch.backend.composite import StableHLOCompositeBuilder
    _COMPOSITE_AVAILABLE = True
except ImportError:
    _COMPOSITE_AVAILABLE = False


class MammothRMSNorm(nn.Module):
    """Drop-in replacement for nn.RMSNorm annotated for GPU delegate fusion."""

    def __init__(self, eps: float, weight: torch.Tensor) -> None:
        super().__init__()
        self._eps = eps
        self.weight = nn.Parameter(weight.detach())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        composite = StableHLOCompositeBuilder("odml.rms_norm", {"epsilon": self._eps})
        x, w = composite.mark_inputs(x, self.weight)
        y = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self._eps) * w
        y = composite.mark_outputs(y)
        return y


def replace_rms_norms(module: nn.Module) -> None:
    """Recursively replace all nn.RMSNorm submodules with MammothRMSNorm.

    No-op when litert_torch (StableHLOCompositeBuilder) is unavailable.
    """
    if not _COMPOSITE_AVAILABLE:
        return
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, MammothRMSNorm(child.eps, child.weight))
        else:
            replace_rms_norms(child)


# ── SDPA monkey-patch ─────────────────────────────────────────────────────────

_original_sdpa = F.scaled_dot_product_attention


def _patched_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    **kwargs,
) -> torch.Tensor:
    effective_scale = scale if scale is not None else 1.0 / (query.shape[-1] ** 0.5)
    composite = StableHLOCompositeBuilder(
        "odml.scaled_dot_product_attention", {"scale": effective_scale}
    )
    if attn_mask is None:
        query, key, value = composite.mark_inputs(query, key, value)
    else:
        query, key, value, attn_mask = composite.mark_inputs(query, key, value, attn_mask)
    out = _original_sdpa(
        query, key, value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        **kwargs,
    )
    out = composite.mark_outputs(out)
    return out


@contextlib.contextmanager
def sdpa_patch() -> Generator[None, None, None]:
    """Context manager that installs the composite-annotated SDPA during torch.export.

    No-op when litert_torch (StableHLOCompositeBuilder) is unavailable.
    """
    if not _COMPOSITE_AVAILABLE:
        yield
        return
    F.scaled_dot_product_attention = _patched_sdpa
    try:
        yield
    finally:
        F.scaled_dot_product_attention = _original_sdpa
