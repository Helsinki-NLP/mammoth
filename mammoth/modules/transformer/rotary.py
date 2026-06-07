import torch
import torch.nn as nn
from torch import Tensor


class RotaryEmbedding(nn.Module):
    def __init__(self, dim_head: int, base: int = 10000):
        super().__init__()
        self.dim_head = dim_head
        self.base = base
        self._seq_len_cached = 0
        self._cos_cached: Tensor | None = None
        self._sin_cached: Tensor | None = None

    def _build_cache(self, seq_len: int, device: torch.device) -> None:
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim_head, 2, device=device).float() / self.dim_head))
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cached = emb.cos()
        self._sin_cached = emb.sin()
        self._seq_len_cached = seq_len

    def forward(self, seq_len: int, device: torch.device, offset: int = 0) -> tuple[Tensor, Tensor]:
        total = offset + seq_len
        if total > self._seq_len_cached:
            self._build_cache(total, device)
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached[offset:offset + seq_len], self._sin_cached[offset:offset + seq_len]


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    # q, k: (batch, heads, seq, dim_head)
    # cos, sin: (seq, dim_head)
    cos = cos.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)  # (1, 1, seq, dim_head)
    sin = sin.unsqueeze(0).unsqueeze(0).to(dtype=q.dtype)
    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
