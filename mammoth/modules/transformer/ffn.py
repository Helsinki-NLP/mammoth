import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: float = 2.67, ff_dropout: float = 0.0):
        super().__init__()
        inner = int(dim * ff_mult)
        self.w1 = nn.Linear(dim, inner, bias=False)  # gate
        self.w2 = nn.Linear(inner, dim, bias=False)
        self.w3 = nn.Linear(dim, inner, bias=False)  # up
        self.dropout = nn.Dropout(ff_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
