import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_mult: float = 4.0, ff_dropout: float = 0.0):
        super().__init__()
        inner = int(dim * ff_mult)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(inner, dim, bias=False)
        self.dropout = nn.Dropout(ff_dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.gelu(self.w1(x))))
