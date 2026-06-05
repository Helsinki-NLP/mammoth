import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Literal


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        ff_mult: float = 2.67,
        ff_dropout: float = 0.0,
        activation: Literal["swiglu", "gelu"] = "swiglu"
    ):
        super().__init__()
        inner = int(dim * ff_mult)
        self.activation = activation

        if activation == "swiglu":
            # SWiGLU uses three matrices: gate, up, and down
            self.w1 = nn.Linear(dim, inner, bias=False)  # gate
            self.w2 = nn.Linear(inner, dim, bias=False)  # down
            self.w3 = nn.Linear(dim, inner, bias=False)  # up
        else:  # gelu
            # Standard GELU uses two matrices: up and down
            self.w1 = nn.Linear(dim, inner, bias=False)  # up
            self.w2 = nn.Linear(inner, dim, bias=False)  # down
            self.w3 = None  # Not used for GELU

        self.dropout = nn.Dropout(ff_dropout)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            # SWiGLU: SiLU(gate) * up
            return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
        else:  # gelu
            # GELU: standard feed-forward
            return self.dropout(self.w2(F.gelu(self.w1(x))))
