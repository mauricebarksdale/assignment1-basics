import torch
from torch import nn
from .linear import Linear
from . import functional as F

class PositionWiseFeedForward(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.d_ff: int = d_ff
        self.w1 = Linear(self.d_model, self.d_ff, device, dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device, dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate: torch.Tensor = self.w1(x)
        val: torch.Tensor = self.w3(x)
        return self.w2(F.silu(gate) * val)