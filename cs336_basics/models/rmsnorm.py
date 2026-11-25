import torch
from torch import nn
from einops import einsum

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.g = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype: torch.dtype = x.dtype
        x = x.to(torch.float32)
        rms_x: torch.Tensor = ((x ** 2).mean(dim=-1, keepdim=True) + self.eps) ** 0.5
        result = einsum(x / rms_x, self.g, "batch_size seq d_model, d_model -> batch_size seq d_model")
        return result.to(in_dtype)