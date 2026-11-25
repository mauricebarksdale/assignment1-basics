import torch
from torch import nn
from einops import einsum
from .utils import RandomDistribution, init_tensor_with_random_distribution

class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        
        self.in_features: int = in_features
        self.out_features: int = out_features

        variance: float = 2.0 / (in_features + out_features)
        std: float = variance ** 0.5
        random_distribution = RandomDistribution(0, std, -3 * std, 3 * std)
        self.weight: nn.Parameter = nn.Parameter(init_tensor_with_random_distribution(out_features, in_features, random_distribution, device, dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")