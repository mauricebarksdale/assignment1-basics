import torch
from torch import nn
from dataclasses import dataclass

@dataclass
class RandomDistribution:
    mean: float | int
    std: float | int
    lower_bound: float | int
    upper_bound: float | int

def init_tensor_with_random_distribution(
    x_dim: int,
    y_dim: int,
    random_distribution: RandomDistribution,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.empty((x_dim, y_dim), device=device, dtype=dtype)
    return nn.init.trunc_normal_(tensor, random_distribution.mean, random_distribution.std, random_distribution.lower_bound, random_distribution.upper_bound)