import torch
from torch import nn
from .utils import RandomDistribution, init_tensor_with_random_distribution

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings: int, 
        embedding_dim: int, 
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        random_distribution: RandomDistribution = RandomDistribution(0, 1, -3, 3)
        self.E = nn.Parameter(init_tensor_with_random_distribution(num_embeddings, embedding_dim, random_distribution, device, dtype))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.E[token_ids]