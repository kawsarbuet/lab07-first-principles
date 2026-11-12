"""Hand-defined fully connected ReLU network for predicting bike counts."""

import torch
from torch import nn


class BikeNet(nn.Module):
    """At least two hidden layers; single unrestricted regression output."""

    def __init__(self, in_features: int, hidden: int = 64, layers: int = 2) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("At least two hidden layers are required")
        blocks: list[nn.Module] = []
        width = in_features
        for _ in range(layers):
            blocks.extend((nn.Linear(width, hidden), nn.ReLU()))
            width = hidden
        blocks.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
