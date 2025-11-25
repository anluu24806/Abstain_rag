from __future__ import annotations
import torch
from torch import nn
from .base import BaseGate


class WhiteboxGate(BaseGate):
    """
    Gate MLP:
      input:  features [B, E, D]
      output: logits   [B, E]
    """
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        layers = []
        in_dim = feature_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # 1 logit / expert
        self.mlp = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, E, D = features.shape
        x = features.view(B * E, D)
        logits = self.mlp(x)        # [B*E, 1]
        logits = logits.view(B, E)  # [B, E]
        return logits
