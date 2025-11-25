from __future__ import annotations
from typing import Tuple
import torch
from torch import nn


class RFHead(nn.Module):
    """
    ReFrag-style head:
      eq: [B, D], ed: [B, k, D]
      → doc_logits: [B, k], base_logit: [B]
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.base_bias = nn.Parameter(torch.zeros(1))

    def forward(self, eq: torch.Tensor, ed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, k, D = ed.shape
        eq_expanded = eq.unsqueeze(1).expand(-1, k, -1)   # [B, k, D]
        prod = eq_expanded * ed                           # [B, k, D]
        feat = torch.cat([eq_expanded, ed, prod], dim=-1) # [B, k, 3D]
        logits = self.proj(feat.view(B * k, 3 * D))       # [B*k, 1]
        logits = logits.view(B, k)
        base_logit = self.base_bias.expand(B)
        return logits, base_logit
