from __future__ import annotations
from typing import Dict, Any
import torch
from .base import BaseSelector


class ClosedBookSelector(BaseSelector):
    """Chỉ base expert (no docs)."""
    def __init__(self):
        super().__init__()

    def forward(self, batch: Dict[str, Any], gate_logits=None):
        B = batch["retriever_scores"].shape[0] if "retriever_scores" in batch else 1
        expert_logits = torch.zeros(B, 1)
        expert_probs = torch.ones(B, 1)
        return {"expert_logits": expert_logits, "expert_probs": expert_probs}
