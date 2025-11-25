from __future__ import annotations
from typing import Dict, Any
import torch
from .base import BaseSelector


class NaiveRAGSelector(BaseSelector):
    """
    Luôn dùng tất cả docs, không abstain:
      experts = [base, doc1..dock]
      prob(base)=~0, docs uniform.
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, batch: Dict[str, Any], gate_logits=None):
        B = batch["retriever_scores"].shape[0]
        E = self.k + 1
        logits = torch.zeros(B, E)
        logits[:, 0] = -1e9    # base ~ 0
        logits[:, 1:] = 0.0    # docs
        probs = torch.softmax(logits, dim=-1)
        return {"expert_logits": logits, "expert_probs": probs}
