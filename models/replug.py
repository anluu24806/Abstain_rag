from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from .base import BaseSelector


class ReplugSelector(BaseSelector):
    """
    REPLUG-style: mixture of doc experts, no base.
      experts = [doc1..dock]
    """
    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(
        self,
        batch: Dict[str, Any],
        gate_logits: Optional[torch.Tensor] = None,
    ):
        B = batch["retriever_scores"].shape[0]
        if gate_logits is None:
            logits = batch["retriever_scores"][:, : self.k]
        else:
            assert gate_logits.shape[1] == self.k
            logits = gate_logits
        probs = torch.softmax(logits, dim=-1)
        return {"expert_logits": logits, "expert_probs": probs}
