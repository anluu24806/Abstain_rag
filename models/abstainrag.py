from __future__ import annotations
from typing import Dict, Any, Optional
import torch
from torch import nn
from .base import BaseSelector


class AbstainRAGSelector(BaseSelector):
    """
    Experts = [base, doc1..dock, (optional) abstain]
    E = k+2 nếu có abstain, k+1 nếu không.
    """
    def __init__(self, k: int, has_abstain: bool = True):
        super().__init__()
        self.k = k
        self.has_abstain = has_abstain

    def forward(
        self,
        batch: Dict[str, Any],
        gate_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = batch["retriever_scores"].shape[0]
        k = self.k
        E = k + 2 if self.has_abstain else k + 1

        if gate_logits is None:
            if "q_ensemble" in batch:
                q = batch["q_ensemble"]  # [B, k+1] = [base, docs...]
                if self.has_abstain:
                    q_pad = torch.zeros(B, E, dtype=q.dtype, device=q.device)
                    q_pad[:, : q.shape[1]] = q
                    q_pad[:, -1] = 1e-9   # abstain ~ 0
                    expert_logits = torch.log(q_pad + 1e-9)
                else:
                    expert_logits = torch.log(q + 1e-9)
            else:
                retriever_scores = batch["retriever_scores"][:, :k]  # [B, k]
                base_logit = torch.zeros(B, 1, device=retriever_scores.device)
                if self.has_abstain:
                    abstain_logit = torch.full(
                        (B, 1), -1e9, device=retriever_scores.device
                    )
                    expert_logits = torch.cat(
                        [base_logit, retriever_scores, abstain_logit], dim=-1
                    )
                else:
                    expert_logits = torch.cat(
                        [base_logit, retriever_scores], dim=-1
                    )
        else:
            assert gate_logits.shape[1] == E
            expert_logits = gate_logits

        expert_probs = torch.softmax(expert_logits, dim=-1)
        return {
            "expert_logits": expert_logits,
            "expert_probs": expert_probs,
        }
