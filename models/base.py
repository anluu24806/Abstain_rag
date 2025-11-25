from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import torch
from torch import nn


@dataclass
class RetrievedDoc:
    doc_id: str
    title: str
    text: str
    score: float
    rank: int


@dataclass
class RAGExample:
    qid: str
    question: str
    answers: List[str]
    retrieved: List[RetrievedDoc]
    scenario: Optional[str] = None
    teacher_models: Optional[List[str]] = None
    mean_delta: Optional[List[float]] = None     # len = k
    q_ensemble: Optional[List[float]] = None     # len = k+1 (base+docs)
    k: Optional[int] = None


class BaseGate(nn.Module):
    """Nhận features [B, E, D] → logits [B, E]."""
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaseSelector(nn.Module):
    """
    Selector/policy trên experts:
      input: batch dict + (tuỳ chọn) gate_logits [B, E]
      output: expert_logits [B, E], expert_probs [B, E]
    """
    def forward(
        self,
        batch: Dict[str, Any],
        gate_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
