# src/federated/fedavg.py
from __future__ import annotations
from typing import Dict, List
import torch

def get_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def set_state_dict(model, state: Dict[str, torch.Tensor]):
    model.load_state_dict(state, strict=True)

def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[int]) -> Dict[str, torch.Tensor]:
    assert len(states) == len(weights) and len(states) > 0
    total = float(sum(weights))
    out = {}
    for k in states[0].keys():
        out[k] = sum((w / total) * states[i][k] for i, w in enumerate(weights))
    return out