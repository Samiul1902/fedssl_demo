# src/metrics.py
from __future__ import annotations
import torch


@torch.no_grad()
def dice_iou_from_logits(logits: torch.Tensor, target: torch.Tensor, class_idx: int, eps: float = 1e-6):
    """
    logits: [B,C,H,W]
    target: [B,H,W] with class ids
    Computes Dice and IoU for a single class using argmax prediction.
    """
    pred = torch.argmax(logits, dim=1)  # [B,H,W]

    pred_c = (pred == class_idx).float()
    targ_c = (target == class_idx).float()

    inter = (pred_c * targ_c).sum(dim=(1, 2))
    union = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2))

    dice = (2.0 * inter + eps) / (union + eps)

    # IoU
    iou_den = (pred_c + targ_c - pred_c * targ_c).sum(dim=(1, 2))
    iou = (inter + eps) / (iou_den + eps)

    return dice.mean().item(), iou.mean().item()


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, class_idx: int, eps: float = 1e-6):
    """
    Soft Dice loss for one class (good for imbalance).
    """
    probs = torch.softmax(logits, dim=1)  # [B,C,H,W]
    p = probs[:, class_idx, :, :]         # [B,H,W]
    t = (target == class_idx).float()     # [B,H,W]

    inter = (p * t).sum(dim=(1, 2))
    union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))

    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()