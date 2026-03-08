# src/train_seg.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.metrics import dice_iou_from_logits, soft_dice_loss


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    dice_weight: float = 0.5     # total_loss = CE + dice_weight * dice_loss(tumor)
    tumor_class: int = 2
    use_amp: bool = True


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device, cfg: TrainConfig, scaler=None) -> Dict:
    model.train()
    total_loss = 0.0
    n = 0

    for xb, yb, _meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_on = (cfg.use_amp and device.type == "cuda" and scaler is not None)
        with torch.autocast(device_type=device.type, enabled=amp_on):
            logits = model(xb)
            ce = F.cross_entropy(logits, yb)
            dloss = soft_dice_loss(logits, yb, class_idx=cfg.tumor_class)
            loss = ce + cfg.dice_weight * dloss

        if amp_on:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

    return {"train_loss": total_loss / max(n, 1)}


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Dict:
    model.eval()
    total_loss = 0.0
    n = 0

    # metrics
    tumor_dice_sum = 0.0
    tumor_iou_sum = 0.0
    kidney_dice_sum = 0.0
    kidney_iou_sum = 0.0
    batches = 0

    for xb, yb, _meta in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        ce = F.cross_entropy(logits, yb)
        dloss = soft_dice_loss(logits, yb, class_idx=cfg.tumor_class)
        loss = ce + cfg.dice_weight * dloss

        bs = xb.size(0)
        total_loss += loss.item() * bs
        n += bs

        # tumor class = 2
        t_dice, t_iou = dice_iou_from_logits(logits, yb, class_idx=2)
        tumor_dice_sum += t_dice
        tumor_iou_sum += t_iou

        # kidney class = 1
        k_dice, k_iou = dice_iou_from_logits(logits, yb, class_idx=1)
        kidney_dice_sum += k_dice
        kidney_iou_sum += k_iou

        batches += 1

    return {
        "val_loss": total_loss / max(n, 1),
        "tumor_dice": tumor_dice_sum / max(batches, 1),
        "tumor_iou": tumor_iou_sum / max(batches, 1),
        "kidney_dice": kidney_dice_sum / max(batches, 1),
        "kidney_iou": kidney_iou_sum / max(batches, 1),
    }


def save_checkpoint(model, optimizer, epoch: int, metrics: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
        },
        str(path),
    )


def save_model_weights(model, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))