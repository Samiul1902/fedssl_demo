# src/ssl/simclr.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimCLR(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        base = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # -> [B,512,1,1]
        self.proj = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x).flatten(1)        # [B,512]
        z = self.proj(h)                      # [B,proj_dim]
        z = F.normalize(z, dim=1)
        return z

def ntxent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2):
    """
    NT-Xent loss (SimCLR) — stable under AMP on CUDA/Windows.
    Forces similarity computation to float32 by disabling autocast.
    """
    B = z1.size(0)
    device_type = z1.device.type  # "cuda" or "cpu"

    # turn off autocast INSIDE the loss
    with torch.autocast(device_type=device_type, enabled=False):
        z = torch.cat([z1, z2], dim=0).float()        # [2B, D] float32
        sim = (z @ z.T) / float(temperature)          # [2B, 2B] float32

        # dtype-safe diagonal mask
        mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

        pos = torch.arange(B, device=sim.device)
        positives = torch.cat([pos + B, pos], dim=0)  # [2B]

        loss = F.cross_entropy(sim, positives)

    return loss