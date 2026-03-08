# src/models_resnet_unet.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # ensure shapes match
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNet18UNet(nn.Module):
    """
    Input:  [B,3,256,256]
    Output: [B,num_classes,256,256]
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)

        # Encoder blocks
        self.enc0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)  # [B,64,128,128]
        self.pool = self.backbone.maxpool                                                   # [B,64,64,64]
        self.enc1 = self.backbone.layer1                                                     # [B,64,64,64]
        self.enc2 = self.backbone.layer2                                                     # [B,128,32,32]
        self.enc3 = self.backbone.layer3                                                     # [B,256,16,16]
        self.enc4 = self.backbone.layer4                                                     # [B,512,8,8]

        # Decoder
        self.up1 = UpBlock(512, 256, 256)   # 8->16
        self.up2 = UpBlock(256, 128, 128)   # 16->32
        self.up3 = UpBlock(128, 64, 64)     # 32->64
        self.up4 = UpBlock(64, 64, 64)      # 64->128 (skip from enc0)
        self.up5 = nn.Sequential(           # 128->256 (no skip)
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            DoubleConv(32, 32),
        )

        self.outc = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.enc0(x)        # 64,128,128
        x1 = self.pool(x0)       # 64,64,64
        x1 = self.enc1(x1)       # 64,64,64
        x2 = self.enc2(x1)       # 128,32,32
        x3 = self.enc3(x2)       # 256,16,16
        x4 = self.enc4(x3)       # 512,8,8

        d = self.up1(x4, x3)     # 256,16,16
        d = self.up2(d, x2)      # 128,32,32
        d = self.up3(d, x1)      # 64,64,64
        d = self.up4(d, x0)      # 64,128,128
        d = self.up5(d)          # 32,256,256

        return self.outc(d)

    def load_simclr_encoder(self, simclr_encoder_ckpt: str | None):
        """
        Loads weights saved from: torch.save(simclr_model.encoder.state_dict()).
        This works by creating a Sequential view of this backbone's children[:-1]
        and loading the saved state dict into it (which updates backbone weights).
        """
        if simclr_encoder_ckpt is None:
            return

        state = torch.load(simclr_encoder_ckpt, map_location="cpu")
        seq = nn.Sequential(*list(self.backbone.children())[:-1])  # conv1..avgpool
        missing, unexpected = seq.load_state_dict(state, strict=False)

        # strict=False because our model doesn't use avgpool output directly
        return {"missing": missing, "unexpected": unexpected}