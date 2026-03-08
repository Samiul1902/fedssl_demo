# src/ssl/augmentations.py
from __future__ import annotations
import torchvision.transforms as T

def simclr_augment(size: int = 256):
    """
    Medical-safe SimCLR-ish augmentations (mild).
    Operates on PIL image.
    """
    return T.Compose([
        T.RandomResizedCrop(size=size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.ColorJitter(brightness=0.15, contrast=0.15),  # mild for CT PNGs
        T.ToTensor(),  # -> [0,1], shape [1,H,W] for grayscale
        T.Lambda(lambda x: x.repeat(3, 1, 1)),          # -> 3ch for ResNet
    ])