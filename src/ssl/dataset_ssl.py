# src/ssl/dataset_ssl.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, List
from PIL import Image
from torch.utils.data import Dataset

from src.dataset import resolve_split_paths, build_pairs

class Kits2DSSLDataset(Dataset):
    """
    SSL dataset: returns two augmented views of the same image.
    Uses train split images (mask ignored).
    """
    def __init__(self, root: Path, split: str, transform: Callable):
        self.root = root
        self.split = split
        self.transform = transform

        sp = resolve_split_paths(root, split)
        pairs = build_pairs(sp.images, sp.masks)
        self.images: List[Path] = [ip for ip, _ in pairs]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        ip = self.images[idx]
        img = Image.open(ip).convert("L")  # PIL grayscale

        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2