# src/dataset.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

@dataclass
class SplitPaths:
    images: Path
    masks: Path

def _find_subdir(parent: Path, candidates: List[str]) -> Path:
    for name in candidates:
        p = parent / name
        if p.exists():
            return p
    # fallback search
    for p in parent.glob("*"):
        if p.is_dir() and p.name.lower() in [c.lower() for c in candidates]:
            return p
    raise FileNotFoundError(f"Could not find any of {candidates} under {parent}")

def resolve_split_paths(root: Path, split: str) -> SplitPaths:
    sdir = root / split
    if not sdir.exists():
        raise FileNotFoundError(f"Missing split folder: {sdir}")
    img_dir = _find_subdir(sdir, ["image", "images"])
    msk_dir = _find_subdir(sdir, ["mask", "masks"])
    return SplitPaths(images=img_dir, masks=msk_dir)

def list_pngs(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*.png") if p.is_file()])

def build_pairs(img_dir: Path, msk_dir: Path) -> List[Tuple[Path, Path]]:
    imgs = list_pngs(img_dir)
    msks = list_pngs(msk_dir)

    img_map: Dict[str, Path] = {p.stem: p for p in imgs}
    msk_map: Dict[str, Path] = {p.stem: p for p in msks}

    keys = sorted(set(img_map) & set(msk_map))
    pairs = [(img_map[k], msk_map[k]) for k in keys]

    if len(pairs) != len(imgs) or len(pairs) != len(msks):
        # this should not happen in your data (you already verified perfect pairing)
        missing_img = sorted(set(msk_map) - set(img_map))
        missing_msk = sorted(set(img_map) - set(msk_map))
        raise RuntimeError(
            f"Pairing mismatch. imgs={len(imgs)} masks={len(msks)} pairs={len(pairs)} "
            f"missing_img={len(missing_img)} missing_msk={len(missing_msk)}"
        )
    return pairs

def load_gray(path: Path) -> np.ndarray:
    # ensure grayscale uint8
    return np.array(Image.open(path).convert("L"))

class Kits2DSegDataset(Dataset):
    """
    Returns:
      image: float tensor [C,H,W]
      mask:  long tensor [H,W] with classes {0,1,2}
      meta: dict with paths / filename
    """
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable] = None):
        self.root = root
        self.split = split
        sp = resolve_split_paths(root, split)
        self.pairs = build_pairs(sp.images, sp.masks)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        ip, mp = self.pairs[idx]
        img = load_gray(ip)
        mask = np.array(Image.open(mp))  # keep raw values; transform will map

        if self.transform is not None:
            img_t, mask_t = self.transform(img, mask)
        else:
            # minimal fallback (no resize)
            img_t, mask_t = img, mask

        meta = {
            "image_path": str(ip),
            "mask_path": str(mp),
            "filename": ip.name,
            "split": self.split
        }
        return img_t, mask_t, meta