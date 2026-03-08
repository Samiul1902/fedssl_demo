# src/transforms.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2
import torch

@dataclass
class TransformConfig:
    out_size: int = 256          # start with 256 for RTX 3060 Ti; can go 320/384 later
    to_3ch: bool = True          # repeat grayscale -> 3ch to support ResNet/ImageNet later
    normalize_01: bool = True    # scale image to [0,1]

def pad_to_square(img: np.ndarray,
                  mask: Optional[np.ndarray] = None,
                  pad_value: int = 0,
                  mask_pad_value: int = 0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Pads (H,W) or (H,W,C) image to square by symmetric padding.
    Mask (H,W) padded with mask_pad_value.
    """
    h, w = img.shape[:2]
    size = max(h, w)
    pad_h = size - h
    pad_w = size - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    img_pad = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 borderType=cv2.BORDER_CONSTANT, value=pad_value)

    if mask is None:
        return img_pad, None

    mask_pad = cv2.copyMakeBorder(mask, top, bottom, left, right,
                                  borderType=cv2.BORDER_CONSTANT, value=mask_pad_value)
    return img_pad, mask_pad

def resize_pair(img: np.ndarray,
                mask: Optional[np.ndarray],
                out_size: int) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    img_r = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    if mask is None:
        return img_r, None
    mask_r = cv2.resize(mask, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
    return img_r, mask_r

def to_tensor_image(img: np.ndarray, to_3ch: bool, normalize_01: bool) -> torch.Tensor:
    # img is HxW grayscale uint8
    if normalize_01:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.float32)

    if to_3ch:
        img = np.repeat(img[..., None], 3, axis=2)  # HWC, 3ch
    else:
        img = img[..., None]  # HWC, 1ch

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)

def map_mask_values(mask: np.ndarray) -> np.ndarray:
    """
    Map original mask values [0,127,255] -> class ids [0,1,2]
    0   = background
    127 = kidney
    255 = tumor
    """
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask == 127] = 1
    out[mask == 255] = 2
    return out

class SegTransform:
    def __init__(self, cfg: TransformConfig):
        self.cfg = cfg

    def __call__(self, img: np.ndarray, mask: np.ndarray):
        img, mask = pad_to_square(img, mask, pad_value=0, mask_pad_value=0)
        img, mask = resize_pair(img, mask, self.cfg.out_size)

        img_t = to_tensor_image(img, to_3ch=self.cfg.to_3ch, normalize_01=self.cfg.normalize_01)
        mask_c = map_mask_values(mask)
        mask_t = torch.from_numpy(mask_c.astype(np.int64))  # HxW, int64 for CE loss

        return img_t, mask_t