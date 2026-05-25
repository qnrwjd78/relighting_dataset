from __future__ import annotations

import numpy as np


def reinhard(x: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    y = np.maximum(x * exposure, 0.0)
    return y / (1.0 + y)


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)
