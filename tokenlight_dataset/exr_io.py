from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def read_exr(path: str | Path) -> np.ndarray:
    path = str(path)
    errors: list[str] = []

    try:
        import imageio.v3 as iio

        img = iio.imread(path)
        img = np.asarray(img, dtype=np.float32)
        return _normalize_channels(img)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(f"imageio: {exc}")

    try:
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError("cv2.imread returned None")
        if img.ndim == 3 and img.shape[2] >= 3:
            img = img[:, :, :3][:, :, ::-1]
        img = np.asarray(img, dtype=np.float32)
        return _normalize_channels(img)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(f"opencv: {exc}")

    try:
        import Imath
        import OpenEXR

        exr = OpenEXR.InputFile(path)
        header = exr.header()
        dw = header["dataWindow"]
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = []
        for name in ("R", "G", "B"):
            data = exr.channel(name, pixel_type)
            channels.append(np.frombuffer(data, dtype=np.float32).reshape(height, width))
        return np.stack(channels, axis=-1)
    except Exception as exc:  # pragma: no cover - backend dependent
        errors.append(f"OpenEXR: {exc}")

    raise RuntimeError(f"Could not read EXR {path}. Tried: {'; '.join(errors)}")


def _normalize_channels(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.ndim != 3:
        raise ValueError(f"Expected HxWxC image, got shape {img.shape}")
    if img.shape[2] == 4:
        img = img[:, :, :3]
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    if img.shape[2] < 3:
        raise ValueError(f"Expected at least 3 channels, got shape {img.shape}")
    return img[:, :, :3].astype(np.float32, copy=False)
