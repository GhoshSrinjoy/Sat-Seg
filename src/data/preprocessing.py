"""Explicit display conversion for RGB imagery; does not alter scientific rasters."""

import numpy as np
from PIL import Image


def rgb_preview(array, lower_percentile: float = 2, upper_percentile: float = 98) -> Image.Image:
    """Convert three caller-selected bands to a uint8 PIL RGB preview.

    Masked/no-data/nonfinite pixels are excluded from the stretch and displayed
    black. This per-band stretch is for inspection, not a training normalization.
    """
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError("Expected exactly three bands in (3, height, width) order.")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")
    values = np.ma.asarray(array, dtype=np.float32)
    valid_pixel = ~np.any(np.ma.getmaskarray(values), axis=0)
    valid_pixel &= np.all(np.isfinite(values.data), axis=0)
    output = np.zeros(values.shape, dtype=np.uint8)
    for index, band in enumerate(values.data):
        if not valid_pixel.any():
            continue
        low, high = np.percentile(band[valid_pixel], [lower_percentile, upper_percentile])
        if high > low:
            scaled = np.clip((band[valid_pixel] - low) / (high - low), 0, 1)
            output[index][valid_pixel] = (scaled * 255).astype(np.uint8)
    return Image.fromarray(output.transpose(1, 2, 0))
