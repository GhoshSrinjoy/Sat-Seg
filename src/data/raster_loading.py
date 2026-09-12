"""Windowed raster I/O; band order is explicit and georeferencing is preserved."""

from pathlib import Path
from typing import Sequence

import rasterio
from rasterio.windows import Window


def read_raster(path: str | Path, bands: Sequence[int] | None = None,
                window: Window | None = None):
    """Return a masked (bands, rows, columns) array and its updated raster profile.

    Band indexes are one-based. Use windows for large scenes to bound memory.
    Values retain their original radiometric units; no RGB assumption is made.
    """
    with rasterio.open(path) as dataset:
        indexes = list(bands) if bands is not None else list(dataset.indexes)
        if not indexes or any(index not in dataset.indexes for index in indexes):
            raise ValueError(f"Invalid bands {indexes}; available: {dataset.indexes}")
        if window is not None:
            if any(value != int(value) for value in window.flatten()):
                raise ValueError("Use integer pixel offsets and dimensions for read windows.")
            # Clip explicitly so the returned transform describes the pixels
            # actually read, including when a requested tile crosses an edge.
            window = window.intersection(Window(0, 0, dataset.width, dataset.height))
        array = dataset.read(indexes=indexes, window=window, masked=True)
        profile = dataset.profile.copy()
        profile.update(
            count=array.shape[0], height=array.shape[1], width=array.shape[2],
            transform=dataset.window_transform(window) if window is not None else dataset.transform,
        )
    return array, profile
