# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Abraham Joseph Okayli Masaryk
"""2D generalised recurrence plots, networks, joint/cross RP, sliding windows.

The heavy computation is done by the C++/OpenMP ``_core`` extension. This
module re-exports the classes/functions and adds a few Python-level helpers
(adaptive thresholding, raster I/O via rasterio, quick plot helpers).
"""

from __future__ import annotations

import numpy as np

from ._core import (
    GeneralizedRecurrencePlot,
    SpatialRQA,
    SpatialRecurrenceNetwork,
    JointGeneralizedRecurrencePlot,
    CrossGeneralizedRecurrencePlot,
    SlidingWindowResult,
    sliding_window_fixed_eps,
    sliding_window_target_rr,
)

from . import ebsd

__all__ = [
    "GeneralizedRecurrencePlot",
    "SpatialRQA",
    "SpatialRecurrenceNetwork",
    "JointGeneralizedRecurrencePlot",
    "CrossGeneralizedRecurrencePlot",
    "SlidingWindowResult",
    "sliding_window_fixed_eps",
    "sliding_window_target_rr",
    "adaptive_threshold",
    "load_raster_field",
    "ebsd",
]


def adaptive_threshold(
    field: np.ndarray,
    target_rr: float,
    *,
    n_samples: int = 200_000,
    seed: int = 42,
) -> float:
    """Pick eps such that R has approximately target_rr recurrence rate.

    Samples pixel pairs, computes absolute value differences, and returns
    the target_rr-quantile. Fast — does not touch the 4D tensor.
    """
    if field.ndim != 2:
        raise ValueError("field must be 2-D")
    if not 0.0 < target_rr < 1.0:
        raise ValueError("target_rr must be in (0, 1)")
    rng = np.random.default_rng(seed)
    nx, ny = field.shape
    i1 = rng.integers(0, nx, n_samples)
    i2 = rng.integers(0, ny, n_samples)
    j1 = rng.integers(0, nx, n_samples)
    j2 = rng.integers(0, ny, n_samples)
    diffs = np.abs(field[i1, i2] - field[j1, j2])
    return float(np.quantile(diffs, target_rr))


def load_raster_field(
    path: str,
    *,
    band: int = 1,
    downsample: int = 1,
    dtype=np.float64,
) -> np.ndarray:
    """Read a single raster band as a 2D float64 numpy array.

    Requires rasterio (``pip install 'recurrence2d[raster]'``). Handles
    GeoTIFF, ENVI, and anything GDAL can read. ``downsample`` > 1 takes
    every Nth row/col (naive subsampling — use dedicated resampling if
    you care about anti-aliasing).
    """
    try:
        import rasterio
    except ImportError as e:
        raise ImportError(
            "load_raster_field requires rasterio. "
            "Install with: pip install 'recurrence2d[raster]'"
        ) from e
    with rasterio.open(path) as ds:
        arr = ds.read(band, out_dtype="float64")
    if downsample > 1:
        arr = arr[::downsample, ::downsample]
    return np.ascontiguousarray(arr, dtype=dtype)
