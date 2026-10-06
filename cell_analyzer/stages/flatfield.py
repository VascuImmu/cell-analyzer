"""
Shared flatfield / background-correction utilities.

Loads a background field previously estimated by stages/background.py
and applies it to raw images.

Important: `stages/background.py`'s `save_path` saves the
*absolute-scale* smoothed background field (same intensity units as the
image), not the mean-normalized flatfield. Dividing by that directly would
rescale overall intensity, not just remove spatial non-uniformity. So here
we always re-normalize (divide by its own mean) before using it as a
correction factor.

The saved field is also typically downsampled (for memory/speed during
estimation). Since illumination bias is a smooth, low-frequency pattern by
construction, upsampling it back to the raw image's resolution via simple
bilinear interpolation is a safe assumption here.
"""

import numpy as np
from scipy.ndimage import zoom


def load_and_normalize_background(path):
    """
    Load a background_field .npy file and normalize it to mean == 1, so it
    can be used directly as a multiplicative/divisive flatfield correction.
    """
    background_field = np.load(path).astype(np.float32)
    mean_val = background_field.mean()
    if mean_val <= 0:
        raise ValueError(f"Background field at {path} has non-positive mean ({mean_val}); cannot normalize.")
    return (background_field / mean_val).astype(np.float32)


def apply_flatfield_correction(image, flatfield):
    """
    Divide `image` by `flatfield` (mean-normalized), upsampling `flatfield`
    to match image's last two (Y, X) dimensions first if needed.

    Works for 2D (Y, X) or 3D (Z, Y, X) images -- a resized 2D field
    broadcasts over any leading Z dimension automatically.

    Parameters
    ----------
    image : np.ndarray
        Raw image, last two dims are (Y, X).
    flatfield : np.ndarray (2D)
        Mean-normalized flatfield, e.g. from load_and_normalize_background.

    Returns
    -------
    corrected : np.ndarray (float32)
    """
    target_shape = image.shape[-2:]
    if flatfield.shape != target_shape:
        zoom_factors = (target_shape[0] / flatfield.shape[0],
                         target_shape[1] / flatfield.shape[1])
        flatfield = zoom(flatfield, zoom_factors, order=1)

    # guard against near-zero values at the edges of interpolation, which
    # would otherwise blow up the correction
    flatfield = np.clip(flatfield, 1e-3, None)

    return (image.astype(np.float32) / flatfield).astype(np.float32)
