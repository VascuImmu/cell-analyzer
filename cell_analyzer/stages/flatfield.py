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


# -------------------------------------------------
# Corrections with a description file (background mode 'image')
# -------------------------------------------------
# A correction is a field <name>.npy, optionally with <name>.json next to it:
#   {"operation": "divide" | "subtract",       divide = flatfield, subtract = blank sample
#    "offset": 100.0 | null,                   constant camera offset (divide only)
#    "dark": "<name>_dark.npy" | null}         dark image, same grid as the field (divide only)
# Without the .json file a field is a flatfield without offset -- exactly the behaviour of the
# fields made by the background stage and of existing .npy files.
import json
from pathlib import Path


def _resize(field, shape):
    if field.shape != tuple(shape):
        field = zoom(field, (shape[0] / field.shape[0], shape[1] / field.shape[1]), order=1)
    return field


def load_correction(path):
    """-> dict(operation, field, offset) ready for correct_image()."""
    path = Path(path)
    meta = {}
    side = path.with_suffix(".json")
    if side.exists():
        with open(side) as f:
            meta = json.load(f)
    op = meta.get("operation", "divide")
    field = np.load(path).astype(np.float32)
    offset = meta.get("offset")
    if meta.get("dark"):
        dark_path = Path(meta["dark"])
        if not dark_path.is_absolute():
            dark_path = path.parent / dark_path
        offset = np.load(dark_path).astype(np.float32)
    if op == "divide":
        mean_val = float(field.mean())
        if mean_val <= 0:
            raise ValueError(f"Flatfield {path} has non-positive mean ({mean_val}); cannot normalize.")
        field = field / mean_val
    elif op != "subtract":
        raise ValueError(f"{side}: unknown operation '{op}'")
    return dict(operation=op, field=field, offset=offset, path=str(path))


def correct_image(image, corr):
    """Apply a correction from load_correction() to a 2D (Y, X) or 3D (Z, Y, X) image."""
    img = image.astype(np.float32)
    shape = img.shape[-2:]
    field = _resize(corr["field"], shape)
    if corr["operation"] == "subtract":
        return (img - field).astype(np.float32)
    off = corr.get("offset")
    if off is not None:
        img = img - (_resize(off, shape) if np.ndim(off) == 2 else np.float32(off))
    return (img / np.clip(field, 1e-3, None)).astype(np.float32)


def save_correction(path, field, operation="divide", offset=None, dark=None, info=None):
    """Write <path>.npy (+ <path>.json). `dark` is an array saved as <stem>_dark.npy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(field, dtype=np.float32))
    meta = dict(operation=operation, offset=None if offset in (None, 0, 0.0) else float(offset), dark=None)
    if dark is not None:
        dp = path.with_name(path.stem + "_dark.npy")
        np.save(dp, np.asarray(dark, dtype=np.float32))
        meta["dark"] = dp.name
        meta["offset"] = None
    if info:
        meta["info"] = info
    with open(path.with_suffix(".json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
