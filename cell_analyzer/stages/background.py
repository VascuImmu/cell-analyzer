"""
Estimate a flatfield / background illumination field by accumulating one
channel (nuclei by default) across many scenes -- of one file, or pooled
across many files.

Rationale
---------
Real nuclei are spatially random from scene to scene, but illumination bias
(vignetting, uneven excitation, camera gain non-uniformity) is spatially
consistent across the whole acquisition. Averaging over many scenes
therefore cancels the biological signal and leaves the smooth illumination
pattern; a large Gaussian blur mops up residual structure.

For a folder of single-scene files (typical .czi export) a per-file field
would just be one blurred image -- use `sources` pooled over all files
instead (background_mode = "pooled" in the pipeline).

Memory strategy
----------------
- Only the requested channel is pulled from disk (lazy dask slicing).
- Each scene is reduced to 2D and downsampled *before* compute().
- Projections are small; they are summed in the parent process.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import numpy as np
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..io_utils import open_image, list_scenes


# -------------------------------------------------
# Per-scene worker (top-level for ProcessPoolExecutor)
# -------------------------------------------------
def get_channel_projection(path_str, scene, channel=0, downsample=4, z_projection="max", timepoint=0):
    import dask.array as da

    img = open_image(path_str)
    img.set_scene(scene)
    ch = img.get_image_dask_data("TCZYX")[timepoint, channel]  # lazy ZYX

    if downsample > 1:
        z, h, w = ch.shape
        h2, w2 = h - (h % downsample), w - (w % downsample)
        ch = ch[:, :h2, :w2]
        ch = da.coarsen(np.mean, ch, {0: 1, 1: downsample, 2: downsample})

    if z_projection == "max":
        proj = ch.max(axis=0)
    elif z_projection == "mean":
        proj = ch.mean(axis=0)
    elif z_projection == "middle":
        proj = ch[ch.shape[0] // 2]
    else:
        raise ValueError(f"Unknown z_projection: {z_projection}")

    return np.asarray(proj.compute(), dtype=np.float32)


# -------------------------------------------------
# Main accumulation routine
# -------------------------------------------------
def estimate_background_field(
    sources,
    channel=0,
    downsample=4,
    z_projection="max",
    smooth_sigma=50,
    n_workers=None,
    max_scenes=None,
    scene_filter=None,
    save_path=None,
    plot_path=None,
    timepoint=0,
):
    """
    Parameters
    ----------
    sources : path, or list of (path, scene) tuples
        A single file (all its scenes, optionally filtered by `scene_filter`)
        or an explicit list of (file, scene) pairs -- possibly spanning many
        files, for a pooled estimate.
    scene_filter : callable, "merged_only" or None
        Only used when `sources` is a single path.
    (other parameters as before)

    Returns
    -------
    background_field, flatfield, mean_field : np.ndarray (float32)
    """
    if isinstance(sources, (str, Path)):
        path = Path(sources)
        scenes = list_scenes(path)
        if scene_filter == "merged_only":
            scenes = [s for s in scenes if "_Merged" in s]
        elif callable(scene_filter):
            scenes = [s for s in scenes if scene_filter(s)]
        pairs = [(str(path), s) for s in scenes]
    else:
        pairs = [(str(p), s) for p, s in sources]

    if max_scenes:
        # spread the sample evenly over the list rather than taking the first N
        if len(pairs) > max_scenes:
            idx = np.linspace(0, len(pairs) - 1, max_scenes).round().astype(int)
            pairs = [pairs[i] for i in sorted(set(idx))]

    if not pairs:
        raise ValueError("No scenes to process after filtering.")

    if not n_workers:
        n_workers = min(4, max(1, multiprocessing.cpu_count() - 1))

    print(f"Accumulating channel {channel} from {len(pairs)} scene(s) using {n_workers} workers...", flush=True)

    projections = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(get_channel_projection, p, s, channel, downsample, z_projection, timepoint): (p, s)
            for p, s in pairs
        }
        for future in as_completed(futures):
            p, s = futures[future]
            try:
                proj = future.result()
            except Exception as e:
                print(f"✗ Skipping {Path(p).name} / {s}: {e}", flush=True)
                continue
            projections.append(proj)
            print(f"  + {Path(p).name} / {s}  ({len(projections)}/{len(pairs)})", flush=True)

    if not projections:
        raise RuntimeError("No scenes were successfully processed.")

    min_h = min(p.shape[0] for p in projections)
    min_w = min(p.shape[1] for p in projections)
    if len({p.shape for p in projections}) > 1:
        print(f"⚠ Scene shapes vary; cropping all to ({min_h}, {min_w})", flush=True)

    acc = np.zeros((min_h, min_w), dtype=np.float64)
    for proj in projections:
        acc += proj[:min_h, :min_w]
    mean_field = (acc / len(projections)).astype(np.float32)

    background_field = gaussian_filter(mean_field, sigma=smooth_sigma).astype(np.float32)
    flatfield = (background_field / background_field.mean()).astype(np.float32)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, background_field)
        print(f"✓ Saved background field to {save_path}", flush=True)

    if plot_path is not None:
        _plot_background_diagnostics(mean_field, background_field, flatfield, plot_path)

    return background_field, flatfield, mean_field


def _plot_background_diagnostics(mean_field, background_field, flatfield, save_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for a, img, title, cmap in [
        (ax[0], mean_field, "Raw mean projection", "gray"),
        (ax[1], background_field, "Smoothed background field", "gray"),
        (ax[2], flatfield, "Flatfield correction map (mean=1)", "magma"),
    ]:
        im = a.imshow(img, cmap=cmap)
        a.set_title(title)
        a.axis("off")
        plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved diagnostic plot to {save_path}", flush=True)


# -------------------------------------------------
# CLI (single file)
# -------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cell_analyzer.stages.background <image_file> [channel] [downsample] [n_workers]")
        sys.exit(1)

    path = sys.argv[1]
    channel = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    downsample = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None

    out_dir = Path(path).parent / "background_field"
    out_dir.mkdir(exist_ok=True)
    stem = Path(path).stem
    estimate_background_field(
        path, channel=channel, downsample=downsample, n_workers=n_workers,
        save_path=out_dir / f"{stem}_ch{channel}_background.npy",
        plot_path=out_dir / f"{stem}_ch{channel}_background.png",
    )
