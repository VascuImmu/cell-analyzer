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
# Corrections from reference images recorded on the microscope (background mode 'image')
# -------------------------------------------------
def _mean_projection(path, scenes, channel, downsample, z_projection):
    # runs in the main process: dask's thread pool must not be started here, otherwise the worker
    # processes forked later (Linux) can inherit a held lock and hang
    import dask
    with dask.config.set(scheduler="synchronous"):
        projs = [get_channel_projection(str(path), s, channel, downsample, z_projection, 0) for s in scenes]
    h = min(p.shape[0] for p in projs)
    w = min(p.shape[1] for p in projs)
    return np.mean([p[:h, :w] for p in projs], axis=0).astype(np.float32)


def _pick_scenes(path, include):
    from ..io_utils import name_matches
    from ..config import parse_str_list
    scenes = list_scenes(path)
    inc = parse_str_list(include)
    if inc:
        scenes = [s for s in scenes if name_matches(s, inc, [], False, False)]
    if not scenes:
        raise ValueError(f"{Path(path).name}: no scene matches {inc}")
    return scenes


def fields_from_reference_image(ref_path, channels, out_dir, kind="flatfield", channel_map=None,
                                scene_include="", dark_path=None, offset=0.0, downsample=4,
                                z_projection="max", smooth_sigma=50.0):
    """
    Build one correction per data channel from a reference image recorded on the microscope.

    kind 'flatfield': field = smoothed (reference - dark or offset); images are divided by it (mean 1).
    kind 'blank'    : field = smoothed reference; it is subtracted from the images.
    All scenes of the reference (optionally filtered) are averaged. Returns {data_channel: npy_path}.
    """
    from .flatfield import save_correction
    from ..io_utils import file_stem, acquisition_info
    ref_path = Path(ref_path)
    channel_map = channel_map or {}
    scenes = _pick_scenes(ref_path, scene_include)
    n_c = acquisition_info(ref_path, scenes[0])["n_channels"]
    dark_scenes = _pick_scenes(dark_path, "") if dark_path else None
    dark_nc = acquisition_info(dark_path, dark_scenes[0])["n_channels"] if dark_path else 0
    out_dir = Path(out_dir)
    stem = file_stem(ref_path)
    print(f"Reference {ref_path.name}: {len(scenes)} scene(s) averaged, used as {kind}"
          + ((f", dark image {Path(dark_path).name}" if dark_path else (f", camera offset {offset:g}" if offset else ""))
             if kind == "flatfield" else ""),
          flush=True)
    got = {}
    shown = False
    for ch in channels:
        rc = channel_map.get(ch, ch)
        if rc >= n_c:
            raise ValueError(f"data channel {ch} -> reference channel {rc}, but the reference has only {n_c} "
                             f"channel(s) (counting from 0). Set 'Channel mapping'.")
        mean = _mean_projection(ref_path, scenes, rc, downsample, z_projection)
        npy = out_dir / f"{stem}_ch{ch}_{kind}.npy"
        # A reference of a uniform sample only needs its noise removed, not the illumination pattern:
        # never smooth more than 3 % of the image width (the setting is meant for estimates from cells).
        sigma = float(min(smooth_sigma, max(1.0, 0.03 * max(mean.shape))))
        if not shown and sigma < smooth_sigma:
            print(f"  smoothing the reference with sigma {sigma:.1f} (downsampled px) instead of {smooth_sigma:g} "
                  f"-- 3 % of its width", flush=True)
            shown = True
        dark = None
        if kind == "flatfield":
            if dark_path:
                dark = _mean_projection(dark_path, dark_scenes, rc if rc < dark_nc else 0, downsample, z_projection)
                h, w = min(dark.shape[0], mean.shape[0]), min(dark.shape[1], mean.shape[1])
                dark, mean = gaussian_filter(dark[:h, :w], 2), mean[:h, :w]
                signal = mean - dark
            else:
                signal = mean - float(offset or 0)
            field = gaussian_filter(signal, sigma=sigma).astype(np.float32)
            if field.min() <= 0:
                raise ValueError(f"reference channel {rc}: after subtracting the offset/dark image the field is "
                                 f"not positive everywhere -- is this really a uniformly bright sample, and is the "
                                 f"offset too high?")
            norm = field / field.mean()
            save_correction(npy, field, "divide", offset=None if dark is not None else offset, dark=dark,
                            info=dict(reference=str(ref_path), reference_channel=rc, scenes=len(scenes)))
            print(f"  ✓ ch{ch} (reference ch{rc}): illumination {100 * norm.min():.0f} % … {100 * norm.max():.0f} % "
                  f"of the mean -> {npy.name}", flush=True)
            _plot_background_diagnostics(mean, field, norm, out_dir / f"{stem}_ch{ch}_{kind}.png")
        else:
            field = gaussian_filter(mean, sigma=sigma).astype(np.float32)
            save_correction(npy, field, "subtract",
                            info=dict(reference=str(ref_path), reference_channel=rc, scenes=len(scenes)))
            print(f"  ✓ ch{ch} (reference ch{rc}): blank level {field.min():.0f} … {field.max():.0f} grey values "
                  f"(subtracted) -> {npy.name}", flush=True)
            _plot_background_diagnostics(mean, field, field / field.mean(), out_dir / f"{stem}_ch{ch}_{kind}.png")
        got[ch] = str(npy)
    return got


def compare_acquisition(ref, img, rel_tol=0.02):
    """
    Compare acquisition_info() of the reference with that of an input image.
    Returns a list of (level, text); level 'error' = the reference does not fit, 'warning' = check it.
    """
    out = []
    ro, io = ref.get("objective") or {}, img.get("objective") or {}
    if ro and io:
        if ro.get("magnification") and io.get("magnification") and abs(ro["magnification"] - io["magnification"]) > 0.01:
            out.append(("error", f"objective magnification {ro['magnification']:g}x (reference) vs {io['magnification']:g}x"))
        if ro.get("na") and io.get("na") and abs(ro["na"] - io["na"]) > 0.005:
            out.append(("error", f"objective NA {ro['na']:g} (reference) vs {io['na']:g}"))
        if ro.get("name") and io.get("name") and str(ro["name"]).strip() != str(io["name"]).strip():
            out.append(("warning", f"objective '{ro['name']}' (reference) vs '{io['name']}'"))
    elif ro or io:
        out.append(("warning", "objective information is only stored in one of the two files"))
    else:
        out.append(("info", "no objective information in either file"))
    rp, ip = ref.get("pixel_size_um"), img.get("pixel_size_um")
    if rp and ip:
        if abs(rp - ip) / ip > rel_tol:
            out.append(("error", f"pixel size {rp:.4g} µm (reference) vs {ip:.4g} µm"))
    else:
        out.append(("warning", "pixel size missing in " + ("the reference" if not rp else "the image")))
    rf, imf = ref.get("field_of_view_um"), img.get("field_of_view_um")
    if rf and imf:
        if any(abs(a - b) / b > rel_tol for a, b in zip(rf, imf)):
            out.append(("error", f"field of view {rf[1]:.0f} × {rf[0]:.0f} µm (reference) vs {imf[1]:.0f} × {imf[0]:.0f} µm "
                                 f"-- the illumination pattern cannot be mapped onto the images"))
    elif tuple(ref["shape"]) != tuple(img["shape"]):
        out.append(("error", f"image size {ref['shape'][1]} × {ref['shape'][0]} px (reference) vs "
                             f"{img['shape'][1]} × {img['shape'][0]} px"))
    if tuple(ref["shape"]) != tuple(img["shape"]) and not any(l == "error" and "field of view" in t for l, t in out):
        out.append(("info", f"image size differs ({ref['shape'][1]} × {ref['shape'][0]} vs {img['shape'][1]} × "
                            f"{img['shape'][0]} px, e.g. binning) -- the field is rescaled"))
    rn, inn = ref.get("channel_names") or [], img.get("channel_names") or []
    if rn and inn and rn != inn:
        out.append(("warning", f"channel names differ: {rn} (reference) vs {inn} -- check 'Channel mapping'"))
    return out


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
