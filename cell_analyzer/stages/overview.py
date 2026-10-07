"""
cell_analyzer/stages/overview.py -- overview pictures of the raw images.

One figure per scene: every channel in its own panel (max projection over Z, each channel
scaled to its own bright percentile), optionally a merged panel (additive "screen" blending,
as in napari), with a scale bar when the pixel size is known.

    per_file/<dataset>/overview/<scene>_overview.png

Independent of the segmentation: it only reads the images.

Standalone:
    from cell_analyzer.stages.overview import multi_channel_plot
    multi_channel_plot(img_cyx, channel_names=[...], merge=True, pixel_size=0.65, save_path="x.png")
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from .. import io_utils

# colour-blind friendly order: cyan / magenta / yellow are told apart with every common colour-vision
# deficiency; green comes last because it can be confused with yellow
DEFAULT_COLORS = ["gray", "cyan", "magenta", "yellow", "green"]
_BRIGHT = {"green": (0.0, 1.0, 0.0), "gray": (1.0, 1.0, 1.0), "grey": (1.0, 1.0, 1.0), "white": (1.0, 1.0, 1.0),
           "blue": (0.25, 0.45, 1.0), "red": (1.0, 0.0, 0.0), "orange": (1.0, 0.55, 0.0)}
_NICE = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]


def channel_cmap(color):
    """Black -> colour colormap ('gray' = black -> white). Unknown names fall back to white."""
    name = str(color).strip().lower()
    try:
        rgb = _BRIGHT.get(name) or to_rgb(name)
    except ValueError:
        rgb = (1.0, 1.0, 1.0)
    return LinearSegmentedColormap.from_list(f"ca_{name}", [(0, 0, 0), rgb])


def _parse_list(text):
    if text is None:
        return []
    if isinstance(text, (list, tuple)):
        return [str(t).strip() for t in text if str(t).strip()]
    return [t.strip() for t in str(text).replace(";", ",").split(",") if t.strip()]


def auto_scalebar_um(width_um):
    """A round length of about 1/5 of the image width."""
    target = width_um / 5.0
    return float(max([n for n in _NICE if n <= target] or [_NICE[0]]))


def normalize_channel(ch, percentile=99.5, low_percentile=0.0):
    lo = float(np.percentile(ch, low_percentile)) if low_percentile > 0 else 0.0
    hi = float(np.percentile(ch, percentile))
    if hi <= lo:
        hi = float(ch.max()) if float(ch.max()) > lo else lo + 1.0
    return np.clip((ch - lo) / (hi - lo), 0, 1).astype(np.float32)


def multi_channel_plot(img, channel_names=None, merge=False, colored_subplots=True, title=None,
                       pixel_size=None, scalebar_length=None, save_path=None, dpi=200,
                       colors=None, percentile=99.5, low_percentile=0.0, merge_channels=None,
                       panel_size=4.0):
    """
    img: (C, Y, X) array (numpy or dask), any number of channels.
    colors: one colour name per channel (cycled); merge_channels: indices into `img` for the merged panel
    (None = all). scalebar_length in µm (None = automatic); a bar is drawn only if pixel_size is given.
    """
    if hasattr(img, "compute"):
        img = img.compute()
    img = np.asarray(img).squeeze()
    if img.ndim == 2:
        img = img[None]
    n = img.shape[0]
    colors = list(colors or DEFAULT_COLORS)
    cmaps = [channel_cmap(colors[i % len(colors)]) for i in range(n)]
    gray = channel_cmap("gray")
    names = list(channel_names or [])
    names = [names[i] if i < len(names) and names[i] else f"Ch{i}" for i in range(n)]
    norm = [normalize_channel(img[i], percentile, low_percentile) for i in range(n)]
    h, w = norm[0].shape
    del img

    do_merge = bool(merge) and n > 1
    n_panels = n + 1 if do_merge else n
    fig, ax = plt.subplots(1, n_panels, figsize=(panel_size * n_panels, panel_size * h / w + 0.5 + (0.4 if title else 0)),
                           squeeze=False)
    ax = ax[0]
    for i in range(n):
        ax[i].imshow(norm[i], cmap=cmaps[i] if colored_subplots else gray, vmin=0, vmax=1, interpolation="nearest")
        ax[i].set_title(names[i], fontsize=12)
        ax[i].axis("off")

    if do_merge:
        idx = [i for i in (merge_channels if merge_channels else range(n)) if 0 <= i < n] or list(range(n))
        inv = np.ones((h, w, 3), dtype=np.float32)
        for i in idx:                                   # screen blending: 1 - prod(1 - rgb)
            inv *= 1.0 - cmaps[i](norm[i])[..., :3].astype(np.float32)
        ax[-1].imshow(np.clip(1.0 - inv, 0, 1), interpolation="nearest")
        ax[-1].set_title("Merge" if len(idx) == n else "Merge (" + ", ".join(names[i] for i in idx) + ")", fontsize=12)
        ax[-1].axis("off")

    if title:
        fig.suptitle(title, fontsize=14)
    fig.subplots_adjust(wspace=0.02, left=0.005, right=0.995, bottom=0.01, top=0.86 if title else 0.92)

    if pixel_size:
        bar_um = float(scalebar_length) if scalebar_length else auto_scalebar_um(w * pixel_size)
        label = f"{bar_um:g} µm"
        for a in ax:
            a.add_artist(AnchoredSizeBar(a.transData, bar_um / pixel_size, label, "lower right", pad=0.1, borderpad=0.5,
                                         sep=3, color="white", frameon=False, size_vertical=max(1.0, 0.012 * h),
                                         label_top=True, fontproperties=dict(size=8)))

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    return fig


def overview_scene(path_str, scene, stem, out_dir_str, p):
    """Worker: one overview figure for one scene. Returns dict(status, scene, file)."""
    out = Path(out_dir_str) / f"{io_utils.scene_stem(stem, scene)}_overview.png"
    if out.exists() and not p.get("overwrite_overview", True):
        return dict(status="skipped_exists", scene=scene, file=str(out))
    stack, ps, file_names = io_utils.load_mip_stack(path_str, scene, timepoint=p.get("timepoint", 0))
    n = stack.shape[0]
    chans = [int(c) for c in _parse_list(p.get("overview_channels"))] or list(range(n))
    bad = [c for c in chans if not 0 <= c < n]
    if bad:
        raise ValueError(f"overview channel(s) {bad} do not exist (the image has {n} channels, counting from 0)")
    names = _parse_list(p.get("channel_names")) or file_names
    names = [names[c] if c < len(names) else f"Ch{c}" for c in chans]
    ps = p.get("pixel_size_um") or ps
    stack = stack[chans]

    step = 1                                             # shrink very large (stitched) images
    max_px = int(p.get("overview_max_px") or 0)
    if max_px > 0 and max(stack.shape[1:]) > max_px:
        step = int(np.ceil(max(stack.shape[1:]) / max_px))
        y, x = (stack.shape[1] // step) * step, (stack.shape[2] // step) * step
        stack = stack[:, :y, :x].reshape(len(chans), y // step, step, x // step, step).mean(axis=(2, 4))

    merge_sel = [int(c) for c in _parse_list(p.get("overview_merge_channels"))]
    merge_idx = [chans.index(c) for c in merge_sel if c in chans] or None
    title = f"{stem} · {scene}" if p.get("overview_title", True) else None
    multi_channel_plot(stack, channel_names=names, merge=p.get("overview_merge", True),
                       colored_subplots=p.get("overview_colored", True), title=title,
                       pixel_size=(ps * step) if (ps and p.get("overview_scalebar", True)) else None,
                       scalebar_length=p.get("overview_scalebar_um") or None, save_path=out,
                       dpi=int(p.get("overview_dpi") or 200), colors=_parse_list(p.get("overview_colors")) or None,
                       percentile=float(p.get("overview_percentile") or 99.5),
                       low_percentile=float(p.get("overview_low_percentile") or 0.0), merge_channels=merge_idx)
    return dict(status="ok", scene=scene, file=str(out))
