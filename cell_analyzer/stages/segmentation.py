"""
Marker-controlled watershed segmentation (any format AICSImage reads:
.lif, .czi, .ome.tif, ...).

Channel `nuclei_channel` (DNA/DAPI) seeds the watershed, channel
`edge_channel` (junction/membrane marker) is the surface it floods across,
so cell boundaries land on the junction signal.

Pipeline per scene
-------------------
1. Load ONLY the nuclei + edge channel (lazy dask slicing).
2. Reduce to 2D (MIP by default; per-slice or true-3D also available).
3. Optional flatfield correction, BEFORE segmentation.
4. Nuclei: smooth -> Otsu -> clean -> distance transform -> peaks -> seeds
   -> watershed -> one label per nucleus.
5. Confluence decision (see below).
6. Cells: watershed on the edge surface (gradient or intensity), seeded
   by the nucleus labels -> cell labels share the nucleus IDs.
7. Multi-nucleated cells: two neighbouring cells whose nuclei are close and whose shared
   boundary carries NO junction signal are merged back into one cell (see below).
8. Size filter, border clearing, save .tif + QC plot + per-scene summary + size statistics.

Multi-nucleated cells
----------------------
Every nucleus seeds one cell, so a cell with two nuclei is first cut in two. The cut
runs exactly where a junction would have to be -- between the two nuclei. For every pair
of touching cells the edge-channel intensity along their shared boundary is compared
with (a) the non-junction level of the scene (median edge intensity under the nuclei) and
(b) the typical boundary of the scene (median over all cell-cell boundaries):

    contrast = (boundary - non_junction) / (typical_boundary - non_junction)

contrast ~ 1 -> a normal junction; ~ 0 -> no junction. Pairs with contrast below
`multinuc_junction_ratio` AND nuclei closer than `multinuc_max_distance` are merged
(weakest boundaries first, at most `multinuc_max_nuclei` nuclei per cell). The merged
cell and its nuclei share one label ID.

Confluent vs. non-confluent scenes
-----------------------------------
A scene with >= `min_nuclei_confluent` nuclei is treated as a confluent
monolayer: the cell watershed may fill the whole field of view.

With fewer nuclei:
  * nonconfluent_enabled=True  -> the watershed is restricted to a
    foreground mask ("union" | "nuclei" | "edge"), so cells don't flood
    across empty space.
  * nonconfluent_enabled=False -> `sparse_policy` decides:
      "confluent" = segment like any other scene (no mask)
      "skip"      = don't segment the scene at all.

A per-dataset <stem>_segmentation_summary.csv records n_nuclei, n_cells and
the confluence decision for every scene; measurement adds it to each cell.
"""

import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import warnings
import numpy as np

# skimage >= 0.26 renames some arguments/properties; keep the log readable
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import tifffile

from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes, disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, mark_boundaries, clear_border

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .flatfield import load_and_normalize_background, apply_flatfield_correction
from ..io_utils import load_channel, n_z, scene_stem

# Defaults for calling the functions of this module directly, in PIXELS (size_units="px").
# The pipeline passes the settings from config.py instead (µm, or empty = automatic) and
# segment_scene converts them per image with sizes.resolve_sizes.
DEFAULT_SEG_PARAMS = dict(
    size_units="px", pixel_size_um=None, typical_nucleus_diameter_um=10.0, typical_cell_area_um2=1000.0,
    seed_source="nuclei",
    nuclei_channel=0,
    edge_channel=1,
    seg_mode="mip",
    timepoint=0,
    nuclei_sigma=1.5,
    min_nucleus_size=1100,
    min_distance=20,
    split_touching_nuclei=True,
    edge_sigma=3.0,
    edge_mode="intensity",
    min_label_size=2000,
    merge_multinucleated=True,
    multinuc_max_distance=125,
    multinuc_junction_ratio=0.5,
    multinuc_max_nuclei=2,
    multinuc_band_px=2,
    seed_sigma=15.0,
    seed_min_depth=0.02,
    junction_merge_ratio=0.4,
    max_cell_area=160000,
    clear_border_labels=True,
    clear_border_buffer=10,
    nonconfluent_enabled=True,
    min_nuclei_confluent=2580,
    foreground_mask="union",
    edge_threshold=165.0,
    edge_threshold_percentile=20.0,
    nuc_dilation_rad=7,
    nuc_dilation_iterations=3,
    min_hole_size=2000,
    boundary_band_px=1,
    sparse_policy="confluent",
    overwrite_segmentation=True,
    save_nuclei_mask=True,
    save_qc_plot=True,
    qc_dpi=100,
    save_adjacency=True,
)


def seg_params_from_config(cfg):
    return {k: cfg[k] for k in DEFAULT_SEG_PARAMS}


# -------------------------------------------------
# Shape normalization helper
# -------------------------------------------------
def crop_to_common_shape(*arrays):
    shapes = [a.shape for a in arrays]
    if len(set(shapes)) == 1:
        return arrays
    min_shape = tuple(min(s[ax] for s in shapes) for ax in range(len(shapes[0])))
    print(f"⚠ Shape mismatch {sorted(set(shapes))} -> cropping to {min_shape}", flush=True)
    slices = tuple(slice(0, m) for m in min_shape)
    return tuple(a[slices] for a in arrays)


def _struct(radius, ndim):
    return disk(radius) if ndim == 2 else np.ones((radius,) * 3, dtype=bool)


# -------------------------------------------------
# Stage 1: nuclei
# -------------------------------------------------
def segment_nuclei(nuclei_img, min_distance=20, nuclei_sigma=1.5, min_nucleus_size=1100,
                   split_touching_nuclei=True, diag=None):
    """`diag` (optional dict) receives 'raw_object_areas': areas of every thresholded object
    BEFORE the min-size filter -- shows debris vs. nuclei when choosing min_nucleus_size."""
    nuclei_smooth = gaussian(nuclei_img, sigma=nuclei_sigma, preserve_range=True)
    empty = np.zeros_like(nuclei_img, dtype=np.int32)
    try:
        thresh = threshold_otsu(nuclei_smooth)
    except ValueError:
        return np.zeros_like(nuclei_img, dtype=bool), empty, 0

    nuclei_mask = nuclei_smooth > thresh
    nuclei_mask = ndi.binary_fill_holes(nuclei_mask)
    if diag is not None:
        raw_lab, n_raw = ndi.label(nuclei_mask)
        diag["raw_object_areas"] = np.bincount(raw_lab.ravel())[1:].astype(np.int64) if n_raw else np.zeros(0, np.int64)
        del raw_lab
    nuclei_mask = remove_small_objects(nuclei_mask, min_size=min_nucleus_size)
    if not nuclei_mask.any():
        return nuclei_mask, empty, 0

    distance = ndi.distance_transform_edt(nuclei_mask)
    coords = peak_local_max(distance, min_distance=min_distance, labels=nuclei_mask)
    seed_mask = np.zeros_like(nuclei_mask, dtype=bool)
    seed_mask[tuple(coords.T)] = True
    markers, n_seeds = ndi.label(seed_mask)
    if n_seeds == 0:
        return nuclei_mask, empty, 0

    nuclei_labels = watershed(-distance, markers=markers, mask=nuclei_mask)
    if split_touching_nuclei:
        # keep the watershed split; just renumber 1..N
        from skimage.segmentation import relabel_sequential
        nuclei_labels = relabel_sequential(nuclei_labels)[0]
        n_nuclei = int(nuclei_labels.max())
    else:
        # behaviour of the original script: re-labelling connected components merges the
        # watershed pieces again, so touching nuclei end up as ONE label
        nuclei_labels, n_nuclei = ndi.label(nuclei_labels > 0, structure=np.ones([3] * nuclei_labels.ndim))
    return nuclei_mask, nuclei_labels.astype(np.int32), int(n_nuclei)


# -------------------------------------------------
# Foreground mask for non-confluent scenes
# -------------------------------------------------
def build_foreground_mask(
    kind, edge_smooth, nuclei_mask,
    edge_threshold=None, edge_threshold_percentile=20,
    nuc_dilation_rad=7, nuc_dilation_iterations=3, min_hole_size=2000,
):
    struct = _struct(nuc_dilation_rad, nuclei_mask.ndim)
    if edge_threshold is None:
        edge_threshold = np.percentile(edge_smooth, edge_threshold_percentile)

    if kind == "nuclei":
        mask = ndi.binary_dilation(nuclei_mask, structure=struct, iterations=nuc_dilation_iterations)
    elif kind == "edge":
        mask = ndi.binary_closing(edge_smooth > edge_threshold, struct)
    elif kind == "union":
        mask = ndi.binary_dilation(nuclei_mask, structure=struct, iterations=nuc_dilation_iterations)
        mask = mask | (edge_smooth > edge_threshold)
        mask = ndi.binary_closing(mask, struct)
    else:
        raise ValueError(f"Unknown foreground_mask option: {kind}")

    if min_hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=min_hole_size)
    return mask


# -------------------------------------------------
# Multi-nucleated cells: merge neighbours that have no junction between them
# -------------------------------------------------
def merge_multinucleated_cells(cell_labels, nuclei_labels, edge_smooth, nuclei_mask,
                               max_distance=0, junction_ratio=0.5, max_nuclei=2, band_px=2,
                               min_boundary_px=5):
    """
    Merge pairs of touching cells whose shared boundary has no junction signal and whose
    nuclei are close together (2D only). Returns (cell_labels, nuclei_labels, info).

    info: n_merged_pairs, pairs [(y0, x0, y1, x1) nucleus centroids], nuclei_per_label {label: n},
          max_distance, reference, baseline
    """
    info = dict(n_merged_pairs=0, pairs=[], nuclei_per_label={}, max_distance=float(max_distance or 0),
                reference=np.nan, baseline=np.nan)
    n = int(nuclei_labels.max())
    if cell_labels.ndim != 2 or n < 2:
        return cell_labels, nuclei_labels, info

    idx = np.arange(1, n + 1)
    present = nuclei_labels > 0
    areas = np.asarray(ndi.sum(present, nuclei_labels, idx), dtype=float)
    cents = np.asarray(ndi.center_of_mass(present, nuclei_labels, idx), dtype=float)
    if not max_distance or max_distance <= 0:
        # automatic: two typical nucleus diameters between the centres
        max_distance = 2.0 * float(np.median(2.0 * np.sqrt(areas[areas > 0] / np.pi)))
    info["max_distance"] = float(max_distance)

    # edge intensity on the boundary pixels of every pair of touching cells
    E = ndi.maximum_filter(edge_smooth, size=2 * band_px + 1) if band_px > 0 else edge_smooth
    L = cell_labels
    keys, vals = [], []
    for a, b, ea, eb in ((L[:, :-1], L[:, 1:], E[:, :-1], E[:, 1:]), (L[:-1], L[1:], E[:-1], E[1:])):
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * (n + 1) + hi)
        vals.append(0.5 * (ea[m] + eb[m]))
    if not keys:
        return cell_labels, nuclei_labels, info
    keys, vals = np.concatenate(keys), np.concatenate(vals)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=vals) / counts

    long_enough = counts >= min_boundary_px
    if not long_enough.any():
        return cell_labels, nuclei_labels, info
    baseline = float(np.median(edge_smooth[nuclei_mask])) if nuclei_mask.any() else float(np.percentile(edge_smooth, 5))
    reference = float(np.median(means[long_enough]))
    info.update(reference=reference, baseline=baseline)
    if not reference > baseline:        # no junction contrast at all in this scene -> don't touch anything
        return cell_labels, nuclei_labels, info

    contrast = (means - baseline) / (reference - baseline)
    lo, hi = uniq // (n + 1), uniq % (n + 1)
    dist = np.linalg.norm(cents[lo - 1] - cents[hi - 1], axis=1)
    cand = np.where(long_enough & (contrast < junction_ratio) & (dist <= max_distance))[0]
    if cand.size == 0:
        return cell_labels, nuclei_labels, info

    # union-find, weakest boundary first, capped at max_nuclei per cell
    parent = np.arange(n + 1)
    size = np.ones(n + 1, dtype=int)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for k in cand[np.argsort(contrast[cand])]:
        a, b = find(int(lo[k])), find(int(hi[k]))
        if a == b or size[a] + size[b] > max_nuclei:
            continue
        root, other = (a, b) if a < b else (b, a)
        parent[other] = root
        size[root] += size[other]
        info["pairs"].append((*cents[lo[k] - 1], *cents[hi[k] - 1]))
    if not info["pairs"]:
        return cell_labels, nuclei_labels, info

    mapping = np.array([find(i) for i in range(n + 1)])
    info["n_merged_pairs"] = len(info["pairs"])
    info["nuclei_per_label"] = {int(l): int(c) for l, c in enumerate(np.bincount(mapping[1:], minlength=n + 1)) if c > 1}
    return mapping[cell_labels].astype(cell_labels.dtype), mapping[nuclei_labels].astype(nuclei_labels.dtype), info


def object_statistics(individual_nuclei, cell_labels, nuclei_per_label=None):
    """Per-object size/shape tables (pixels) for the size-statistics plots.
    `individual_nuclei` = nuclei labels BEFORE multi-nucleated merging (one label per nucleus)."""
    from skimage.measure import regionprops_table
    from scipy.spatial import cKDTree
    props = ("label", "area", "eccentricity", "equivalent_diameter_area", "axis_major_length",
             "axis_minor_length", "centroid")
    out = {}
    for name, lab in (("nuclei", individual_nuclei), ("cells", cell_labels)):
        if lab is None or lab.ndim != 2 or not lab.any():
            out[name] = pd.DataFrame()
            continue
        t = pd.DataFrame(regionprops_table(lab, properties=props)).rename(columns={
            "label": "label_id", "area": "area_px", "equivalent_diameter_area": "equivalent_diameter_px",
            "axis_major_length": "major_axis_px", "axis_minor_length": "minor_axis_px",
            "centroid-0": "centroid_y", "centroid-1": "centroid_x"})
        if name == "nuclei":
            if len(t) > 1:
                d, _ = cKDTree(t[["centroid_y", "centroid_x"]].to_numpy()).query(
                    t[["centroid_y", "centroid_x"]].to_numpy(), k=2)
                t["nearest_neighbour_px"] = d[:, 1]
            else:
                t["nearest_neighbour_px"] = np.nan
        else:
            npl = nuclei_per_label or {}
            t["n_nuclei"] = [npl.get(int(l), 1) for l in t["label_id"]] if individual_nuclei is not None else np.nan
        out[name] = t.drop(columns=["centroid_y", "centroid_x"]).round(3)
    return out


# -------------------------------------------------
# Which cells touch each other
# -------------------------------------------------
def _touching_pairs(labels, values=None):
    """All pairs of different non-zero labels that share a pixel edge (4-connectivity).
    Returns (label_a, label_b, n_contact_pixel_pairs, mean of `values` on the contact) with a < b."""
    n = int(labels.max())
    keys, vals = [], []
    views = [(labels[:, :-1], labels[:, 1:]), (labels[:-1], labels[1:])]
    vviews = [(values[:, :-1], values[:, 1:]), (values[:-1], values[1:])] if values is not None else [None, None]
    for (a, b), vv in zip(views, vviews):
        m = (a != b) & (a > 0) & (b > 0)
        if not m.any():
            continue
        lo = np.minimum(a[m], b[m]).astype(np.int64)
        hi = np.maximum(a[m], b[m]).astype(np.int64)
        keys.append(lo * (n + 1) + hi)
        if vv is not None:
            vals.append(0.5 * (vv[0][m] + vv[1][m]))
    if not keys:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, np.zeros(0)
    keys = np.concatenate(keys)
    uniq, inv, counts = np.unique(keys, return_inverse=True, return_counts=True)
    means = np.bincount(inv, weights=np.concatenate(vals)) / counts if vals else np.full(len(uniq), np.nan)
    return uniq // (n + 1), uniq % (n + 1), counts, means


def cell_adjacency(cell_labels, edge_smooth=None):
    """
    Contact-based adjacency of the final cell labels (2D): two cells are neighbours when their
    label regions share a boundary -- NOT when their centres are merely close.

    Returns a DataFrame: cell_id_a < cell_id_b, contact_px (number of touching pixel pairs, i.e. the
    length of the shared boundary; diagonal stretches are over-counted by up to 1.4x) and
    junction_intensity (mean smoothed edge-channel intensity on that boundary).
    """
    if cell_labels.ndim != 2 or not cell_labels.any():
        return pd.DataFrame(columns=["cell_id_a", "cell_id_b", "contact_px", "junction_intensity"])
    a, b, counts, means = _touching_pairs(cell_labels, edge_smooth)
    return pd.DataFrame({"cell_id_a": a.astype(int), "cell_id_b": b.astype(int),
                         "contact_px": counts.astype(int), "junction_intensity": np.round(means, 3)})


def adjacency_matrix(adjacency, n_labels=None):
    """Sparse symmetric matrix A with A[i, j] = contact length (px) between cell i and cell j;
    rows/columns are indexed directly by cell_id (row 0 = background, always empty)."""
    from scipy import sparse
    n = int(n_labels if n_labels is not None else (adjacency[["cell_id_a", "cell_id_b"]].to_numpy().max() if len(adjacency) else 0)) + 1
    i, j, v = adjacency["cell_id_a"].to_numpy(), adjacency["cell_id_b"].to_numpy(), adjacency["contact_px"].to_numpy()
    return sparse.coo_matrix((np.r_[v, v], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()


# -------------------------------------------------
# Shared finishing steps: size filter, border clearing, adjacency
# -------------------------------------------------
def _finish_cells(cell_labels, edge_smooth, min_label_size, clear_border_labels, clear_border_buffer, info):
    if min_label_size > 0:
        cell_labels = remove_small_objects(cell_labels, min_size=min_label_size)
    if clear_border_labels:
        cell_labels = clear_border(cell_labels, buffer_size=clear_border_buffer)
    cell_labels = cell_labels.astype(np.int32)
    n_cells = int(len(np.unique(cell_labels)) - (1 if (cell_labels == 0).any() else 0))
    info["adjacency"] = cell_adjacency(cell_labels, edge_smooth)
    return cell_labels, n_cells


def _surface(edge_smooth, edge_mode, edge_sigma=1.0):
    """Watershed landscape: high on junctions, low inside cells.

    "intensity": the smoothed junction image itself -- boundaries run along the junction centre.
    "gradient":  Sobel magnitude. It has a dip along the centre line of every junction, through
                 which one label could flood the whole junction network; a grey closing fills
                 that dip, so that neighbouring cells touch each other.
    """
    if edge_mode == "intensity":
        return edge_smooth
    if edge_mode in ("gradient", "LoG"):
        g = np.sqrt(sum(ndi.sobel(edge_smooth, axis=a) ** 2 for a in range(edge_smooth.ndim)))
        k = int(2 * np.ceil(1.5 * float(edge_sigma or 0) + 2) + 1)
        return ndi.grey_closing(g, size=(k,) * g.ndim)
    raise ValueError(f"Unknown edge_mode: {edge_mode}")


# -------------------------------------------------
# Stage 2a: cells, seeded from nuclei
# -------------------------------------------------
def segment_cells_from_nuclei(
    edge_img, nuclei_labels, nuclei_mask, confluent,
    edge_sigma=3.0, edge_mode="intensity", min_label_size=2000,
    clear_border_labels=True, clear_border_buffer=10,
    nonconfluent_enabled=True, foreground_mask="union",
    edge_threshold=165.0, edge_threshold_percentile=20.0,
    nuc_dilation_rad=7, nuc_dilation_iterations=3, min_hole_size=2000,
    merge_multinucleated=True, multinuc_max_distance=0, multinuc_junction_ratio=0.5,
    multinuc_max_nuclei=2, multinuc_band_px=2,
):
    """Returns (cell_labels, nuclei_labels, n_cells, info). nuclei_labels differ from the
    input only where multi-nucleated cells were merged (their nuclei then share the cell's ID).
    info: n_merged_pairs, pairs, nuclei_per_label, adjacency (DataFrame)."""
    edge_smooth = gaussian(edge_img, sigma=edge_sigma, preserve_range=True)

    mask = None
    if not confluent and nonconfluent_enabled:
        mask = build_foreground_mask(
            foreground_mask, edge_smooth, nuclei_mask,
            edge_threshold=edge_threshold, edge_threshold_percentile=edge_threshold_percentile,
            nuc_dilation_rad=nuc_dilation_rad, nuc_dilation_iterations=nuc_dilation_iterations,
            min_hole_size=min_hole_size,
        )

    cell_labels = watershed(_surface(edge_smooth, edge_mode, edge_sigma), markers=nuclei_labels, mask=mask)

    info = dict(n_merged_pairs=0, pairs=[], nuclei_per_label={})
    if merge_multinucleated and multinuc_max_nuclei > 1:
        cell_labels, nuclei_labels, info = merge_multinucleated_cells(
            cell_labels, nuclei_labels, edge_smooth, nuclei_mask,
            max_distance=multinuc_max_distance, junction_ratio=multinuc_junction_ratio,
            max_nuclei=multinuc_max_nuclei, band_px=multinuc_band_px)

    cell_labels, n_cells = _finish_cells(cell_labels, edge_smooth, min_label_size, clear_border_labels,
                                         clear_border_buffer, info)
    return cell_labels, nuclei_labels.astype(np.int32), n_cells, info


# -------------------------------------------------
# Stage 2b: cells from the junction channel alone (no nuclear stain)
# -------------------------------------------------
def seeds_from_junctions(edge_img, seed_sigma=15.0, min_depth=0.02):
    """
    One seed per dark basin of the strongly blurred junction image (h-minima).
    Computed on a down-sampled copy for speed; returns a marker image at full resolution.
    """
    from skimage.morphology import h_minima
    from skimage.transform import downscale_local_mean
    f = int(max(1, min(8, seed_sigma // 2)))
    small = downscale_local_mean(edge_img, (f, f)) if f > 1 else edge_img
    sm = gaussian(small, sigma=max(seed_sigma / f, 0.5), preserve_range=True)
    lo, hi = np.percentile(sm, [1, 99])
    if not hi > lo:
        return np.zeros(edge_img.shape, dtype=np.int32), 0
    markers_small, n = ndi.label(h_minima(sm, min_depth * (hi - lo)))
    if f > 1:
        markers = np.repeat(np.repeat(markers_small, f, axis=0), f, axis=1)[:edge_img.shape[0], :edge_img.shape[1]]
        if markers.shape != edge_img.shape:      # image size not a multiple of f
            markers = np.pad(markers, [(0, edge_img.shape[0] - markers.shape[0]), (0, edge_img.shape[1] - markers.shape[1])])
    else:
        markers = markers_small
    return markers.astype(np.int32), int(n)


def merge_regions_without_junction(labels, edge_smooth, markers, junction_ratio=0.4, max_area=160000,
                                   band_px=2, min_boundary_px=5):
    """
    Junction-only mode deliberately starts with too many regions. Here every pair of touching
    regions is tested: the edge intensity along their shared boundary is compared with the level at
    the cell centres (0) and with a clear junction of the same image (1 = 90th percentile of all
    boundaries, weighted by length). Boundaries below `junction_ratio` are removed, weakest first,
    never creating a region larger than `max_area`. Returns (labels, n_merged).
    """
    n = int(labels.max())
    if n < 2:
        return labels, 0
    E = ndi.maximum_filter(edge_smooth, size=2 * band_px + 1) if band_px > 0 else edge_smooth
    lo, hi, counts, means = _touching_pairs(labels, E)
    ok = counts >= min_boundary_px
    if not ok.any():
        return labels, 0
    baseline = float(np.median(edge_smooth[markers > 0])) if (markers > 0).any() else float(np.percentile(edge_smooth, 5))
    order = np.argsort(means[ok])
    cw = np.cumsum(counts[ok][order])
    reference = float(means[ok][order][min(np.searchsorted(cw, 0.9 * cw[-1]), len(order) - 1)])
    if not reference > baseline:
        return labels, 0
    contrast = (means - baseline) / (reference - baseline)

    area = np.bincount(labels.ravel(), minlength=n + 1).astype(float)
    parent = np.arange(n + 1)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    merged = 0
    for k in np.argsort(contrast):
        if contrast[k] >= junction_ratio:
            break
        if not ok[k]:
            continue
        a, b = find(int(lo[k])), find(int(hi[k]))
        if a == b or area[a] + area[b] > max_area:
            continue
        root, other = (a, b) if a < b else (b, a)
        parent[other] = root
        area[root] += area[other]
        merged += 1
    if not merged:
        return labels, 0
    mapping = np.array([find(i) for i in range(n + 1)])
    return mapping[labels].astype(labels.dtype), merged


def enclosed_by_junctions_mask(edge_smooth, edge_threshold=None, edge_threshold_percentile=20,
                               close_rad=7, min_hole_size=2000):
    """Foreground for sparse scenes without nuclei: everything enclosed by junction signal."""
    if edge_threshold is None:
        edge_threshold = np.percentile(edge_smooth, edge_threshold_percentile)
    junction = ndi.binary_closing(edge_smooth > edge_threshold, _struct(max(1, close_rad), edge_smooth.ndim))
    return ndi.binary_fill_holes(junction)


def segment_cells_from_junctions(
    edge_img, min_cells_confluent=0,
    seed_sigma=15.0, seed_min_depth=0.02, junction_merge_ratio=0.4, max_cell_area=160000,
    edge_sigma=3.0, edge_mode="intensity", min_label_size=2000, multinuc_band_px=2,
    clear_border_labels=True, clear_border_buffer=10,
    nonconfluent_enabled=True, sparse_policy="confluent",
    edge_threshold=None, edge_threshold_percentile=20.0, nuc_dilation_rad=7, min_hole_size=2000,
):
    """Returns (cell_labels, n_cells, info); info: n_seeds, n_merged, confluent, skipped, adjacency."""
    info = dict(n_seeds=0, n_merged=0, confluent=True, skipped=False, n_regions=0)
    empty = np.zeros(edge_img.shape, dtype=np.int32)
    if edge_img.ndim != 2:
        raise ValueError("finding cells from junctions alone needs 2D images (Z handling 'mip' or 'per_slice')")
    markers, n_seeds = seeds_from_junctions(edge_img, seed_sigma, seed_min_depth)
    info["n_seeds"] = n_seeds
    if n_seeds == 0:
        info["adjacency"] = cell_adjacency(empty)
        return empty, 0, info

    edge_smooth = gaussian(edge_img, sigma=edge_sigma, preserve_range=True)
    cell_labels = watershed(_surface(edge_smooth, edge_mode, edge_sigma), markers=markers)
    cell_labels, info["n_merged"] = merge_regions_without_junction(
        cell_labels, edge_smooth, markers, junction_ratio=junction_merge_ratio, max_area=max_cell_area,
        band_px=multinuc_band_px)
    from skimage.segmentation import relabel_sequential
    cell_labels = relabel_sequential(cell_labels)[0]
    info["n_regions"] = int(cell_labels.max())

    info["confluent"] = bool(info["n_regions"] >= min_cells_confluent)
    if not info["confluent"]:
        if nonconfluent_enabled:
            mask = enclosed_by_junctions_mask(edge_smooth, edge_threshold, edge_threshold_percentile,
                                              nuc_dilation_rad, min_hole_size)
            cell_labels = np.where(mask, cell_labels, 0)
        elif sparse_policy == "skip":
            info["skipped"] = True
            info["adjacency"] = cell_adjacency(empty)
            return empty, 0, info

    cell_labels, n_cells = _finish_cells(cell_labels, edge_smooth, min_label_size, clear_border_labels,
                                         clear_border_buffer, info)
    return cell_labels, n_cells, info


def marker_controlled_watershed(nuclei_img, edge_img, p):
    """
    Full segmentation of one 2D/3D image (pair). All sizes in `p` are in PIXELS.
    `nuclei_img` may be None when p["seed_source"] == "junctions".

    Returns dict(nuclei_mask, nuclei_labels, cell_labels, n_nuclei, n_cells, confluent, skipped,
                 n_merged_pairs, merged_pairs, n_multinucleated, n_seeds, stats, adjacency)
    stats = {"nuclei": DataFrame, "cells": DataFrame, "raw_object_areas": array} (2D only)
    """
    p = {**DEFAULT_SEG_PARAMS, **p}
    if p["seed_source"] == "junctions":
        cell_labels, n_cells, info = segment_cells_from_junctions(
            edge_img, min_cells_confluent=p["min_nuclei_confluent"],
            seed_sigma=p["seed_sigma"], seed_min_depth=p["seed_min_depth"],
            junction_merge_ratio=p["junction_merge_ratio"], max_cell_area=p["max_cell_area"],
            edge_sigma=p["edge_sigma"], edge_mode=p["edge_mode"], min_label_size=p["min_label_size"],
            multinuc_band_px=p["multinuc_band_px"],
            clear_border_labels=p["clear_border_labels"], clear_border_buffer=p["clear_border_buffer"],
            nonconfluent_enabled=p["nonconfluent_enabled"], sparse_policy=p["sparse_policy"],
            edge_threshold=p["edge_threshold"], edge_threshold_percentile=p["edge_threshold_percentile"],
            nuc_dilation_rad=p["nuc_dilation_rad"], min_hole_size=p["min_hole_size"],
        )
        out = dict(nuclei_mask=None, nuclei_labels=None, cell_labels=cell_labels, n_nuclei=np.nan, n_cells=n_cells,
                   confluent=info["confluent"], skipped=info["skipped"], n_merged_pairs=info["n_merged"],
                   merged_pairs=[], n_multinucleated=0, n_seeds=info["n_seeds"], n_regions=info["n_regions"],
                   adjacency=info["adjacency"], stats={"raw_object_areas": np.zeros(0, np.int64)})
        try:
            out["stats"].update(object_statistics(None, cell_labels, None))
        except Exception as e:
            print(f"⚠ size statistics failed: {e}", flush=True)
        return out

    diag = {}
    nuclei_mask, nuclei_labels, n_nuclei = segment_nuclei(
        nuclei_img, min_distance=p["min_distance"], nuclei_sigma=p["nuclei_sigma"],
        min_nucleus_size=p["min_nucleus_size"], split_touching_nuclei=p["split_touching_nuclei"], diag=diag,
    )
    individual_nuclei = nuclei_labels
    confluent = n_nuclei >= p["min_nuclei_confluent"]
    out = dict(nuclei_mask=nuclei_mask, nuclei_labels=nuclei_labels, n_nuclei=n_nuclei,
               confluent=bool(confluent), skipped=False, n_merged_pairs=0, merged_pairs=[],
               n_multinucleated=0, n_seeds=n_nuclei, adjacency=cell_adjacency(np.zeros((1, 1), np.int32)),
               stats={"raw_object_areas": diag.get("raw_object_areas", np.zeros(0, np.int64))})

    if n_nuclei == 0 or (not confluent and not p["nonconfluent_enabled"] and p["sparse_policy"] == "skip"):
        out.update(cell_labels=np.zeros_like(nuclei_labels), n_cells=0, skipped=n_nuclei > 0)
        return out

    cell_labels, nuclei_labels, n_cells, merge_info = segment_cells_from_nuclei(
        edge_img, nuclei_labels, nuclei_mask, confluent,
        edge_sigma=p["edge_sigma"], edge_mode=p["edge_mode"], min_label_size=p["min_label_size"],
        clear_border_labels=p["clear_border_labels"], clear_border_buffer=p["clear_border_buffer"],
        nonconfluent_enabled=p["nonconfluent_enabled"], foreground_mask=p["foreground_mask"],
        edge_threshold=p["edge_threshold"], edge_threshold_percentile=p["edge_threshold_percentile"],
        nuc_dilation_rad=p["nuc_dilation_rad"], nuc_dilation_iterations=p["nuc_dilation_iterations"],
        min_hole_size=p["min_hole_size"],
        merge_multinucleated=p["merge_multinucleated"], multinuc_max_distance=p["multinuc_max_distance"],
        multinuc_junction_ratio=p["multinuc_junction_ratio"], multinuc_max_nuclei=p["multinuc_max_nuclei"],
        multinuc_band_px=p["multinuc_band_px"],
    )
    surviving = set(np.unique(cell_labels).tolist())
    npl = {l: c for l, c in merge_info["nuclei_per_label"].items() if l in surviving}
    out.update(cell_labels=cell_labels, nuclei_labels=nuclei_labels, n_cells=n_cells,
               n_merged_pairs=merge_info["n_merged_pairs"], merged_pairs=merge_info["pairs"],
               n_multinucleated=len(npl), adjacency=merge_info["adjacency"])
    try:
        out["stats"].update(object_statistics(individual_nuclei, cell_labels, npl))
    except Exception as e:  # statistics must never break a segmentation
        print(f"⚠ size statistics failed: {e}", flush=True)
    return out


def _add_metric_columns(stats, adjacency, ps):
    """Add µm / µm² columns next to the pixel columns (ps = pixel size in µm, may be None)."""
    if not ps:
        return
    for t in (stats.get("nuclei"), stats.get("cells")):
        if t is None or not len(t):
            continue
        t["pixel_size_um"] = ps
        t["area_um2"] = (t["area_px"] * ps * ps).round(3)
        for c in ("equivalent_diameter", "major_axis", "minor_axis", "nearest_neighbour"):
            if f"{c}_px" in t:
                t[f"{c}_um"] = (t[f"{c}_px"] * ps).round(3)
    if adjacency is not None and len(adjacency):
        adjacency["contact_um"] = (adjacency["contact_px"] * ps).round(3)


# -------------------------------------------------
# Per-scene worker (top-level for ProcessPoolExecutor)
# -------------------------------------------------
def segment_scene(file_str, scene, dataset_stem, seg_dir_str, qc_dir_str, p,
                  background_path_nuclei=None, background_path_edge=None, adj_dir_str=None):
    """Returns a summary dict for this scene (never raises)."""
    from ..sizes import resolve_sizes, report_text
    from ..io_utils import scene_geometry
    p = {**DEFAULT_SEG_PARAMS, **p}
    seg_dir = Path(seg_dir_str)
    stem = scene_stem(dataset_stem, scene)
    cell_path = seg_dir / f"{stem}_cell_labels.tif"
    nuclei_labels_path = seg_dir / f"{stem}_nuclei_labels.tif"
    nuclei_mask_path = seg_dir / f"{stem}_nuclei_mask.tif"
    qc_path = Path(qc_dir_str) / f"{stem}_qc.png" if qc_dir_str else None
    use_nuclei = p["seed_source"] != "junctions"

    summary = dict(dataset=dataset_stem, file=Path(file_str).name, scene=scene, scene_stem=stem,
                   seed_source=p["seed_source"], pixel_size_um=np.nan,
                   n_nuclei=np.nan, n_cells=np.nan, n_multinucleated=np.nan, confluent=None,
                   status="", message="", stats=None, adjacency=None, sizes_text=None, sizes_px=None)

    if cell_path.exists() and not p["overwrite_segmentation"]:
        summary.update(status="skipped_exists", message=f"↷ Skipped (exists): {cell_path.name}")
        return summary

    try:
        # ---- sizes: µm settings -> pixels for THIS image ----
        meta_ps, shape = scene_geometry(file_str, scene)
        ps = float(p["pixel_size_um"]) if p["pixel_size_um"] else meta_ps
        px, report = resolve_sizes(p, ps, shape)
        p = {**p, **{k: v for k, v in px.items() if k != "pixel_size_um"}}
        summary.update(pixel_size_um=ps if ps else np.nan, sizes_text=report_text(report, ps, indent="    "),
                       sizes_px={k: v for k, v in px.items()})

        ff_nuc = load_and_normalize_background(background_path_nuclei) if (background_path_nuclei and use_nuclei) else None
        ff_edge = load_and_normalize_background(background_path_edge) if background_path_edge else None

        def load_pair(mode, z=None):
            edge = load_channel(file_str, scene, p["edge_channel"], mode, z, p["timepoint"])
            if ff_edge is not None:
                edge = apply_flatfield_correction(edge, ff_edge)
            if not use_nuclei:
                return None, edge
            nuc = load_channel(file_str, scene, p["nuclei_channel"], mode, z, p["timepoint"])
            if ff_nuc is not None:
                nuc = apply_flatfield_correction(nuc, ff_nuc)
            nuc, edge = crop_to_common_shape(nuc, edge)
            return nuc, edge

        adjacency = None
        if p["seg_mode"] in ("mip", "3d"):
            nuclei_img, edge_img = load_pair(p["seg_mode"])
            r = marker_controlled_watershed(nuclei_img, edge_img, p)
            nuclei_mask, nuclei_labels, cell_labels = r["nuclei_mask"], r["nuclei_labels"], r["cell_labels"]
            n_nuclei, n_cells, confluent, skipped = r["n_nuclei"], r["n_cells"], r["confluent"], r["skipped"]
            n_multi, merged_pairs, stats, adjacency = r["n_multinucleated"], r["merged_pairs"], r["stats"], r["adjacency"]
            n_seeds, n_merged = r.get("n_seeds", 0), r["n_merged_pairs"]

        elif p["seg_mode"] == "per_slice":
            masks, nlabs, clabs = [], [], []
            max_id, n_nuclei, n_cells, confl, skipped = 0, 0, 0, [], False
            n_multi, merged_pairs, stats = 0, [], None   # no per-object statistics / QC marks / adjacency per slice
            n_seeds = n_merged = 0
            for z in range(n_z(file_str, scene)):
                nuc, edge = load_pair("per_slice", z)
                r = marker_controlled_watershed(nuc, edge, p)
                clabs.append(np.where(r["cell_labels"] > 0, r["cell_labels"] + max_id, 0))
                if use_nuclei:
                    nlabs.append(np.where(r["nuclei_labels"] > 0, r["nuclei_labels"] + max_id, 0))
                    masks.append(r["nuclei_mask"])
                    max_id += r["n_nuclei"]
                    n_nuclei += r["n_nuclei"]
                else:
                    max_id += int(r["cell_labels"].max())
                n_cells += r["n_cells"]
                n_multi += r["n_multinucleated"]
                n_seeds += r.get("n_seeds", 0)
                n_merged += r["n_merged_pairs"]
                confl.append(r["confluent"])
            if len({l.shape for l in clabs}) > 1:
                clabs = list(crop_to_common_shape(*clabs))
                if use_nuclei:
                    masks, nlabs = (list(crop_to_common_shape(*x)) for x in (masks, nlabs))
            cell_labels = np.stack(clabs)
            nuclei_mask, nuclei_labels = (np.stack(masks), np.stack(nlabs)) if use_nuclei else (None, None)
            if not use_nuclei:
                n_nuclei = np.nan
            confluent = all(confl)
            nuclei_img, edge_img = load_pair("mip")  # for QC display
        else:
            raise ValueError(f"Unknown seg_mode: {p['seg_mode']}")

        summary.update(n_nuclei=n_nuclei, n_cells=n_cells, n_multinucleated=n_multi, confluent=confluent)
        if stats and cell_labels.ndim == 2:
            _add_metric_columns(stats, adjacency, ps)
            for t in (stats.get("nuclei"), stats.get("cells")):
                if t is not None and len(t):
                    t.insert(0, "scene", scene)
            summary["stats"] = stats

        if skipped:
            what = f"{n_nuclei} nuclei" if use_nuclei else "too few cells"
            summary.update(status="skipped_sparse",
                           message=f"↷ Skipped non-confluent scene ({what} < {p['min_nuclei_confluent']}): {scene}")
            return summary

        seg_dir.mkdir(parents=True, exist_ok=True)
        max_label = int(max(cell_labels.max(), nuclei_labels.max() if use_nuclei else 0))
        dtype = np.uint16 if max_label < 65535 else np.uint32
        tifffile.imwrite(cell_path, cell_labels.astype(dtype), compression="zlib")
        if use_nuclei:
            tifffile.imwrite(nuclei_labels_path, nuclei_labels.astype(dtype), compression="zlib")
            if p["save_nuclei_mask"]:
                tifffile.imwrite(nuclei_mask_path, nuclei_mask.astype(np.uint8) * 255, compression="zlib")
        else:
            for stale in (nuclei_labels_path, nuclei_mask_path):   # from an earlier run with nuclei
                if stale.exists():
                    stale.unlink()

        # ---- adjacency: which cells touch ----
        if p["save_adjacency"] and adjacency is not None and cell_labels.ndim == 2:
            adjacency.insert(0, "scene", scene)
            summary["adjacency"] = adjacency
            if adj_dir_str:
                from scipy import sparse
                Path(adj_dir_str).mkdir(parents=True, exist_ok=True)
                sparse.save_npz(Path(adj_dir_str) / f"{stem}_adjacency_matrix.npz",
                                adjacency_matrix(adjacency, max_label))

        tag = "confluent" if confluent else ("non-confluent, masked" if p["nonconfluent_enabled"]
                                             else "non-confluent, unmasked")
        if use_nuclei:
            msg = f"✓ Segmented {scene}: {n_nuclei} nuclei, {n_cells} cells ({tag})"
            if n_multi:
                msg += f", {n_multi} multi-nucleated"
        else:
            msg = (f"✓ Segmented {scene} from junctions: {n_cells} cells ({tag}; {n_seeds} starting points, "
                   f"{n_merged} merged)")

        if p["save_qc_plot"] and qc_path is not None:
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            cl2d = cell_labels if cell_labels.ndim == 2 else cell_labels.max(axis=0)
            if edge_img.ndim == 3:
                edge_img = edge_img.max(axis=0)
                nuclei_img = nuclei_img.max(axis=0) if nuclei_img is not None else None
            if nuclei_img is not None:
                a, b, c = crop_to_common_shape(nuclei_img, edge_img, cl2d)
            else:
                a = None
                b, c = crop_to_common_shape(edge_img, cl2d)
            _save_qc_plot(a, b, c, qc_path, title_extra=tag, dpi=p["qc_dpi"],
                          merged_pairs=merged_pairs if cell_labels.ndim == 2 else None)
            msg += " + QC"

        summary.update(status="ok", message=msg)
        return summary

    except Exception as e:
        summary.update(status="error", message=f"✗ Error segmenting {scene}: {type(e).__name__}: {e}")
        return summary


def _save_qc_plot(nuclei_img, edge_img, cell_labels_2d, save_path, title_extra="", dpi=100, merged_pairs=None):
    def norm(x):
        vmin, vmax = np.percentile(x, [0.5, 99.5])
        return np.clip((x - vmin) / (vmax - vmin + 1e-8), 0, 1)

    en = norm(edge_img)
    nn = norm(nuclei_img) if nuclei_img is not None else np.zeros_like(en)
    composite = np.clip(np.stack([nn + en, nn, nn + en], axis=-1), 0, 1).astype(np.float32)
    overlay = mark_boundaries(composite, cell_labels_2d, color=(1, 1, 0), mode="thick")
    n = len(np.unique(cell_labels_2d)) - 1

    try:
        from cmap import Colormap
        lab_cmap = Colormap("glasbey:glasbey").to_mpl()
    except Exception:
        lab_cmap = "nipy_spectral"

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(composite)
    ax[0].set_title("Nuclei (white) / Edge marker (magenta)" if nuclei_img is not None else "Edge marker (no nuclei channel)")
    ax[1].imshow(overlay); ax[1].set_title(f"Cell boundaries ({n} cells) {title_extra}")
    if merged_pairs:
        # cyan link between the nuclei of cells that were kept together (no junction between them)
        shown = 0
        for y0, x0, y1, x1 in merged_pairs:
            if cell_labels_2d[int(round(y0)) % cell_labels_2d.shape[0], int(round(x0)) % cell_labels_2d.shape[1]] == 0:
                continue  # that cell was removed afterwards (size / border filter)
            ax[1].plot([x0, x1], [y0, y1], color="cyan", lw=1.4, marker="o", ms=2.5)
            shown += 1
        if shown:
            ax[1].set_title(f"Cell boundaries ({n} cells) {title_extra} · cyan = {shown} multi-nucleated")
        ax[1].set_xlim(-0.5, cell_labels_2d.shape[1] - 0.5); ax[1].set_ylim(cell_labels_2d.shape[0] - 0.5, -0.5)
    ax[2].imshow(np.zeros_like(cell_labels_2d), cmap="gray", vmin=0, vmax=1)
    ax[2].imshow(np.ma.masked_equal(cell_labels_2d, 0), cmap=lab_cmap, interpolation="nearest"); ax[2].set_title(f"Cell labels ({n} cells)")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()


# -------------------------------------------------
# Summaries
# -------------------------------------------------
def write_segmentation_summary(summaries, seg_dir, dataset_stem):
    """Merge new per-scene summaries into <seg_dir>/<stem>_segmentation_summary.csv."""
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    path = seg_dir / f"{dataset_stem}_segmentation_summary.csv"
    drop = ("stats", "adjacency", "sizes_text", "sizes_px")
    rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if k not in drop}
        for k, v in (s.get("sizes_px") or {}).items():      # the pixel values actually used for this image
            if k != "pixel_size_um":
                row[f"used_{k}" + ("" if k == "min_nuclei_confluent" else "_px")] = v
        rows.append(row)
    new = pd.DataFrame(rows)
    # rows for skipped-because-exists scenes carry no numbers -> keep the old rows for them
    new = new[new["status"] != "skipped_exists"]
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["scene"].isin(new["scene"])]
        new = pd.concat([old, new], ignore_index=True)
    new.drop(columns=["message"], errors="ignore").to_csv(path, index=False)
    return path


STATS_FILES = {"nuclei": "nuclei_statistics", "cells": "cell_statistics", "raw_object_areas": "raw_object_areas"}


def write_object_statistics(summaries, seg_dir, dataset_stem):
    """Write/merge the per-object tables used by the size-statistics plots:
       <stem>_nuclei_statistics.csv, <stem>_cell_statistics.csv, <stem>_raw_object_areas.csv"""
    seg_dir = Path(seg_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    new = {k: [] for k in STATS_FILES}
    for s in summaries:
        st = s.get("stats")
        if not st:
            continue
        for k in ("nuclei", "cells"):
            if st.get(k) is not None and len(st[k]):
                new[k].append(st[k])
        raw = st.get("raw_object_areas")
        if raw is not None and len(raw):
            d = pd.DataFrame({"scene": s["scene"], "area_px": raw})
            ps = s.get("pixel_size_um")
            if ps and ps == ps:
                d["area_um2"] = (d["area_px"] * ps * ps).round(3)
            new["raw_object_areas"].append(d)
    for k, parts in new.items():
        if not parts:
            continue
        df = pd.concat(parts, ignore_index=True)
        path = seg_dir / f"{dataset_stem}_{STATS_FILES[k]}.csv"
        if path.exists():
            try:
                old = pd.read_csv(path)
                df = pd.concat([old[~old["scene"].isin(df["scene"].unique())], df], ignore_index=True)
            except Exception:
                pass
        df.to_csv(path, index=False)


def write_adjacency(summaries, adj_dir, dataset_stem):
    """Write/merge <adj_dir>/<stem>_cell_adjacency.csv: one row per pair of touching cells
    (scene, cell_id_a, cell_id_b, contact_px, contact_um, junction_intensity)."""
    parts = [s["adjacency"] for s in summaries if s.get("adjacency") is not None]
    done = [s["scene"] for s in summaries if s.get("status") == "ok"]
    if not parts and not done:
        return None
    adj_dir = Path(adj_dir)
    adj_dir.mkdir(parents=True, exist_ok=True)
    path = adj_dir / f"{dataset_stem}_cell_adjacency.csv"
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["scene", "cell_id_a", "cell_id_b", "contact_px"])
    if path.exists():
        try:
            old = pd.read_csv(path)
            df = pd.concat([old[~old["scene"].isin(done)], df], ignore_index=True)
        except Exception:
            pass
    df.to_csv(path, index=False)
    return path


# -------------------------------------------------
# Convenience: one file, parallel over scenes
# -------------------------------------------------
def segment_file(file_path, scenes, dataset_stem, seg_dir, qc_dir, params, n_workers=None,
                 background_path_nuclei=None, background_path_edge=None):
    if not n_workers:
        n_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
    summaries = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(segment_scene, str(file_path), s, dataset_stem, str(seg_dir),
                          str(qc_dir) if qc_dir else None, params,
                          background_path_nuclei, background_path_edge) for s in scenes]
        for f in as_completed(futs):
            r = f.result()
            print(r["message"], flush=True)
            summaries.append(r)
    write_segmentation_summary(summaries, seg_dir, dataset_stem)
    write_object_statistics(summaries, seg_dir, dataset_stem)
    return summaries


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m cell_analyzer.stages.segmentation <image_file> <output_dir> [n_workers]")
        print("For folders, filters and all parameters use the GUI (python -m cell_analyzer) or python -m cell_analyzer.pipeline.")
        sys.exit(1)
    from ..io_utils import list_scenes, file_stem
    f = Path(sys.argv[1])
    out = Path(sys.argv[2])
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else None
    segment_file(f, list_scenes(f), file_stem(f), out / "segmentation", out / "segmentation_qc",
                 DEFAULT_SEG_PARAMS, n_workers=nw)
