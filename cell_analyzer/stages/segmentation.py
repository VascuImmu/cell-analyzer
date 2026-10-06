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

# Default segmentation parameters (identical to the defaults in config.py)
DEFAULT_SEG_PARAMS = dict(
    nuclei_channel=0,
    edge_channel=1,
    seg_mode="mip",
    timepoint=0,
    nuclei_sigma=1.5,
    min_nucleus_size=1100,
    min_distance=20,
    split_touching_nuclei=True,
    edge_sigma=3.0,
    edge_mode="gradient",
    min_label_size=2000,
    merge_multinucleated=True,
    multinuc_max_distance=0,
    multinuc_junction_ratio=0.5,
    multinuc_max_nuclei=2,
    multinuc_band_px=2,
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
    sparse_policy="confluent",
    overwrite_segmentation=True,
    save_nuclei_mask=True,
    save_qc_plot=True,
    qc_dpi=100,
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
        if lab.ndim != 2 or not lab.any():
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
            t["n_nuclei"] = [npl.get(int(l), 1) for l in t["label_id"]]
        out[name] = t.drop(columns=["centroid_y", "centroid_x"]).round(3)
    return out


# -------------------------------------------------
# Stage 2: cells, seeded from nuclei
# -------------------------------------------------
def segment_cells_from_nuclei(
    edge_img, nuclei_labels, nuclei_mask, confluent,
    edge_sigma=3.0, edge_mode="gradient", min_label_size=2000,
    clear_border_labels=True, clear_border_buffer=10,
    nonconfluent_enabled=True, foreground_mask="union",
    edge_threshold=165.0, edge_threshold_percentile=20.0,
    nuc_dilation_rad=7, nuc_dilation_iterations=3, min_hole_size=2000,
    merge_multinucleated=True, multinuc_max_distance=0, multinuc_junction_ratio=0.5,
    multinuc_max_nuclei=2, multinuc_band_px=2,
):
    """Returns (cell_labels, nuclei_labels, n_cells, merge_info). nuclei_labels differ from the
    input only where multi-nucleated cells were merged (their nuclei then share the cell's ID)."""
    edge_smooth = gaussian(edge_img, sigma=edge_sigma, preserve_range=True)

    mask = None
    if not confluent and nonconfluent_enabled:
        mask = build_foreground_mask(
            foreground_mask, edge_smooth, nuclei_mask,
            edge_threshold=edge_threshold, edge_threshold_percentile=edge_threshold_percentile,
            nuc_dilation_rad=nuc_dilation_rad, nuc_dilation_iterations=nuc_dilation_iterations,
            min_hole_size=min_hole_size,
        )

    if edge_mode in ("gradient", "LoG"):
        surface = np.sqrt(sum(ndi.sobel(edge_smooth, axis=a) ** 2 for a in range(edge_smooth.ndim)))
    elif edge_mode == "intensity":
        surface = edge_smooth
    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")

    cell_labels = watershed(surface, markers=nuclei_labels, mask=mask)

    merge_info = dict(n_merged_pairs=0, pairs=[], nuclei_per_label={})
    if merge_multinucleated and multinuc_max_nuclei > 1:
        cell_labels, nuclei_labels, merge_info = merge_multinucleated_cells(
            cell_labels, nuclei_labels, edge_smooth, nuclei_mask,
            max_distance=multinuc_max_distance, junction_ratio=multinuc_junction_ratio,
            max_nuclei=multinuc_max_nuclei, band_px=multinuc_band_px)

    if min_label_size > 0:
        cell_labels = remove_small_objects(cell_labels, min_size=min_label_size)
    if clear_border_labels:
        cell_labels = clear_border(cell_labels, buffer_size=clear_border_buffer)

    n_cells = int(len(np.unique(cell_labels)) - (1 if (cell_labels == 0).any() else 0))
    return cell_labels.astype(np.int32), nuclei_labels.astype(np.int32), n_cells, merge_info


def marker_controlled_watershed(nuclei_img, edge_img, p):
    """
    Full two-stage segmentation of one 2D/3D image pair.

    Returns dict(nuclei_mask, nuclei_labels, cell_labels, n_nuclei, n_cells, confluent, skipped,
                 n_merged_pairs, merged_pairs, n_multinucleated, stats)
    stats = {"nuclei": DataFrame, "cells": DataFrame, "raw_object_areas": array} (2D only)
    """
    p = {**DEFAULT_SEG_PARAMS, **p}
    diag = {}
    nuclei_mask, nuclei_labels, n_nuclei = segment_nuclei(
        nuclei_img, min_distance=p["min_distance"], nuclei_sigma=p["nuclei_sigma"],
        min_nucleus_size=p["min_nucleus_size"], split_touching_nuclei=p["split_touching_nuclei"], diag=diag,
    )
    individual_nuclei = nuclei_labels
    confluent = n_nuclei >= p["min_nuclei_confluent"]
    out = dict(nuclei_mask=nuclei_mask, nuclei_labels=nuclei_labels, n_nuclei=n_nuclei,
               confluent=bool(confluent), skipped=False, n_merged_pairs=0, merged_pairs=[],
               n_multinucleated=0, stats={"raw_object_areas": diag.get("raw_object_areas", np.zeros(0, np.int64))})

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
               n_multinucleated=len(npl))
    try:
        out["stats"].update(object_statistics(individual_nuclei, cell_labels, npl))
    except Exception as e:  # statistics must never break a segmentation
        print(f"⚠ size statistics failed: {e}", flush=True)
    return out


# -------------------------------------------------
# Per-scene worker (top-level for ProcessPoolExecutor)
# -------------------------------------------------
def segment_scene(file_str, scene, dataset_stem, seg_dir_str, qc_dir_str, p,
                  background_path_nuclei=None, background_path_edge=None):
    """Returns a summary dict for this scene (never raises)."""
    p = {**DEFAULT_SEG_PARAMS, **p}
    seg_dir = Path(seg_dir_str)
    stem = scene_stem(dataset_stem, scene)
    cell_path = seg_dir / f"{stem}_cell_labels.tif"
    nuclei_labels_path = seg_dir / f"{stem}_nuclei_labels.tif"
    nuclei_mask_path = seg_dir / f"{stem}_nuclei_mask.tif"
    qc_path = Path(qc_dir_str) / f"{stem}_qc.png" if qc_dir_str else None

    summary = dict(dataset=dataset_stem, file=Path(file_str).name, scene=scene, scene_stem=stem,
                   n_nuclei=np.nan, n_cells=np.nan, n_multinucleated=np.nan, confluent=None,
                   status="", message="", stats=None)

    if cell_path.exists() and not p["overwrite_segmentation"]:
        summary.update(status="skipped_exists", message=f"↷ Skipped (exists): {cell_path.name}")
        return summary

    try:
        ff_nuc = load_and_normalize_background(background_path_nuclei) if background_path_nuclei else None
        ff_edge = load_and_normalize_background(background_path_edge) if background_path_edge else None

        def load_pair(mode, z=None):
            nuc = load_channel(file_str, scene, p["nuclei_channel"], mode, z, p["timepoint"])
            edge = load_channel(file_str, scene, p["edge_channel"], mode, z, p["timepoint"])
            nuc, edge = crop_to_common_shape(nuc, edge)
            if ff_nuc is not None:
                nuc = apply_flatfield_correction(nuc, ff_nuc)
            if ff_edge is not None:
                edge = apply_flatfield_correction(edge, ff_edge)
            return nuc, edge

        if p["seg_mode"] in ("mip", "3d"):
            nuclei_img, edge_img = load_pair(p["seg_mode"])
            r = marker_controlled_watershed(nuclei_img, edge_img, p)
            nuclei_mask, nuclei_labels, cell_labels = r["nuclei_mask"], r["nuclei_labels"], r["cell_labels"]
            n_nuclei, n_cells, confluent, skipped = r["n_nuclei"], r["n_cells"], r["confluent"], r["skipped"]
            n_multi, merged_pairs, stats = r["n_multinucleated"], r["merged_pairs"], r["stats"]

        elif p["seg_mode"] == "per_slice":
            masks, nlabs, clabs = [], [], []
            max_id, n_nuclei, n_cells, confl, skipped = 0, 0, 0, [], False
            n_multi, merged_pairs, stats = 0, [], None   # no per-object statistics / QC marks in per-slice mode
            for z in range(n_z(file_str, scene)):
                nuc, edge = load_pair("per_slice", z)
                r = marker_controlled_watershed(nuc, edge, p)
                nlabs.append(np.where(r["nuclei_labels"] > 0, r["nuclei_labels"] + max_id, 0))
                clabs.append(np.where(r["cell_labels"] > 0, r["cell_labels"] + max_id, 0))
                masks.append(r["nuclei_mask"])
                max_id += r["n_nuclei"]
                n_nuclei += r["n_nuclei"]
                n_cells += r["n_cells"]
                n_multi += r["n_multinucleated"]
                confl.append(r["confluent"])
            if len({l.shape for l in clabs}) > 1:
                masks, nlabs, clabs = (list(crop_to_common_shape(*x)) for x in (masks, nlabs, clabs))
            nuclei_mask, nuclei_labels, cell_labels = np.stack(masks), np.stack(nlabs), np.stack(clabs)
            confluent = all(confl)
            nuclei_img, edge_img = load_pair("mip")  # for QC display
        else:
            raise ValueError(f"Unknown seg_mode: {p['seg_mode']}")

        summary.update(n_nuclei=n_nuclei, n_cells=n_cells, n_multinucleated=n_multi, confluent=confluent)
        if stats and nuclei_labels.ndim == 2:
            for t in (stats.get("nuclei"), stats.get("cells")):
                if t is not None and len(t):
                    t.insert(0, "scene", scene)
            summary["stats"] = stats

        if skipped:
            summary.update(status="skipped_sparse",
                           message=f"↷ Skipped non-confluent scene ({n_nuclei} nuclei < "
                                   f"{p['min_nuclei_confluent']}): {scene}")
            return summary

        seg_dir.mkdir(parents=True, exist_ok=True)
        max_label = int(max(cell_labels.max(), nuclei_labels.max()))
        dtype = np.uint16 if max_label < 65535 else np.uint32
        tifffile.imwrite(cell_path, cell_labels.astype(dtype), compression="zlib")
        tifffile.imwrite(nuclei_labels_path, nuclei_labels.astype(dtype), compression="zlib")
        if p["save_nuclei_mask"]:
            tifffile.imwrite(nuclei_mask_path, nuclei_mask.astype(np.uint8) * 255, compression="zlib")

        tag = "confluent" if confluent else ("non-confluent, masked" if p["nonconfluent_enabled"]
                                             else "non-confluent, unmasked")
        msg = f"✓ Segmented {scene}: {n_nuclei} nuclei, {n_cells} cells ({tag})"
        if n_multi:
            msg += f", {n_multi} multi-nucleated"

        if p["save_qc_plot"] and qc_path is not None:
            qc_path.parent.mkdir(parents=True, exist_ok=True)
            cl2d = cell_labels if cell_labels.ndim == 2 else cell_labels.max(axis=0)
            if nuclei_img.ndim == 3:
                nuclei_img, edge_img = nuclei_img.max(axis=0), edge_img.max(axis=0)
            a, b, c = crop_to_common_shape(nuclei_img, edge_img, cl2d)
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

    nn, en = norm(nuclei_img), norm(edge_img)
    composite = np.clip(np.stack([nn + en, nn, nn + en], axis=-1), 0, 1).astype(np.float32)
    overlay = mark_boundaries(composite, cell_labels_2d, color=(1, 1, 0), mode="thick")
    n = len(np.unique(cell_labels_2d)) - 1

    try:
        from cmap import Colormap
        lab_cmap = Colormap("glasbey:glasbey").to_mpl()
    except Exception:
        lab_cmap = "nipy_spectral"

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(composite); ax[0].set_title("Nuclei (white) / Edge marker (magenta)")
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
    new = pd.DataFrame([{k: v for k, v in s.items() if k != "stats"} for s in summaries])
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
            new["raw_object_areas"].append(pd.DataFrame({"scene": s["scene"], "area_px": raw}))
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
