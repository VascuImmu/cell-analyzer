"""
Measure per-cell intensity, shape and edge-boundary descriptors from the
watershed output (stages/segmentation.py) and the original raw data (any
format AICSImage reads).

Per cell (cell and nucleus share the label ID):
  * shape: area, perimeter, eccentricity, orientation, axes, solidity,
    extent, equivalent diameter, centroid, ruggedness
  * per-channel mean/median/std/sum in the cell (and nucleus: nuc_*)
  * boundary-ring intensity (edge_mean/std/cv_*) for junction channels
  * n_nuclei: nuclei in the cell (>1 for multi-nucleated cells kept together by the segmentation;
    their nuc_* values then describe all nuclei of the cell together)
  * n_neighbours / neighbour_contact_um: number of cells this cell touches and the total length of
    those contacts (from the contact-based adjacency of the segmentation)
  * identifiers: dataset, file, scene, well_row / well_column / well_id
    (NaN when there is no well information, e.g. plain .czi files),
    and the scene's confluence state from the segmentation summary.

Needs 2D label images (segmentation Z handling = 'mip').

Outputs per dataset:
  <stem>_cell_measurements.csv   -- one row per cell
  <stem>_cell_measurements.pkl   -- same table + boundary intensity arrays
"""

import sys
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

import warnings
import numpy as np

# skimage >= 0.26 renames some arguments/properties; keep the log readable
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter1d, binary_erosion
from skimage.measure import regionprops, find_contours

from .flatfield import load_and_normalize_background, apply_flatfield_correction
from ..io_utils import load_mip_stack, scene_stem, parse_well

DEFAULT_MEAS_PARAMS = dict(
    channel_names="",
    edge_channels="1, 3",
    boundary_band_px=1,
    size_units="px", typical_nucleus_diameter_um=10.0, typical_cell_area_um2=1000.0,
    ruggedness_smooth_frac=0.02,
    ruggedness_min_sigma=1.5,
    measure_nuclei=True,
    pixel_size_um=None,
    well_regex=r"(?P<row>[A-Za-z]{1,2})/(?P<col>\d{1,2})(?:_|$)",
    well_regex_fallback=True,
    timepoint=0,
)


def meas_params_from_config(cfg):
    return {k: cfg[k] for k in DEFAULT_MEAS_PARAMS}


def compute_ruggedness(region_mask, smooth_sigma_frac=0.02, min_sigma=1.5):
    padded = np.pad(region_mask, 1, mode="constant", constant_values=0)
    contours = find_contours(padded.astype(float), level=0.5)
    if not contours:
        return np.nan
    contour = max(contours, key=len)

    def perim(c):
        d = np.diff(np.vstack([c, c[0]]), axis=0)
        return np.sum(np.sqrt((d ** 2).sum(axis=1)))

    sigma = max(min_sigma, len(contour) * smooth_sigma_frac)
    smoothed = np.stack([gaussian_filter1d(contour[:, 0], sigma, mode="wrap"),
                         gaussian_filter1d(contour[:, 1], sigma, mode="wrap")], axis=1)
    ps = perim(smoothed)
    return float(perim(contour) / ps) if ps > 0 else np.nan


def sample_boundary_intensity(region_mask, channel_crop, band_px=1):
    ring = region_mask & ~binary_erosion(region_mask, iterations=band_px)
    if not ring.any():
        ring = region_mask
    return channel_crop[ring].astype(np.float32)


def crop_last2_to_shape(*arrays):
    hw = [a.shape[-2:] for a in arrays]
    if len(set(hw)) == 1:
        return arrays
    mh, mw = min(h for h, _ in hw), min(w for _, w in hw)
    print(f"⚠ Shape mismatch {sorted(set(hw))} -> cropping to ({mh}, {mw})", flush=True)
    return tuple(a[..., :mh, :mw] for a in arrays)


def _parse_ints(v):
    if isinstance(v, (list, tuple)):
        return [int(x) for x in v]
    return [int(x) for x in str(v).split(",") if x.strip()]


def _parse_names(v):
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [x.strip() for x in str(v or "").split(",") if x.strip()]


# -------------------------------------------------
# Per-scene worker
# -------------------------------------------------
def measure_scene(file_str, scene, dataset_stem, seg_dir_str, p, background_paths=None, scene_info=None):
    p = {**DEFAULT_MEAS_PARAMS, **p}
    seg_dir = Path(seg_dir_str)
    stem = scene_stem(dataset_stem, scene)
    cell_path = seg_dir / f"{stem}_cell_labels.tif"
    nuclei_path = seg_dir / f"{stem}_nuclei_labels.tif"
    scene_info = scene_info or {}

    if not cell_path.exists():
        why = " (skipped as non-confluent)" if scene_info.get("status") == "skipped_sparse" else ""
        return [], {}, f"↷ No cell labels for {scene}{why}"

    cell_labels = tifffile.imread(cell_path)
    if cell_labels.ndim != 2:
        return [], {}, f"✗ {scene}: labels are not 2D {cell_labels.shape}; use segmentation mode 'mip'."

    nuclei_labels = tifffile.imread(nuclei_path) if (p["measure_nuclei"] and nuclei_path.exists()) else None

    stack, pixel_size, file_ch_names = load_mip_stack(file_str, scene, p["timepoint"])
    if p["pixel_size_um"]:
        pixel_size = float(p["pixel_size_um"])
    px = pixel_size if pixel_size else np.nan
    n_channels = stack.shape[0]

    names = _parse_names(p["channel_names"]) or file_ch_names
    names = [str(n) for n in names][:n_channels] + [f"ch{i}" for i in range(len(names), n_channels)]
    names = [n if names.count(n) == 1 else f"{n}_ch{i}" for i, n in enumerate(names)]  # unique

    if nuclei_labels is not None:
        stack, cell_labels, nuclei_labels = crop_last2_to_shape(stack, cell_labels, nuclei_labels)
    else:
        stack, cell_labels = crop_last2_to_shape(stack, cell_labels)

    for ch, path in (background_paths or {}).items():
        if path and int(ch) < n_channels:
            stack[int(ch)] = apply_flatfield_correction(stack[int(ch)], load_and_normalize_background(path))

    well_row, well_col = parse_well(scene, Path(file_str).name, p["well_regex"], p["well_regex_fallback"])
    well_id = f"{well_row}{well_col}" if well_row else None
    edge_channels = [c for c in _parse_ints(p["edge_channels"]) if c < n_channels]

    nuc_props = {q.label: q for q in regionprops(nuclei_labels)} if nuclei_labels is not None else {}

    from ..sizes import resolve_one
    band_px = resolve_one(p, "boundary_band_px", pixel_size)      # µm setting -> pixels for this image
    nuclei_per_cell = scene_info.get("nuclei_per_cell") or {}
    neighbours = scene_info.get("neighbours")                       # {cell_id: (n, contact_px)} or None
    no_nuclei = scene_info.get("seed_source") == "junctions"
    rows, extras = [], {}
    for prop in regionprops(cell_labels):
        lid = int(prop.label)
        r0, c0, r1, c1 = prop.bbox
        m = prop.image
        row = {
            "dataset": dataset_stem,
            "file": Path(file_str).name,
            "scene": scene,
            "well_row": well_row,
            "well_column": well_col,
            "well_id": well_id,
            "scene_confluent": scene_info.get("confluent"),
            "scene_n_nuclei": scene_info.get("n_nuclei"),
            "cell_id": lid,
            "n_nuclei": np.nan if no_nuclei else nuclei_per_cell.get(lid, 1),
            "n_neighbours": neighbours.get(lid, (0, 0))[0] if neighbours is not None else np.nan,
            "neighbour_contact_um": neighbours.get(lid, (0, 0))[1] * px if neighbours is not None else np.nan,
            "pixel_size_um": px,
            "area_px": prop.area,
            "area_um2": prop.area * px ** 2,
            "perimeter_px": prop.perimeter,
            "eccentricity": prop.eccentricity,
            "orientation_rad": prop.orientation,
            "orientation_deg": float(np.degrees(prop.orientation)),
            "major_axis_length_px": prop.major_axis_length,
            "minor_axis_length_px": prop.minor_axis_length,
            "major_axis_length_um": prop.major_axis_length * px,
            "minor_axis_length_um": prop.minor_axis_length * px,
            "solidity": prop.solidity,
            "extent": prop.extent,
            "equivalent_diameter_px": prop.equivalent_diameter,
            "centroid_y": prop.centroid[0],
            "centroid_x": prop.centroid[1],
            "ruggedness": compute_ruggedness(m, p["ruggedness_smooth_frac"], p["ruggedness_min_sigma"]),
        }
        for c in range(n_channels):
            v = stack[c, r0:r1, c0:c1][m]
            row[f"mean_{names[c]}"] = float(v.mean())
            row[f"median_{names[c]}"] = float(np.median(v))
            row[f"std_{names[c]}"] = float(v.std())
            row[f"sum_{names[c]}"] = float(v.sum())

        if nuclei_labels is not None:
            nq = nuc_props.get(lid)
            if nq is not None:
                a0, b0, a1, b1 = nq.bbox
                row["nuc_area_px"] = nq.area
                row["nuc_area_um2"] = nq.area * px ** 2
                row["nuc_eccentricity"] = nq.eccentricity
                row["nuc_orientation_rad"] = nq.orientation
                for c in range(n_channels):
                    v = stack[c, a0:a1, b0:b1][nq.image]
                    row[f"nuc_mean_{names[c]}"] = float(v.mean())
                    row[f"nuc_median_{names[c]}"] = float(np.median(v))
                    row[f"nuc_std_{names[c]}"] = float(v.std())
                    row[f"nuc_sum_{names[c]}"] = float(v.sum())
            else:
                row["nuc_area_px"] = np.nan

        cell_extra = {}
        for c in edge_channels:
            bv = sample_boundary_intensity(m, stack[c, r0:r1, c0:c1], band_px=band_px)
            mean_v = float(bv.mean()) if bv.size else np.nan
            std_v = float(bv.std()) if bv.size else np.nan
            row[f"edge_mean_{names[c]}"] = mean_v
            row[f"edge_std_{names[c]}"] = std_v
            row[f"edge_cv_{names[c]}"] = std_v / mean_v if mean_v else np.nan
            cell_extra[f"boundary_intensity_{names[c]}"] = bv

        rows.append(row)
        extras[lid] = cell_extra

    return rows, extras, f"✓ Measured {len(rows)} cells: {scene}"


def save_measurements(rows, extras_by_scene, out_dir, dataset_stem, save_pickle=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{dataset_stem}_cell_measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved {csv_path} ({len(df):,} cells)", flush=True)
    if save_pickle:
        pkl = out_dir / f"{dataset_stem}_cell_measurements.pkl"
        with open(pkl, "wb") as f:
            pickle.dump({"measurements": df, "boundary_arrays": extras_by_scene}, f)
        print(f"✓ Saved {pkl}", flush=True)
    return csv_path


def load_scene_info(seg_dir, dataset_stem, adj_dir=None):
    path = Path(seg_dir) / f"{dataset_stem}_segmentation_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    info = {r["scene"]: {"confluent": r.get("confluent"), "n_nuclei": r.get("n_nuclei"), "status": r.get("status"),
                         "seed_source": r.get("seed_source")}
            for r in df.to_dict("records")}
    # touching neighbours per cell (contact-based adjacency written by the segmentation)
    adj = Path(adj_dir) / f"{dataset_stem}_cell_adjacency.csv" if adj_dir else None
    if adj is not None and adj.exists():
        try:
            a = pd.read_csv(adj, usecols=["scene", "cell_id_a", "cell_id_b", "contact_px"])
            both = pd.concat([a.rename(columns={"cell_id_a": "cell"})[["scene", "cell", "contact_px"]],
                              a.rename(columns={"cell_id_b": "cell"})[["scene", "cell", "contact_px"]]])
            agg = both.groupby(["scene", "cell"])["contact_px"].agg(["size", "sum"])
            for scene in info:
                info[scene]["neighbours"] = {}
            for (scene, cell), row in agg.iterrows():
                info.setdefault(scene, {}).setdefault("neighbours", {})[int(cell)] = (int(row["size"]), float(row["sum"]))
        except Exception:
            pass
    # nuclei per cell (multi-nucleated cells that the segmentation kept together)
    cs = Path(seg_dir) / f"{dataset_stem}_cell_statistics.csv"
    if cs.exists():
        try:
            c = pd.read_csv(cs, usecols=["scene", "label_id", "n_nuclei"])
            for scene, g in c[c["n_nuclei"] > 1].groupby("scene"):
                info.setdefault(scene, {})["nuclei_per_cell"] = dict(zip(g["label_id"].astype(int), g["n_nuclei"].astype(int)))
        except Exception:
            pass
    return info


def measure_file(file_path, scenes, dataset_stem, seg_dir, out_dir, params, n_workers=None,
                 background_paths=None, save_pickle=True):
    if not n_workers:
        n_workers = min(4, max(1, multiprocessing.cpu_count() - 1))
    info = load_scene_info(seg_dir, dataset_stem)
    rows, extras = [], {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(measure_scene, str(file_path), s, dataset_stem, str(seg_dir), params,
                          background_paths, info.get(s)): s for s in scenes}
        for f in as_completed(futs):
            s = futs[f]
            try:
                r, e, msg = f.result()
            except Exception as err:
                print(f"✗ Error measuring {s}: {err}", flush=True)
                continue
            print(msg, flush=True)
            rows.extend(r)
            extras[s] = e
    if rows:
        return save_measurements(rows, extras, out_dir, dataset_stem, save_pickle)
    print(f"No cells measured for {dataset_stem}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m cell_analyzer.stages.measurement <image_file> <output_dir_of_that_file> [n_workers]")
        print("For folders, filters and all parameters use the GUI (python -m cell_analyzer) or python -m cell_analyzer.pipeline.")
        sys.exit(1)
    from ..io_utils import list_scenes, file_stem
    f = Path(sys.argv[1])
    out = Path(sys.argv[2])
    nw = int(sys.argv[3]) if len(sys.argv) > 3 else None
    measure_file(f, list_scenes(f), file_stem(f), out / "segmentation", out / "measurements",
                 {**DEFAULT_MEAS_PARAMS, "channel_names": "DNA, VE-Cadherin, ICAM1, PECAM1"}, n_workers=nw)
