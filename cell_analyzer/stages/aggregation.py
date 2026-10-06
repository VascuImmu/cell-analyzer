"""
cell_analyzer/stages/aggregation.py -- stage 4

Combines all per-dataset cell-measurement CSVs produced by the pipeline into
one table, optionally adding plate-map metadata and per-dataset Z-scores.

Works for any input type:
  * .lif plates -> dataset names like "..._Plate2_Sample2[_rescan]", scenes
    carry well IDs -> plate map + control-treatment Z-scores work as before.
  * .czi folders / single files -> dataset names need not match the
    Plate/Sample pattern; such datasets are still included (plate/sample =
    NaN), rescan de-duplication just doesn't apply to them. Wells (if any)
    come from the measurement table; controls can be given as a well list.

Expected layout:  <input_dir>/per_file/<dataset>/measurements/<dataset>_cell_measurements.csv
(older versions without per_file/ are found too)
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import parse_str_list, output_paths


def _norm_value(v):
    """Normalize plate-map values so 240, 240.0 and '240' all compare equal."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip()


def _dataset_dirs(input_dir, measurements_subdir, datasets_subdir="per_file"):
    """Dataset folders below input_dir: new layout (<root>/per_file/<dataset>), older versions
    (<root>/<dataset>), or input_dir pointing at the per_file folder itself."""
    input_dir = Path(input_dir)
    seen, out = set(), []
    for parent in (input_dir / datasets_subdir, input_dir):
        if not parent.is_dir():
            continue
        for d in sorted(p for p in parent.iterdir() if p.is_dir()):
            if (d / measurements_subdir).is_dir() and d.name not in seen:
                seen.add(d.name)
                out.append(d)
    return out


def find_and_resolve_scans(input_dir, measurements_subdir="measurements", dataset_regex=None,
                           keep_latest_rescan=True, datasets_subdir="per_file"):
    """Return list of dicts: plate, sample, rescan, csv_path, dataset."""
    input_dir = Path(input_dir)
    rx = re.compile(dataset_regex) if dataset_regex else None
    grouped, ungrouped = {}, []

    for d in _dataset_dirs(input_dir, measurements_subdir, datasets_subdir):
        meas_dir = d / measurements_subdir
        if not meas_dir.is_dir():
            continue
        csvs = sorted(p for p in meas_dir.glob("*_cell_measurements.csv") if not p.name.startswith("._"))
        if not csvs:
            print(f"⚠ No *_cell_measurements.csv in {meas_dir}")
            continue
        if len(csvs) > 1:
            print(f"⚠ Several measurement CSVs in {meas_dir}, using {csvs[0].name}")

        plate = sample = rescan = None
        m = rx.search(d.name) if rx else None
        if m:
            gd = m.groupdict()
            plate = int(gd["plate"]) if gd.get("plate") else None
            sample = int(gd["sample"]) if gd.get("sample") else None
            rescan = int(gd["rescan"]) if gd.get("rescan") else 0
        entry = dict(plate=plate, sample=sample, rescan=rescan, csv_path=csvs[0], dataset=d.name)

        if keep_latest_rescan and plate is not None:
            grouped.setdefault((plate, sample), []).append(entry)
        else:
            ungrouped.append(entry)

    selected = []
    for key, cands in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
        cands.sort(key=lambda c: c["rescan"])
        if len(cands) > 1:
            print(f"↷ Plate{key[0]}_Sample{key[1]}: {len(cands)} scans -> using {cands[-1]['dataset']}; "
                  f"skipping {[c['dataset'] for c in cands[:-1]]}")
        selected.append(cands[-1])
    return selected + ungrouped


def load_plate_map(path, sheet="Treatment_Map", compound_sheet="Compound_IDs",
                   compound_columns=("Source Name", "Class/Target"),
                   plate_column="Plate", treatment_column="Treatment"):
    pm = pd.read_excel(path, sheet_name=sheet)
    pm[treatment_column] = pm[treatment_column].map(_norm_value)
    if compound_sheet and compound_sheet not in pd.ExcelFile(path).sheet_names:
        print(f"  ⚠ plate map has no sheet '{compound_sheet}' -- compound information not added")
        compound_sheet = None
    if compound_sheet:
        comp = pd.read_excel(path, sheet_name=compound_sheet)
        comp[treatment_column] = comp[treatment_column].map(_norm_value)
        cols = [c for c in compound_columns if c in comp.columns]
        for c in cols:
            comp[c] = comp[c].astype(str)
        pm = pm.merge(comp[[treatment_column] + cols], how="left", on=treatment_column)
    pm["well_id"] = pm["well_row"].astype(str).str.upper() + pm["well_column"].map(lambda x: str(int(x)))
    pm[plate_column] = pd.to_numeric(pm[plate_column], errors="coerce")
    return pm


def process_one_scan(entry, plate_map, cfg):
    df = pd.read_csv(entry["csv_path"])
    treat_col = cfg["treatment_column"]

    if "well_id" not in df.columns:
        df["well_id"] = np.nan
    df["well_id"] = df["well_id"].astype("object")

    if cfg.get("exclude_nonconfluent") and "scene_confluent" in df.columns:
        before = len(df)
        df = df[df["scene_confluent"].astype(str).str.lower() != "false"]
        if before != len(df):
            print(f"  ↷ removed {before - len(df)} cells from non-confluent scenes")

    # plate-map metadata
    meta_cols = []
    if plate_map is not None:
        if entry["plate"] is None:
            print(f"  ⚠ {entry['dataset']}: no plate number parsed -- plate-map metadata not added")
        else:
            wi = plate_map[plate_map[cfg["plate_column"]] == entry["plate"]].set_index("well_id")
            meta_cols = [treat_col] + [c for c in parse_str_list(cfg["compound_columns"]) if c in wi.columns]
            for c in meta_cols:
                df[c] = df["well_id"].map(wi[c])

    # control cells
    ctrl = pd.Series(False, index=df.index)
    controls = [_norm_value(v) for v in parse_str_list(cfg["control_treatments"])]
    if treat_col in df.columns and controls:
        ctrl |= df[treat_col].isin(controls)
    wells = [w.upper() for w in parse_str_list(cfg["control_wells"])]
    if wells:
        wells = [re.sub(r"^([A-Z]+)0*(\d+)$", r"\1\2", w) for w in wells]
        ctrl |= df["well_id"].isin(wells)
    df["is_control"] = ctrl

    df.insert(0, "plate_num", entry["plate"])
    df.insert(1, "sample_num", entry["sample"])
    df.insert(2, "rescan_num", entry["rescan"])
    df.insert(3, "source_dataset", entry["dataset"])
    df.insert(4, "source_file", entry["csv_path"].name)
    return df, meta_cols


def add_zscores(df, cfg):
    """
    Z-score each requested column relative to the control cells of its reference group:
      dataset -> each input file / scan separately (original behaviour, one plate per .lif)
      plate   -> all datasets sharing a plate number (e.g. one .czi per well)
      all     -> one pooled reference over everything
    """
    ref = cfg.get("zscore_reference", "dataset")
    if ref == "dataset":
        groups = df["source_dataset"]
    elif ref == "plate":
        groups = df["plate_num"].astype(str)
    else:
        groups = pd.Series("all", index=df.index)

    for col in parse_str_list(cfg["zscore_columns"]):
        zc = f"Zscore_{col}"
        df[zc] = np.nan
        if col not in df.columns:
            print(f"  ⚠ column '{col}' not found; {zc} = NaN")
            continue
        for g, idx in df.groupby(groups).groups.items():
            sub = df.loc[idx]
            vals = sub.loc[sub["is_control"], col]
            mu, sd = vals.mean(), vals.std()
            if len(vals) > 1 and sd > 0:
                df.loc[idx, zc] = (sub[col] - mu) / sd
            else:
                print(f"  ⚠ no usable control cells for {ref} '{g}' -- {zc} = NaN there")


def aggregate_measurements(cfg, input_dir=None, output_path=None, measurements_subdir=None):
    """`cfg` is the pipeline config dict (only aggregation keys are used)."""
    paths = output_paths(cfg)
    input_dir = Path(input_dir or cfg["aggregation_input_dir"] or cfg["output_root"])
    output_path = Path(output_path or paths["aggregated"]
                       or input_dir / f"aggregated_cell_measurements.{cfg['aggregation_format']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plate_map = None
    if cfg["plate_map_path"]:
        plate_map = load_plate_map(
            cfg["plate_map_path"], cfg["plate_map_sheet"], cfg["compound_sheet"] or None,
            parse_str_list(cfg["compound_columns"]), cfg["plate_column"], cfg["treatment_column"])

    selected = find_and_resolve_scans(
        input_dir, measurements_subdir or cfg.get("measurements_subdir", "measurements"),
        cfg["dataset_regex"], cfg["keep_latest_rescan"], cfg.get("datasets_subdir", "per_file"))
    if not selected:
        print("No measurement CSVs found to aggregate.")
        return None

    print(f"\nAggregating {len(selected)} dataset(s)...")
    dfs, meta_cols = [], set()
    for e in selected:
        print(f"  + {e['dataset']}")
        df, mc = process_one_scan(e, plate_map, cfg)
        dfs.append(df)
        meta_cols.update(mc)

    combined = pd.concat(dfs, ignore_index=True, sort=False)
    add_zscores(combined, cfg)
    print(f"Combined: {len(combined):,} cells, {combined.shape[1]} columns, "
          f"~{combined.memory_usage(deep=True).sum() / 1e9:.2f} GB in memory")

    if cfg["aggregation_format"] == "parquet" or output_path.suffix == ".parquet":
        for c in list(meta_cols) + ["well_row", "well_column", "well_id", "scene_confluent"]:
            if c in combined.columns:
                combined[c] = combined[c].astype(str)
        combined.to_parquet(output_path, index=False)
    else:
        combined.to_csv(output_path, index=False)
    print(f"✓ Saved {output_path}")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m cell_analyzer.stages.aggregation <results_dir> <output_path> [plate_map.xlsx]")
        sys.exit(1)
    from ..config import default_config
    cfg = default_config()
    cfg["aggregation_input_dir"] = sys.argv[1]
    cfg["aggregation_output_path"] = sys.argv[2]
    cfg["aggregation_format"] = "parquet" if sys.argv[2].endswith(".parquet") else "csv"
    if len(sys.argv) > 3:
        cfg["plate_map_path"] = sys.argv[3]
    aggregate_measurements(cfg)
