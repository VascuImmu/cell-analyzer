"""
cell_analyzer/stages/diagnostics.py -- size statistics of the segmented objects (in pixels).

Purpose: make educated guesses for the segmentation settings instead of trial and error.
After segmentation, the per-object tables written by stages/segmentation.py

    per_file/<dataset>/segmentation/<dataset>_nuclei_statistics.csv
    per_file/<dataset>/segmentation/<dataset>_cell_statistics.csv
    per_file/<dataset>/segmentation/<dataset>_raw_object_areas.csv

are turned into one figure per input file and one for all files together:

    per_file/<dataset>/segmentation_qc/<dataset>_size_statistics.png
    results/size_statistics/all_files_size_statistics.png
    results/size_statistics/size_statistics_summary.csv      (medians, percentiles, suggestions)

Each panel shows the distribution, its median, the CURRENT value of the related setting
(red dashed) and a SUGGESTED starting value (black dotted):

    min. nucleus size      ~ 1/3 of the median nucleus area
    min. distance (seeds)  ~ 0.35 x median nucleus diameter  (= 0.7 x radius)
    min. cell size         ~ 1/3 of the median cell area

Caveat: nuclei and cells are measured AFTER the current filters, so with badly wrong
settings the medians are biased -- adjust, re-run on a few scenes, and look again.
The first panel (all bright objects before the size filter) does not have this bias.

Standalone:  python -m cell_analyzer.stages.diagnostics --config settings.json
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import output_paths

NUC, CELL, RAW = "#2a78d6", "#1baf7a", "#9a9892"
CUR, INK, INK2, GRID = "#e34948", "#1f1f1e", "#52514e", "#e4e3df"
STATS_FILES = {"nuclei": "nuclei_statistics", "cells": "cell_statistics", "raw": "raw_object_areas"}


def load_tables(seg_dir, dataset):
    out = {}
    for key, name in STATS_FILES.items():
        f = Path(seg_dir) / f"{dataset}_{name}.csv"
        out[key] = pd.read_csv(f) if f.exists() else pd.DataFrame()
    return out


def _med(df, col):
    return float(df[col].median()) if len(df) and col in df else np.nan


def suggestions(t):
    """Suggested starting values from the measured medians (NaN when there is no data)."""
    nuc_area, nuc_diam, cell_area = _med(t["nuclei"], "area_px"), _med(t["nuclei"], "equivalent_diameter_px"), \
        _med(t["cells"], "area_px")

    def rnd(v, base):
        return int(max(base, round(v / base) * base)) if np.isfinite(v) else np.nan

    return dict(
        min_nucleus_size=rnd(nuc_area / 3, 10),
        min_distance=rnd(0.35 * nuc_diam, 1),
        min_label_size=rnd(cell_area / 3, 50),
        multinuc_max_distance_auto=rnd(2 * nuc_diam, 1),
    )


def summary_rows(t, dataset, cfg):
    rows = []
    for key, cols in (("nuclei", ["area_px", "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
                                  "eccentricity", "nearest_neighbour_px"]),
                      ("cells", ["area_px", "equivalent_diameter_px", "major_axis_px", "minor_axis_px",
                                 "eccentricity", "n_nuclei"]),
                      ("raw", ["area_px"])):
        df = t[key]
        for c in cols:
            if not len(df) or c not in df:
                continue
            v = df[c].dropna().to_numpy()
            if not len(v):
                continue
            p5, p25, p50, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
            rows.append(dict(dataset=dataset, object={"raw": "raw objects (before size filter)"}.get(key, key),
                             measure=c, n=len(v), median=p50, p5=p5, p25=p25, p75=p75, p95=p95, mean=v.mean()))
    s = suggestions(t)
    for k, label in (("min_nucleus_size", "Min. nucleus size (px)"), ("min_distance", "Min. distance between nucleus seeds (px)"),
                     ("min_label_size", "Min. cell size (px)")):
        rows.append(dict(dataset=dataset, object="SUGGESTION", measure=label, n=np.nan, median=s[k],
                         current_setting=cfg.get(k)))
    if len(t["cells"]) and "n_nuclei" in t["cells"]:
        rows.append(dict(dataset=dataset, object="cells", measure="fraction multi-nucleated",
                         n=len(t["cells"]), median=float((t["cells"]["n_nuclei"] > 1).mean())))
    return rows


def _hist(ax, v, color, title, xlabel, log=False, bins=50, include=()):
    """`include`: setting values the x-range is stretched to, so their lines are visible."""
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)
    ax.tick_params(labelsize=8, colors=INK2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center", color=INK2)
        return None
    inc = [float(x) for x in include if x is not None and np.isfinite(x) and x > 0]
    if log:
        v = v[v > 0]
        lo, hi = max(v.min(), 1), v.max() * 1.05
        for x in inc:                       # stretch (within reason) to show the setting
            if lo / 30 <= x <= hi * 30:
                lo, hi = min(lo, x * 0.8), max(hi, x * 1.25)
        edges = np.logspace(np.log10(lo), np.log10(hi), bins)
        ax.set_xscale("log")
    else:
        lo, hi = np.percentile(v, [0.5, 99.5])
        if hi <= lo:
            lo, hi = v.min() - 0.5, v.max() + 0.5
        span = hi - lo
        for x in inc:
            if lo - 4 * span <= x <= hi + 4 * span:
                lo, hi = min(lo, x - 0.04 * span), max(hi, x + 0.04 * span)
        lo = max(lo, 0)
        edges = np.linspace(lo, hi, bins + 1)
    counts, _, _ = ax.hist(v, bins=edges, color=color, alpha=0.75, edgecolor="white", linewidth=0.4)
    ax.set_ylim(0, max(counts.max(), 1) * 1.5)      # head-room for the legend
    ax.set_ylabel("count", fontsize=8.5, color=INK2)
    return edges[0], edges[-1]


def _vline(ax, x, kind, text, rng=None):
    if x is None or not np.isfinite(x):
        return
    style = dict(median=dict(color=INK, ls="-", lw=1.4), current=dict(color=CUR, ls="--", lw=1.6),
                 suggested=dict(color=INK, ls=":", lw=1.8), info=dict(color=INK2, ls="-.", lw=1.2))[kind]
    outside = rng is not None and not (rng[0] <= x <= rng[1])
    if not outside:
        ax.axvline(x, **style)
    ax.plot([], [], label=text + ("  (outside the plotted range)" if outside else ""), **style)


def make_figure(t, cfg, title, save_path, dpi=120):
    nuc, cells, raw = t["nuclei"], t["cells"], t["raw"]
    s = suggestions(t)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.6))
    ax = axes.ravel()

    def g(df, col):
        return df[col].to_numpy() if len(df) and col in df else np.array([])

    # 1 raw objects
    r = _hist(ax[0], g(raw, "area_px"), RAW, "All bright objects in the nuclei channel\n(before the size filter)",
              "area (px, log scale)\nleft = debris, right = nuclei: the threshold belongs in the gap", log=True,
              include=[cfg["min_nucleus_size"], s["min_nucleus_size"]])
    _vline(ax[0], cfg["min_nucleus_size"], "current", f"current min. nucleus size = {cfg['min_nucleus_size']}", r)
    _vline(ax[0], s["min_nucleus_size"], "suggested", f"suggested ≈ {s['min_nucleus_size']}", r)

    # 2 nucleus area
    r = _hist(ax[1], g(nuc, "area_px"), NUC, "Nucleus area", "area (px)",
              include=[cfg["min_nucleus_size"], s["min_nucleus_size"]])
    _vline(ax[1], _med(nuc, "area_px"), "median", f"median = {_med(nuc, 'area_px'):.0f}", r)
    _vline(ax[1], cfg["min_nucleus_size"], "current", f"current min. nucleus size = {cfg['min_nucleus_size']}", r)
    _vline(ax[1], s["min_nucleus_size"], "suggested", f"suggested ≈ {s['min_nucleus_size']}  (median ÷ 3)", r)

    # 3 nucleus diameter
    r = _hist(ax[2], g(nuc, "equivalent_diameter_px"), NUC, "Nucleus diameter\n(of a circle with the same area)", "diameter (px)")
    _vline(ax[2], _med(nuc, "equivalent_diameter_px"), "median", f"median = {_med(nuc, 'equivalent_diameter_px'):.1f}", r)

    # 4 nearest neighbour
    md = (cfg["multinuc_max_distance"] or s["multinuc_max_distance_auto"]) if cfg.get("merge_multinucleated") else None
    r = _hist(ax[3], g(nuc, "nearest_neighbour_px"), NUC, "Distance to the nearest nucleus\n(centre to centre)", "distance (px)",
              include=[cfg["min_distance"], s["min_distance"], md])
    _vline(ax[3], _med(nuc, "nearest_neighbour_px"), "median", f"median = {_med(nuc, 'nearest_neighbour_px'):.1f}", r)
    _vline(ax[3], cfg["min_distance"], "current", f"current min. seed distance = {cfg['min_distance']}", r)
    _vline(ax[3], s["min_distance"], "suggested", f"suggested ≈ {s['min_distance']}  (0.35 × diameter)", r)
    if md:
        _vline(ax[3], md, "info", f"multi-nucleated: nuclei closer than {md}" + ("" if cfg["multinuc_max_distance"] else " (auto)"), r)

    # 5 nucleus eccentricity
    r = _hist(ax[4], g(nuc, "eccentricity"), NUC, "Nucleus eccentricity\n(0 = round, 1 = elongated)", "eccentricity")
    _vline(ax[4], _med(nuc, "eccentricity"), "median", f"median = {_med(nuc, 'eccentricity'):.2f}", r)

    # 6 cell area
    r = _hist(ax[5], g(cells, "area_px"), CELL, "Cell area", "area (px)",
              include=[cfg["min_label_size"], s["min_label_size"]])
    _vline(ax[5], _med(cells, "area_px"), "median", f"median = {_med(cells, 'area_px'):.0f}", r)
    _vline(ax[5], cfg["min_label_size"], "current", f"current min. cell size = {cfg['min_label_size']}", r)
    _vline(ax[5], s["min_label_size"], "suggested", f"suggested ≈ {s['min_label_size']}  (median ÷ 3)", r)

    # 7 cell eccentricity
    r = _hist(ax[6], g(cells, "eccentricity"), CELL, "Cell eccentricity\n(0 = round, 1 = elongated)", "eccentricity")
    _vline(ax[6], _med(cells, "eccentricity"), "median", f"median = {_med(cells, 'eccentricity'):.2f}", r)

    # 8 nuclei per cell
    a = ax[7]
    a.set_title("Nuclei per cell", loc="left", fontsize=10, color=INK)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.8)
    a.set_axisbelow(True)
    a.tick_params(labelsize=8, colors=INK2)
    nn = g(cells, "n_nuclei")
    if len(nn):
        ks = list(range(1, int(max(3, nn.max())) + 1))
        counts = [int((nn == k).sum()) for k in ks]
        bars = a.bar([str(k) for k in ks], counts, color=CELL, alpha=0.8, width=0.6)
        for b, c in zip(bars, counts):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c:,}\n{100 * c / len(nn):.1f} %",
                   ha="center", va="bottom", fontsize=8, color=INK)
        a.set_ylim(0, max(counts) * 1.25)
        a.set_xlabel("nuclei in the cell", fontsize=8.5, color=INK2)
        a.set_ylabel("cells", fontsize=8.5, color=INK2)
        if not cfg.get("merge_multinucleated"):
            a.text(0.98, 0.97, "'Keep multi-nucleated cells together' is off", transform=a.transAxes, ha="right",
                   va="top", fontsize=7.5, color=INK2)
    else:
        a.text(0.5, 0.5, "no data", transform=a.transAxes, ha="center", va="center", color=INK2)

    for x in ax[:7]:
        if x.get_legend_handles_labels()[0]:
            x.legend(fontsize=7.5, frameon=True, facecolor="white", framealpha=0.85, edgecolor="none",
                     loc="upper left", handlelength=2.6, borderaxespad=0.2)

    if len(nuc) and "scene" in nuc:
        n_scenes = len(nuc[["dataset", "scene"]].drop_duplicates()) if "dataset" in nuc else nuc["scene"].nunique()
    else:
        n_scenes = 0
    fig.suptitle(f"Size statistics · {title}", x=0.01, ha="left", fontsize=13, color=INK)
    fig.text(0.01, 0.945, f"{len(nuc):,} nuclei · {len(cells):,} cells · {n_scenes} scene(s) · all values in pixels   |   "
             f"red dashed = current setting, black dotted = suggested starting value, solid = median",
             fontsize=9, color=INK2)
    fig.text(0.01, 0.008, "Nuclei and cells are measured AFTER the current filters – if a setting is far off, the medians are biased: "
             "adjust, re-run a few scenes, look again.  Blue = nuclei, green = cells, grey = unfiltered objects.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi)
    plt.close(fig)
    return s


def run_size_statistics(cfg, dataset_stems=None):
    """Make the figures for the given datasets (default: every dataset below per_file/) + one pooled figure."""
    paths = output_paths(cfg)
    root = paths["datasets"]
    if root is None or not root.is_dir():
        print("  ↷ no per-file results found -- size statistics skipped")
        return dict(n_figures=0, errors=[])
    stems = dataset_stems or sorted(p.name for p in root.iterdir() if p.is_dir())
    out_dir = paths["size_statistics"]
    pooled = {k: [] for k in STATS_FILES}
    rows, n_fig, sugg = [], 0, {}
    for stem in stems:
        t = load_tables(root / stem / cfg["segmentation_subdir"], stem)
        if not len(t["nuclei"]):
            continue
        for k in pooled:
            if len(t[k]):
                pooled[k].append(t[k].assign(dataset=stem))
        try:
            make_figure(t, cfg, stem, root / stem / cfg["qc_subdir"] / f"{stem}_size_statistics.png", cfg.get("qc_dpi", 100) + 20)
            n_fig += 1
        except Exception as e:
            print(f"  ✗ size statistics for {stem} failed: {type(e).__name__}: {e}")
            plt.close("all")
        rows += summary_rows(t, stem, cfg)
    if not any(pooled.values()) or not pooled["nuclei"]:
        print("  ↷ no object statistics found (run the segmentation stage first)")
        return dict(n_figures=0, errors=[])

    allt = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in pooled.items()}
    out_dir.mkdir(parents=True, exist_ok=True)
    sugg = make_figure(allt, cfg, f"all files ({len(pooled['nuclei'])})", out_dir / "all_files_size_statistics.png", 130)
    n_fig += 1
    rows = summary_rows(allt, "ALL FILES", cfg) + rows
    pd.DataFrame(rows).round(3).to_csv(out_dir / "size_statistics_summary.csv", index=False)

    print(f"  ✓ size statistics: {out_dir / 'all_files_size_statistics.png'}")
    print(f"    median nucleus area {_med(allt['nuclei'], 'area_px'):.0f} px, diameter "
          f"{_med(allt['nuclei'], 'equivalent_diameter_px'):.1f} px; median cell area {_med(allt['cells'], 'area_px'):.0f} px")
    for k, label in (("min_nucleus_size", "Min. nucleus size"), ("min_distance", "Min. distance between nucleus seeds"),
                     ("min_label_size", "Min. cell size")):
        if np.isfinite(sugg[k]):
            print(f"    {label}: current {cfg[k]}, suggested starting value ≈ {sugg[k]}")
    if len(allt["cells"]) and "n_nuclei" in allt["cells"]:
        print(f"    multi-nucleated cells: {100 * (allt['cells']['n_nuclei'] > 1).mean():.1f} %")
    return dict(n_figures=n_fig, output_dir=str(out_dir),
                suggestions={k: (None if not np.isfinite(v) else v) for k, v in sugg.items()}, errors=[])


if __name__ == "__main__":
    from ..config import load_config_file, normalize_config
    ap = argparse.ArgumentParser(description="Size-statistics plots from an existing output folder")
    ap.add_argument("--config", help="settings JSON or analysis log")
    ap.add_argument("--output", help="output folder (overrides config)")
    a = ap.parse_args()
    cfg = load_config_file(a.config) if a.config else normalize_config({})
    if a.output:
        cfg["output_root"] = a.output
    run_size_statistics(cfg)
