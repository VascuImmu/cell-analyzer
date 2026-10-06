"""
cell_analyzer/stages/analysis.py -- stage 5: automated plots and statistics per condition.

Reads the aggregated cell table (stage 4) and, for every "group by" column
(e.g. Treatment, source_dataset, well_id), writes:

  results/plots/
    index.html                         overview page with every figure
    by_<group>/
      summary_stats_by_<group>.csv     n, mean, median, SD, quartiles, change vs control, p-values
      violin/<column>.png
      box/<column>.png
      replicates/<column>.png          violin + one dot per replicate (scene / well / dataset)
      ecdf/<column>.png
      overview_heatmap.png             every condition x every column, shift vs control
    plate_heatmaps/<column>.png        per-well median on the plate layout (if wells are known)

Statistics note: cells from the same well/scene are not independent, so
cell-level p-values are almost always tiny. The replicate-level test (on
per-replicate medians) is the one to trust; both are reported.

Standalone use:
    python -m cell_analyzer.stages.analysis --config settings.json
    python -m cell_analyzer.stages.analysis aggregated.csv out_dir [--group-by Treatment] [--columns "mean_*"]
"""

import argparse
import fnmatch
import html
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from ..config import parse_str_list, output_paths

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- colours: one hue for conditions, neutral grey for controls ----
COND = "#2a78d6"
CTRL = "#9a9892"
INK = "#1f1f1e"
INK2 = "#52514e"
GRID = "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#eef4fc", "#9cc3ef", "#2a78d6", "#123e73"])
SEQ.set_bad("#efeeea")  # empty wells
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f2f1ee", "#e34948"])

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})

META_COLS = ["plate_num", "sample_num", "rescan_num", "source_dataset", "dataset", "file", "scene",
             "well_row", "well_column", "well_id", "scene_confluent", "is_control", "cell_id"]


# -------------------------------------------------
# helpers
# -------------------------------------------------
def _natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def _safe(s):
    return re.sub(r"[^\w.\-]+", "_", str(s)).strip("_")[:80] or "x"


def _read_table(path, columns_wanted=None):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    head = pd.read_csv(path, nrows=0).columns
    use = [c for c in head if columns_wanted is None or columns_wanted(c)]
    return pd.read_csv(path, usecols=use, low_memory=False)


def resolve_plot_columns(df, patterns):
    cols = []
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in META_COLS]
    for pat in patterns:
        hits = [c for c in numeric if fnmatch.fnmatchcase(c, pat)]
        cols += [c for c in hits if c not in cols]
    return cols


def resolve_group_columns(df, requested):
    out = []
    for g in requested:
        if g.lower() == "auto":
            for cand in ("Treatment", "well_id", "source_dataset", "dataset", "file"):
                if cand in df.columns and df[cand].notna().any() and df[cand].astype(str).ne("nan").any():
                    g = cand
                    break
        if g not in df.columns:
            print(f"  ⚠ group column '{g}' not in the table -- skipped")
            continue
        if g not in out:
            out.append(g)
    return out


def add_replicate_column(df, level):
    if level == "dataset":
        key = df["source_dataset"].astype(str) if "source_dataset" in df else df["dataset"].astype(str)
    elif level == "well" and "well_id" in df and df["well_id"].notna().any():
        key = df.get("source_dataset", df.get("dataset")).astype(str) + "|" + df["well_id"].astype(str)
    else:
        key = df.get("source_dataset", df.get("dataset")).astype(str) + "|" + df["scene"].astype(str)
    df["_replicate"] = key
    return df


def group_labels(df, gcol, label_col):
    """Map group value -> display label (e.g. '240 · DMSO')."""
    labels = {}
    if label_col and label_col in df.columns and label_col != gcol:
        first = df.groupby(gcol, observed=True)[label_col].first()
        for g, lab in first.items():
            lab = str(lab)
            labels[g] = f"{g} · {lab}" if lab and lab.lower() != "nan" and lab != str(g) else str(g)
    else:
        for g in df[gcol].dropna().unique():
            labels[g] = str(g)
    return labels


def order_groups(df, gcol, sort, controls_first, first_col):
    groups = [g for g in df[gcol].dropna().unique() if str(g).lower() != "nan"]
    if sort == "name":
        groups = sorted(groups, key=_natural_key)
    elif sort == "median" and first_col:
        med = df.groupby(gcol, observed=True)[first_col].median()
        groups = sorted(groups, key=lambda g: med.get(g, np.nan))
    ctrl = set()
    if "is_control" in df.columns:
        frac = df.groupby(gcol, observed=True)["is_control"].mean()
        ctrl = {g for g in groups if frac.get(g, 0) > 0.5}
    if controls_first:
        groups = [g for g in groups if g in ctrl] + [g for g in groups if g not in ctrl]
    return groups, ctrl


def _bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    res = np.empty_like(pv)
    res[order] = np.clip(ranked, 0, 1)
    out[ok] = res
    return out


# -------------------------------------------------
# statistics
# -------------------------------------------------
def summary_stats(df, gcol, groups, ctrl_groups, columns):
    from scipy.stats import mannwhitneyu, ttest_ind

    ctrl_mask = df[gcol].isin(ctrl_groups) if ctrl_groups else pd.Series(False, index=df.index)
    rows = []
    for col in columns:
        c_vals = df.loc[ctrl_mask, col].dropna().to_numpy()
        c_rep = df.loc[ctrl_mask].groupby("_replicate")[col].median().dropna().to_numpy()
        c_med = np.median(c_vals) if len(c_vals) else np.nan
        block = []
        for g in groups:
            sub = df.loc[df[gcol] == g]
            v = sub[col].dropna().to_numpy()
            rep = sub.groupby("_replicate")[col].median().dropna().to_numpy()
            r = dict(group_by=gcol, group=g, column=col, is_control=g in ctrl_groups,
                     n_cells=len(v), n_replicates=len(rep),
                     mean=np.mean(v) if len(v) else np.nan, median=np.median(v) if len(v) else np.nan,
                     std=np.std(v, ddof=1) if len(v) > 1 else np.nan,
                     q25=np.percentile(v, 25) if len(v) else np.nan, q75=np.percentile(v, 75) if len(v) else np.nan,
                     replicate_median_mean=np.mean(rep) if len(rep) else np.nan,
                     replicate_median_sd=np.std(rep, ddof=1) if len(rep) > 1 else np.nan)
            r["median_ratio_vs_ctrl"] = r["median"] / c_med if c_med not in (0, np.nan) and not np.isnan(c_med) else np.nan
            r["median_diff_vs_ctrl"] = r["median"] - c_med
            p_cell = p_rep = np.nan
            if ctrl_groups and g not in ctrl_groups:
                if len(v) >= 3 and len(c_vals) >= 3:
                    p_cell = mannwhitneyu(v, c_vals, alternative="two-sided").pvalue
                if len(rep) >= 2 and len(c_rep) >= 2:
                    p_rep = ttest_ind(rep, c_rep, equal_var=False).pvalue
            r["p_cells_mannwhitney"] = p_cell
            r["p_replicates_welch"] = p_rep
            block.append(r)
        bdf = pd.DataFrame(block)
        bdf["q_cells_BH"] = _bh(bdf["p_cells_mannwhitney"])
        bdf["q_replicates_BH"] = _bh(bdf["p_replicates_welch"])
        rows.append(bdf)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# -------------------------------------------------
# plotting primitives
# -------------------------------------------------
def _limits(values, clip):
    v = np.concatenate([x for x in values if len(x)]) if any(len(x) for x in values) else np.array([0, 1])
    if clip and clip > 0:
        lo, hi = np.nanpercentile(v, [clip, 100 - clip])
    else:
        lo, hi = np.nanmin(v), np.nanmax(v)
    pad = (hi - lo) * 0.05 or 1
    return lo - pad, hi + pad


def _fig_for(n_groups, labels):
    longest = max((len(l) for l in labels), default=4)
    w = min(max(4.5, 0.42 * n_groups + 1.8), 28)
    h = 4.2 + min(longest, 40) * 0.06
    fig, ax = plt.subplots(figsize=(w, h))
    return fig, ax


def _xticks(ax, labels, ctrl_flags):
    ax.set_xticks(range(len(labels)))
    rot = 0 if max((len(l) for l in labels), default=0) <= 6 and len(labels) <= 12 else 45
    ax.set_xticklabels(labels, rotation=rot, ha="right" if rot else "center")
    for t, is_c in zip(ax.get_xticklabels(), ctrl_flags):
        if is_c:
            t.set_fontweight("bold")
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.grid(axis="x", visible=False)


def _title(ax, col, gcol, page, n_pages, n_cells, pad=8):
    t = f"{col}  by {gcol}"
    if n_pages > 1:
        t += f"  (page {page}/{n_pages})"
    ax.set_title(f"{t}\n", loc="left", color=INK, pad=pad)
    ax.set_title(f"{n_cells:,} cells · controls in grey with bold labels", loc="right",
                 fontsize=8, color=INK2, pad=pad)


def _sub(v, n, rng):
    return v if len(v) <= n else rng.choice(v, n, replace=False)


def draw_violin(ax, data, colors, log):
    idx = [i for i, d in enumerate(data) if len(d) and np.ptp(d) > 0]
    if idx:
        parts = ax.violinplot([data[i] for i in idx], positions=idx,
                              widths=0.8, showextrema=False, showmedians=False)
        bodies = parts["bodies"]
    else:
        bodies = []
    for body, c in zip(bodies, [colors[i] for i in idx]):
        body.set_facecolor(c)
        body.set_edgecolor(c)
        body.set_alpha(0.55)
        body.set_linewidth(0.8)
    for i, d in enumerate(data):
        if len(d):
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            ax.vlines(i, q1, q3, color=INK, lw=2.2)
            ax.scatter([i], [med], s=18, color="white", edgecolor=INK, zorder=3, linewidth=1)


def draw_box(ax, full, colors):
    stats = []
    for d in full:
        if not len(d):
            stats.append(dict(med=np.nan, q1=np.nan, q3=np.nan, whislo=np.nan, whishi=np.nan, mean=np.nan))
            continue
        q1, med, q3 = np.percentile(d, [25, 50, 75])
        iqr = q3 - q1
        lo = d[d >= q1 - 1.5 * iqr].min()
        hi = d[d <= q3 + 1.5 * iqr].max()
        stats.append(dict(med=med, q1=q1, q3=q3, whislo=lo, whishi=hi, mean=np.mean(d)))
    b = ax.bxp(stats, positions=range(len(stats)), widths=0.6, showfliers=False, showmeans=True,
               patch_artist=True, meanprops=dict(marker="D", markersize=4, markerfacecolor="white",
                                                 markeredgecolor=INK),
               medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch, c in zip(b["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
        patch.set_edgecolor(c)


def draw_replicates(ax, data, reps, colors, rng):
    draw_violin(ax, data, colors, False)
    for i, r in enumerate(reps):
        if len(r):
            x = i + rng.uniform(-0.18, 0.18, len(r))
            ax.scatter(x, r, s=22, color=colors[i], edgecolor="white", linewidth=1, zorder=4)
            ax.hlines(np.mean(r), i - 0.3, i + 0.3, color=INK, lw=1.6, zorder=5)


def draw_ecdf(ax, data, labels, colors, ctrl_flags):
    # many groups -> faint lines, controls emphasised; direct labels for <= 6
    n = len(data)
    for d, lab, c, is_c in zip(data, labels, colors, ctrl_flags):
        if not len(d):
            continue
        x = np.sort(d)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where="post", color=c, lw=2 if is_c or n <= 6 else 1,
                alpha=1 if is_c or n <= 6 else 0.45, label=lab)
    if n <= 12:
        ax.legend(fontsize=7, loc="lower right")
    ax.set_ylabel("fraction of cells ≤ value")
    ax.grid(axis="both")


# -------------------------------------------------
# plot sets
# -------------------------------------------------
def plots_for_grouping(df, gcol, columns, cfg, out_dir, rng, figures):
    groups, ctrl = order_groups(df, gcol, cfg["sort_groups"], cfg["controls_first"], columns[0] if columns else None)
    counts = df[gcol].value_counts()
    small = [g for g in groups if counts.get(g, 0) < cfg["min_cells_per_group"]]
    if small:
        print(f"  ↷ {len(small)} group(s) with < {cfg['min_cells_per_group']} cells left out of the plots")
    groups = [g for g in groups if g not in small]
    if not groups:
        print(f"  ⚠ no groups to plot for '{gcol}'")
        return None
    labels_map = group_labels(df, gcol, cfg["label_column"])
    gdir = out_dir / f"by_{_safe(gcol)}"
    gdir.mkdir(parents=True, exist_ok=True)
    fmt, dpi = cfg["figure_format"], cfg["figure_dpi"]
    per = max(1, int(cfg["max_groups_per_figure"]))
    pages = [groups[i:i + per] for i in range(0, len(groups), per)]
    kinds = [k for k, on in [("violin", cfg["plot_violin"]), ("box", cfg["plot_box"]),
                             ("replicates", cfg["plot_superplot"]), ("ecdf", cfg["plot_ecdf"])] if on]
    grouped = df.groupby(gcol, observed=True)

    for col in columns:
        try:
            _plot_column(df, gcol, col, groups, ctrl, labels_map, grouped, pages, kinds, cfg, gdir, rng, figures)
        except Exception as e:
            print(f"  ✗ plotting {col} by {gcol} failed: {type(e).__name__}: {e}")
            plt.close("all")

    stats = None
    if cfg["compute_stats"]:
        stats = summary_stats(df, gcol, groups, ctrl, columns)
        stats.insert(2, "label", stats["group"].map(labels_map))
        stats.to_csv(gdir / f"summary_stats_by_{_safe(gcol)}.csv", index=False)
        print(f"  ✓ summary statistics: {gdir / f'summary_stats_by_{_safe(gcol)}.csv'}")

    if cfg["plot_overview"] and columns:
        p = overview_heatmap(df, gcol, groups, ctrl, columns, labels_map, gdir, fmt, dpi)
        if p:
            figures.append((f"by {gcol}", "overview", "all columns", p))
    return stats


def _plot_column(df, gcol, col, groups, ctrl, labels_map, grouped, pages, kinds, cfg, gdir, rng, figures):
    fmt, dpi = cfg["figure_format"], cfg["figure_dpi"]
    full = {g: grouped.get_group(g)[col].dropna().to_numpy() for g in groups}
    reps = {g: grouped.get_group(g).groupby("_replicate")[col].median().dropna().to_numpy() for g in groups}
    if all(len(v) == 0 for v in full.values()):
        print(f"  ↷ {col}: no values -- skipped")
        return
    ylim = _limits(list(full.values()), cfg["y_percentile_clip"])
    for kind in kinds:
        kdir = gdir / kind
        kdir.mkdir(exist_ok=True)
        for pi, page in enumerate(pages, 1):
            labels = [labels_map.get(g, str(g)) for g in page]
            flags = [g in ctrl for g in page]
            colors = [CTRL if f else COND for f in flags]
            data_full = [full[g] for g in page]
            data_draw = [_sub(full[g], cfg["max_points_per_group"], rng) for g in page]
            if kind == "ecdf":
                fig, ax = plt.subplots(figsize=(7, 4.8))
                ecdf_colors = colors if len(page) > 6 else \
                    [CTRL if f else c for f, c in zip(flags, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                               "#e87ba4", "#008300"])]
                draw_ecdf(ax, data_draw, labels, ecdf_colors, flags)
                ax.set_xlim(*ylim)
                ax.set_xlabel(col)
                if cfg["log_scale"]:
                    ax.set_xscale("log")
            else:
                fig, ax = _fig_for(len(page), labels)
                if kind == "violin":
                    draw_violin(ax, data_draw, colors, cfg["log_scale"])
                elif kind == "box":
                    draw_box(ax, data_full, colors)
                else:
                    draw_replicates(ax, data_draw, [reps[g] for g in page], colors, rng)
                _xticks(ax, labels, flags)
                ax.set_ylim(*ylim)
                ax.set_ylabel(col)
                ax.set_xlabel(gcol)
                if cfg["log_scale"]:
                    ax.set_yscale("log")
                for i, g in enumerate(page):  # n per group, small, above axis
                    ax.text(i, 1.0, f"n={len(full[g]):,}" + (f"\n{len(reps[g])} rep" if kind == "replicates" else ""),
                            transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                            fontsize=6, color=INK2)
            _title(ax, col, gcol, pi, len(pages), int(sum(len(d) for d in data_full)),
                   pad=8 if kind == "ecdf" else (22 if kind == "replicates" else 14))
            fig.tight_layout()
            name = f"{_safe(col)}" + (f"_p{pi}" if len(pages) > 1 else "") + f".{fmt}"
            fig.savefig(kdir / name, dpi=dpi)
            plt.close(fig)
            figures.append((f"by {gcol}", kind, col, kdir / name))


def overview_heatmap(df, gcol, groups, ctrl, columns, labels_map, out_dir, fmt, dpi):
    """Robust shift of each group's median vs control (or vs all cells if no control)."""
    ref = df[df[gcol].isin(ctrl)] if ctrl else df
    mat = np.full((len(groups), len(columns)), np.nan)
    grouped = df.groupby(gcol, observed=True)
    for j, col in enumerate(columns):
        r = ref[col].dropna()
        if r.empty:
            continue
        med = r.median()
        mad = 1.4826 * np.median(np.abs(r - med)) or r.std() or 1
        gm = grouped[col].median()
        for i, g in enumerate(groups):
            mat[i, j] = (gm.get(g, np.nan) - med) / mad
    lim = np.nanpercentile(np.abs(mat), 98) if np.isfinite(mat).any() else 1
    lim = max(lim, 0.5)
    h = min(max(3, 0.28 * len(groups) + 2), 40)
    w = min(max(5, 0.5 * len(columns) + 3.5), 30)
    fig, ax = plt.subplots(figsize=(w, h))
    im = ax.imshow(mat, cmap=DIV, norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([labels_map.get(g, str(g)) for g in groups], fontsize=7)
    for t, g in zip(ax.get_yticklabels(), groups):
        if g in ctrl:
            t.set_fontweight("bold")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=7)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("median shift vs " + ("control" if ctrl else "all cells") + " (robust SD units)", fontsize=8)
    ax.set_title(f"Overview by {gcol}", loc="left", color=INK)
    fig.tight_layout()
    path = out_dir / f"overview_heatmap.{fmt}"
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def plate_heatmaps(df, columns, out_dir, fmt, dpi, figures):
    if not {"well_row", "well_column"}.issubset(df.columns) or df["well_row"].isna().all():
        print("  ↷ no well information -- plate heatmaps skipped")
        return
    d = df.dropna(subset=["well_row", "well_column"]).copy()
    d["_r"] = d["well_row"].astype(str).str.upper().map(lambda s: sum((ord(ch) - 64) * 26 ** i
                                                                      for i, ch in enumerate(reversed(s))) - 1)
    d["_c"] = pd.to_numeric(d["well_column"], errors="coerce") - 1
    d = d.dropna(subset=["_c"])
    d["_c"] = d["_c"].astype(int)
    plate_key = "plate_num" if "plate_num" in d and d["plate_num"].notna().any() else "source_dataset"
    if plate_key == "plate_num":
        d["_plate"] = "Plate " + d["plate_num"].astype("Int64").astype(str)
        if "sample_num" in d and d["sample_num"].notna().any():
            d["_plate"] += " · Sample " + d["sample_num"].astype("Int64").astype(str)
    else:
        d["_plate"] = d[plate_key].astype(str)
    # a plate layout only makes sense if a 'plate' contains several wells
    plates = sorted(d["_plate"].unique(), key=_natural_key)
    if all(d.loc[d["_plate"] == p, "well_id"].nunique() < 2 for p in plates):
        d["_plate"] = "all data"
        plates = ["all data"]
    n_rows = max(8, int(d["_r"].max()) + 1)
    n_cols = max(12, int(d["_c"].max()) + 1)
    pdir = out_dir / "plate_heatmaps"
    pdir.mkdir(parents=True, exist_ok=True)
    per_page = 12
    for col in columns:
        med = d.groupby(["_plate", "_r", "_c"])[col].median()
        vals = med.dropna().to_numpy()
        if not len(vals):
            continue
        vmin, vmax = np.nanpercentile(vals, [2, 98])
        for pi in range(0, len(plates), per_page):
            page = plates[pi:pi + per_page]
            nc = min(3, len(page))
            nr = int(np.ceil(len(page) / nc))
            fig, axes = plt.subplots(nr, nc, figsize=(nc * 0.36 * n_cols + 1.5, nr * 0.36 * n_rows + 1.2),
                                     squeeze=False)
            for ax in axes.flat:
                ax.set_visible(False)
            for ax, p in zip(axes.flat, page):
                ax.set_visible(True)
                grid = np.full((n_rows, n_cols), np.nan)
                for (r, c), v in med.xs(p, level=0).items():
                    grid[int(r), int(c)] = v
                im = ax.imshow(grid, cmap=SEQ, vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_xticks(range(n_cols))
                ax.set_xticklabels(range(1, n_cols + 1), fontsize=6)
                ax.set_yticks(range(n_rows))
                ax.set_yticklabels([chr(65 + i) if i < 26 else str(i + 1) for i in range(n_rows)], fontsize=6)
                ax.grid(False)
                ax.set_title(p, fontsize=8, loc="left", color=INK)
                for s in ax.spines.values():
                    s.set_visible(False)
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02).set_label(f"median {col}", fontsize=8)
            fig.suptitle(f"Per-well median of {col}", x=0.01, ha="left", color=INK)
            name = f"{_safe(col)}" + (f"_p{pi // per_page + 1}" if len(plates) > per_page else "") + f".{fmt}"
            fig.savefig(pdir / name, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            figures.append(("plate heatmaps", "plate", col, pdir / name))


def write_index(out_dir, figures, info):
    sections = {}
    for sec, kind, col, path in figures:
        sections.setdefault(sec, {}).setdefault(kind, []).append((col, path))
    parts = [f"""<!doctype html><html><head><meta charset="utf-8"><title>Analysis plots</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--card:#ffffff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--line:#34332f;--card:#232322}}}}
body{{margin:0;padding:24px 16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1400px;margin:auto}} h1{{font-size:22px;margin:0 0 4px}} h2{{margin-top:36px;border-bottom:1px solid var(--line);padding-bottom:6px}}
h3{{font-size:14px;color:var(--ink2);text-transform:uppercase;letter-spacing:.04em;margin:22px 0 8px}}
.meta{{color:var(--ink2)}} .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:12px}}
figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}}
figure img{{width:100%;height:auto;background:#fff;border-radius:4px}} figcaption{{font-size:12px;color:var(--ink2);margin-top:4px}}
nav a{{margin-right:14px;color:inherit}}
</style></head><body><main><h1>Analysis plots</h1><p class="meta">{html.escape(info)}</p><nav>"""]
    parts += [f'<a href="#{_safe(s)}">{html.escape(s)}</a>' for s in sections]
    parts.append("</nav>")
    for sec, kinds in sections.items():
        parts.append(f'<h2 id="{_safe(sec)}">{html.escape(sec)}</h2>')
        for kind, items in kinds.items():
            parts.append(f"<h3>{html.escape(kind)}</h3><div class='grid'>")
            for col, path in items:
                rel = Path(path).relative_to(out_dir).as_posix()
                if rel.endswith(".pdf"):
                    parts.append(f"<figure><a href='{rel}'>{html.escape(col)} (PDF)</a></figure>")
                else:
                    parts.append(f"<figure><a href='{rel}'><img loading='lazy' src='{rel}' alt='{html.escape(col)}'>"
                                 f"</a><figcaption>{html.escape(col)}</figcaption></figure>")
            parts.append("</div>")
    parts.append("</main></body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


# -------------------------------------------------
# entry point used by the pipeline
# -------------------------------------------------
def run_analysis(cfg):
    paths = output_paths(cfg)
    out_root = paths["root"]
    src = cfg["analysis_input_path"] or paths["aggregated"]
    if src and not Path(src).exists() and out_root:
        old = out_root / f"aggregated_cell_measurements.{cfg['aggregation_format']}"  # older versions
        if old.exists():
            src = old
    if not src or not Path(src).exists():
        msg = f"Aggregated table not found ({src}) -- run the aggregation stage first or set the input table."
        print("✗ " + msg)
        return dict(errors=[msg])
    out_dir = Path(cfg["analysis_output_dir"] or paths["plots"] or Path(src).parent / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    patterns = parse_str_list(cfg["plot_columns"])
    needed_groups = parse_str_list(cfg["group_by"]) or ["auto"]

    def wanted(c):
        return (c in META_COLS or c in needed_groups or c in ("Treatment", cfg.get("label_column"))
                or any(fnmatch.fnmatchcase(c, p) for p in patterns))

    print(f"Reading {src}")
    df = _read_table(src, wanted)
    if "source_dataset" not in df.columns and "dataset" in df.columns:
        df["source_dataset"] = df["dataset"]
    if cfg["analysis_exclude_nonconfluent"] and "scene_confluent" in df.columns:
        n0 = len(df)
        df = df[df["scene_confluent"].astype(str).str.lower() != "false"]
        print(f"  ↷ excluded {n0 - len(df):,} cells from non-confluent scenes")
    if "is_control" in df.columns:
        df["is_control"] = df["is_control"].astype(str).str.lower().isin(["true", "1"])

    columns = resolve_plot_columns(df, patterns)
    if not columns:
        msg = f"No numeric columns match {patterns}"
        print("✗ " + msg)
        return dict(errors=[msg])
    gcols = resolve_group_columns(df, needed_groups)
    add_replicate_column(df, cfg["replicate_level"])
    print(f"  {len(df):,} cells · {len(columns)} column(s): {', '.join(columns)}")
    print(f"  grouping by: {', '.join(gcols) or '—'}")

    rng = np.random.default_rng(0)
    figures, n_groups = [], {}
    for g in gcols:
        # plotting a column against itself makes no sense; also treat groups as strings
        d = df.copy()
        d[g] = d[g].map(lambda x: x if pd.isna(x) else (str(int(x)) if isinstance(x, float) and x.is_integer() else str(x)))
        d = d[d[g].notna() & (d[g].str.lower() != "nan")]
        print(f"\n  -- by {g}: {d[g].nunique()} group(s)")
        plots_for_grouping(d, g, [c for c in columns if c != g], cfg, out_dir, rng, figures)
        n_groups[g] = int(d[g].nunique())

    if cfg["plot_plate_heatmap"]:
        plate_heatmaps(df, columns, out_dir, cfg["figure_format"], cfg["figure_dpi"], figures)

    if cfg["make_html_index"]:
        write_index(out_dir, figures, f"{len(df):,} cells from {src} · replicate unit: {cfg['replicate_level']}")
        print(f"  ✓ overview page: {out_dir / 'index.html'}")
    print(f"✓ {len(figures)} figure(s) written to {out_dir}")
    return dict(output_dir=str(out_dir), n_figures=len(figures), groupings=n_groups, columns=columns, errors=[])


def main(argv=None):
    from ..config import load_config_file, normalize_config
    ap = argparse.ArgumentParser(description="Analysis plots from an aggregated cell table")
    ap.add_argument("table", nargs="?", help="aggregated .csv / .parquet")
    ap.add_argument("out_dir", nargs="?", help="output folder for plots")
    ap.add_argument("--config", help="settings JSON or analysis log")
    ap.add_argument("--group-by")
    ap.add_argument("--columns")
    a = ap.parse_args(argv)
    cfg = load_config_file(a.config) if a.config else normalize_config({})
    if a.table:
        cfg["analysis_input_path"] = a.table
    if a.out_dir:
        cfg["analysis_output_dir"] = a.out_dir
    if a.group_by:
        cfg["group_by"] = a.group_by
    if a.columns:
        cfg["plot_columns"] = a.columns
    run_analysis(cfg)


if __name__ == "__main__":
    main()
