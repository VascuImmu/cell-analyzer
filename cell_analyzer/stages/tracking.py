"""
cell_analyzer/stages/tracking.py -- follow cells over time by mask overlap.

Time-series mode segments every frame on its own. This stage links the cell masks of consecutive
frames of one scene:

  1. For two frames, the overlap of every pair of cells is measured as
         IoU = shared area / (area A + area B - shared area)          (0 = apart, 1 = identical)
  2. Pairs are matched one-to-one so that the summed IoU is as large as possible (Hungarian
     algorithm); pairs below the minimum IoU are never linked.
  3. A cell that finds no partner in the previous frame is compared with the frames before that
     (up to `max_gap` frames back), so a cell that was missed once keeps its track.
  4. A cell that still has no partner starts a new track. If most of its area (>= split_fraction)
     lies inside a cell of the previous frame, that cell's track is recorded as its parent
     (division, or a cell that the segmentation cut in two).

Outputs per dataset (folder per_file/<dataset>/tracking/):
  <dataset>_tracks.csv          one row per cell and frame: scene, timepoint, cell_id, track_id,
                                parent_track_id, iou_previous, frames_bridged
  <dataset>_track_summary.csv   one row per track: first/last timepoint, n_frames, parent, n_daughters
  <scene>_tracked_labels.tif    (optional) T x Y x X label movie, grey value = track_id
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.optimize import linear_sum_assignment

from ..io_utils import scene_stem


def overlap_table(lab_a, lab_b):
    """Shared pixels of every touching pair of labels in two label images of the same shape.
    Returns (a_ids, b_ids, shared, area_a, area_b) as arrays over the overlapping pairs."""
    a = lab_a.ravel().astype(np.int64)
    b = lab_b.ravel().astype(np.int64)
    both = (a > 0) & (b > 0)
    nb = int(b.max()) + 1
    pair = a[both] * nb + b[both]
    uniq, shared = np.unique(pair, return_counts=True)
    ai, bi = uniq // nb, uniq % nb
    area_a = np.bincount(a, minlength=int(a.max()) + 1)
    area_b = np.bincount(b, minlength=nb)
    return ai, bi, shared, area_a[ai], area_b[bi]


def match_frames(lab_prev, lab_cur, min_iou, prev_ids=None, cur_ids=None):
    """One-to-one links prev label -> (cur label, IoU), maximising the summed IoU.
    prev_ids / cur_ids restrict the candidates (None = all)."""
    ai, bi, sh, aa, ab = overlap_table(lab_prev, lab_cur)
    iou = sh / (aa + ab - sh)
    keep = iou >= min_iou
    if prev_ids is not None:
        keep &= np.isin(ai, list(prev_ids))
    if cur_ids is not None:
        keep &= np.isin(bi, list(cur_ids))
    ai, bi, iou = ai[keep], bi[keep], iou[keep]
    if not len(ai):
        return {}
    ua, ia = np.unique(ai, return_inverse=True)
    ub, ib = np.unique(bi, return_inverse=True)
    cost = np.zeros((len(ua), len(ub)), dtype=np.float64)
    cost[ia, ib] = -iou
    r, c = linear_sum_assignment(cost)
    return {int(ua[i]): (int(ub[j]), float(-cost[i, j])) for i, j in zip(r, c) if cost[i, j] < 0}


def track_scene(frames, min_iou=0.25, max_gap=1, divisions=True, split_fraction=0.5):
    """
    frames: list of (timepoint, label_image or None) in time order (None = frame missing / skipped).
    Returns (rows, label_to_track) where rows are dicts (timepoint, cell_id, track_id, parent_track_id,
    iou_previous, frames_bridged) and label_to_track maps timepoint -> {cell_id: track_id}.
    """
    rows, l2t = [], {}
    history = []            # [(frame index, timepoint, labels, {cell_id: track_id})] of recent frames
    parent_of = {}
    next_id = 1
    for fi, (t, lab) in enumerate(frames):
        if lab is None:
            continue
        ids = [int(i) for i in np.unique(lab) if i]
        assign, iou_prev, bridged = {}, {}, {}
        used_tracks = set()
        prev = history[-1] if history else None
        # 1) previous available frame
        if prev is not None and fi - prev[0] - 1 <= max_gap:
            for pa, (cb, iou) in match_frames(prev[2], lab, min_iou).items():
                tr = prev[3].get(pa)
                if tr is not None:
                    assign[cb], iou_prev[cb], bridged[cb] = tr, iou, fi - prev[0] - 1
                    used_tracks.add(tr)
        # 2) older frames for cells still without a partner (bridging missed frames)
        for h in reversed(history[:-1]):
            if fi - h[0] - 1 > max_gap:
                break
            free_cur = [i for i in ids if i not in assign]
            if not free_cur:
                break
            # tracks seen in h that did not continue after it
            later = set().union(*[set(x[3].values()) for x in history if x[0] > h[0]]) if history else set()
            free_prev = [c for c, tr in h[3].items() if tr not in later and tr not in used_tracks]
            if not free_prev:
                continue
            for pa, (cb, iou) in match_frames(h[2], lab, min_iou, free_prev, free_cur).items():
                tr = h[3][pa]
                assign[cb], iou_prev[cb], bridged[cb] = tr, iou, fi - h[0] - 1
                used_tracks.add(tr)
        # 3) new tracks, with parent if the new cell lies mostly inside a cell of the previous frame
        new = [i for i in ids if i not in assign]
        parents = {}
        if divisions and new and prev is not None and fi - prev[0] - 1 <= max_gap:
            ai, bi, sh, aa, ab = overlap_table(prev[2], lab)
            frac = sh / ab
            for cb in new:
                m = (bi == cb) & (frac >= split_fraction)
                if m.any():
                    pa = int(ai[m][np.argmax(frac[m])])
                    if pa in prev[3]:
                        parents[cb] = prev[3][pa]
        for cb in new:
            assign[cb] = next_id
            if cb in parents:
                parent_of[next_id] = parents[cb]
            next_id += 1
        for cb in ids:
            tr = assign[cb]
            rows.append(dict(timepoint=t, cell_id=cb, track_id=tr, parent_track_id=parent_of.get(tr, np.nan)
                             if cb in new else np.nan, iou_previous=iou_prev.get(cb, np.nan),
                             frames_bridged=bridged.get(cb, 0)))
        l2t[t] = dict(assign)
        history.append((fi, t, lab, dict(assign)))
        history = history[-(max_gap + 1):]
    return rows, l2t


def track_summary(tracks):
    """One row per (scene, track): first/last timepoint, number of frames, parent, number of daughters."""
    if not len(tracks):
        return pd.DataFrame()
    g = tracks.groupby(["scene", "track_id"])
    s = g.agg(first_timepoint=("timepoint", "min"), last_timepoint=("timepoint", "max"),
              n_frames=("timepoint", "size"), parent_track_id=("parent_track_id", "max")).reset_index()
    kids = tracks.dropna(subset=["parent_track_id"]).groupby(["scene", "parent_track_id"]).size()
    s["n_daughters"] = [int(kids.get((sc, tr), 0)) for sc, tr in zip(s["scene"], s["track_id"])]
    n_t = tracks.groupby("scene")["timepoint"].nunique()
    s["whole_series"] = s["n_frames"] == s["scene"].map(n_t)
    return s


def track_dataset(dataset_stem, scenes, timepoints, seg_dir, out_dir, min_iou=0.25, max_gap=1,
                  divisions=True, split_fraction=0.5, save_labels=True, log=print):
    """Track every scene of one dataset from the label images on disk. Returns (tracks_df, summary_df)."""
    seg_dir, out_dir = Path(seg_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for scene in scenes:
        frames, shape = [], None
        for t in timepoints:
            f = seg_dir / f"{scene_stem(dataset_stem, scene, t)}_cell_labels.tif"
            lab = tifffile.imread(f) if f.exists() else None
            if lab is not None:
                if lab.ndim != 2:
                    raise ValueError(f"{f.name}: tracking needs 2D labels (Z handling 'mip')")
                if shape is not None and lab.shape != shape:
                    h, w = min(shape[0], lab.shape[0]), min(shape[1], lab.shape[1])
                    lab = lab[:h, :w]
                    frames = [(tt, (x[:h, :w] if x is not None else None)) for tt, x in frames]
                shape = lab.shape
            frames.append((t, lab))
        n_avail = sum(x is not None for _, x in frames)
        if n_avail == 0:
            log(f"[{dataset_stem}] ↷ {scene}: no label images to track")
            continue
        rows, l2t = track_scene(frames, min_iou, max_gap, divisions, split_fraction)
        for r in rows:
            r["scene"] = scene
        all_rows += rows
        n_tr = len({r["track_id"] for r in rows})
        full = sum(1 for tr in {r["track_id"] for r in rows}
                   if sum(1 for r in rows if r["track_id"] == tr) == n_avail)
        n_div = sum(1 for r in rows if r["parent_track_id"] == r["parent_track_id"])
        log(f"[{dataset_stem}] ✓ tracked {scene}: {n_avail} frame(s), {n_tr} tracks, {full} through all frames"
            + (f", {n_div} split(s)/division(s)" if divisions else ""))
        if save_labels:
            movie = np.zeros((len(frames),) + shape, dtype=np.uint32)
            for i, (t, lab) in enumerate(frames):
                if lab is None:
                    continue
                lut = np.zeros(int(lab.max()) + 1, dtype=np.uint32)
                for cid, tr in l2t.get(t, {}).items():
                    lut[cid] = tr
                movie[i] = lut[lab]
            dtype = np.uint16 if movie.max() < 65535 else np.uint32
            tifffile.imwrite(out_dir / f"{scene_stem(dataset_stem, scene)}_tracked_labels.tif",
                             movie.astype(dtype), compression="zlib", imagej=True, metadata={"axes": "TYX"})
    tracks = pd.DataFrame(all_rows, columns=["scene", "timepoint", "cell_id", "track_id", "parent_track_id",
                                             "iou_previous", "frames_bridged"])
    summary = track_summary(tracks)
    tracks.to_csv(out_dir / f"{dataset_stem}_tracks.csv", index=False)
    summary.to_csv(out_dir / f"{dataset_stem}_track_summary.csv", index=False)
    return tracks, summary


def add_tracks_to_measurements(df, tracking_dir, dataset_stem):
    """Merge track_id, parent_track_id and the track length into a measurement table (in place-safe)."""
    p = Path(tracking_dir) / f"{dataset_stem}_tracks.csv"
    if not p.exists() or not len(df):
        return df
    tr = pd.read_csv(p)
    sm = Path(tracking_dir) / f"{dataset_stem}_track_summary.csv"
    if sm.exists():
        s = pd.read_csv(sm)[["scene", "track_id", "n_frames"]].rename(columns={"n_frames": "track_n_frames"})
        tr = tr.merge(s, on=["scene", "track_id"], how="left")
    tr = tr[["scene", "timepoint", "cell_id", "track_id", "parent_track_id", "track_n_frames"]]
    df = df.drop(columns=[c for c in ("track_id", "parent_track_id", "track_n_frames") if c in df.columns])
    df = df.merge(tr, on=["scene", "timepoint", "cell_id"], how="left")
    cols = list(df.columns)
    for c in ("track_n_frames", "parent_track_id", "track_id"):     # right after cell_id
        cols.remove(c)
        cols.insert(cols.index("cell_id") + 1, c)
    return df[cols]
