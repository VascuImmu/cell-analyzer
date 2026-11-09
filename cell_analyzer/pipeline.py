"""
cell_analyzer/pipeline.py

Config-driven orchestrator:
    background / flatfield estimation  ->  segmentation  ->  measurement  ->  aggregation  ->  analysis plots

Input can be
  * a single file (.lif / .czi / .ome.tif / .tif / .nd2),
  * a folder of files of one format (filtered by file name),
  * multi-scene files are filtered by scene name (e.g. "_merged", "stitched").

Usage
-----
  GUI:           python -m cell_analyzer
  config file:   python -m cell_analyzer.pipeline --config my_settings.json
  quick CLI:     python -m cell_analyzer.pipeline --input <file|folder> --output <folder> [--format czi]
                 (asks interactively for format + name filters if a folder is given)

Output layout
-------------
<output_root>/
  logs/
    analysis_log_<timestamp>.json         all parameters, inputs, per-stage results, versions
    analysis_log_<timestamp>.txt          same parameters, human readable
    analysis_log_<timestamp>_console.txt  full console output
  per_file/                               one folder per input file
    <dataset>/
      analysis_parameters.json
      background/  segmentation/  segmentation_qc/  measurements/
  pooled_background/                      (background mode 'pooled')
  results/
    aggregated_cell_measurements.csv      (stage 4)
    plots/                                (stage 5: index.html, plots + statistics per condition)

(The layout is defined in config.output_paths.)

All scenes of all files are processed by ONE process pool per stage, so a
folder of single-scene .czi files is parallelised just as well as a .lif
with hundreds of scenes. Existing outputs are skipped unless the matching
overwrite_* option is set, so an interrupted run can simply be restarted.
"""

import argparse
import datetime as dt
import json
import numpy as np
import multiprocessing
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .config import (output_paths, PIPELINE_VERSION, normalize_config, validate_config, load_config_file,
                             config_as_text, parse_int_list, parse_channel_paths, parse_str_list,
                             SUPPORTED_FORMATS)
from . import io_utils


# -------------------------------------------------
# Logging helpers
# -------------------------------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


def progress(stage, done, total):
    # parsed by the GUI for its progress bar
    print(f"[PROGRESS] {stage} {done}/{total}", flush=True)


def _versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node()}
    for mod in ("numpy", "scipy", "skimage", "pandas", "tifffile", "aicsimageio", "dask", "readlif", "aicspylibczi"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            out[mod] = None
    return out


class AnalysisLog:
    """Writes/updates the JSON + TXT analysis log during the run."""

    def __init__(self, cfg, log_dir):
        self.t0 = time.time()
        self.stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.base = self.log_dir / f"analysis_log_{self.stamp}"
        self.data = dict(
            pipeline_version=PIPELINE_VERSION,
            run_id=self.stamp,
            started=dt.datetime.now().isoformat(timespec="seconds"),
            finished=None, duration_s=None, status="running",
            environment=_versions(),
            config=cfg,
            inputs=[], stages={},
        )
        (self.base.with_suffix(".txt")).write_text(
            f"Analysis log  {self.stamp}\nPipeline version {PIPELINE_VERSION}\n"
            f"Started {self.data['started']} on {self.data['environment']['host']}\n\n"
            + config_as_text(cfg))
        self.save()

    @property
    def console_path(self):
        return Path(str(self.base) + "_console.txt")

    def save(self):
        with open(self.base.with_suffix(".json"), "w") as f:
            json.dump(self.data, f, indent=2, default=str)

    def finish(self, status):
        self.data.update(status=status, finished=dt.datetime.now().isoformat(timespec="seconds"),
                         duration_s=round(time.time() - self.t0, 1))
        self.save()
        with open(self.base.with_suffix(".txt"), "a") as f:
            f.write(f"\nFinished {self.data['finished']}  status={status}  "
                    f"duration={self.data['duration_s']} s\n\nStage results:\n")
            f.write(json.dumps(self.data["stages"], indent=2, default=str))
            f.write("\n")


# -------------------------------------------------
# Stages
# -------------------------------------------------
def _dirs(cfg, stem):
    root = output_paths(cfg)["datasets"] / stem
    return dict(root=root,
                background=root / cfg["background_subdir"],
                seg=root / cfg["segmentation_subdir"],
                qc=root / cfg["qc_subdir"],
                meas=root / cfg["measurements_subdir"],
                adj=root / cfg["adjacency_subdir"],
                overview=root / cfg["overview_subdir"],
                track=root / cfg["tracking_subdir"])


def _n_workers(cfg):
    n = int(cfg["n_workers"] or 0)
    return n if n > 0 else min(4, max(1, multiprocessing.cpu_count() - 1))


class RunStopped(Exception):
    """A check failed in a way the user asked to stop for (message is shown as is)."""


def _background_from_image(cfg, datasets, result, summary):
    from .stages.background import fields_from_reference_image, compare_acquisition, _pick_scenes
    from .config import parse_channel_map
    import pandas as pd
    ref = Path(cfg["background_image_path"])
    out_dir = output_paths(cfg)["measured_background"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Background from reference image: {ref.name} ===", flush=True)

    # ---- does the reference fit the images? ----
    policy = cfg["background_metadata_check"]
    if policy != "off":
        ref_info = io_utils.acquisition_info(ref, _pick_scenes(ref, cfg["background_image_scenes"])[0])
        ob = ref_info["objective"] or {}
        print(f"Reference: objective {ob.get('name') or '?'} ({ob.get('magnification') or '?'}x, NA {ob.get('na') or '?'}), "
              f"pixel size {ref_info['pixel_size_um'] or '?'} µm, {ref_info['shape'][1]} × {ref_info['shape'][0]} px, "
              f"{ref_info['n_channels']} channel(s)", flush=True)
        rows, n_bad = [], 0
        for d in datasets:
            try:
                info = io_utils.acquisition_info(d["file"], d["scenes"][0])
                issues = compare_acquisition(ref_info, info)
            except Exception as e:
                issues = [("warning", f"metadata could not be read: {type(e).__name__}: {e}")]
            bad = [t for l, t in issues if l == "error"]
            warn = [t for l, t in issues if l == "warning"]
            n_bad += bool(bad)
            mark = "✗" if bad else ("⚠" if warn else "✓")
            print(f"  {mark} {d['file'].name}: " + ("; ".join(bad + warn) if (bad or warn) else "matches the reference"),
                  flush=True)
            for l, t in issues:
                rows.append(dict(file=d["file"].name, level=l, message=t))
            if not issues:
                rows.append(dict(file=d["file"].name, level="ok", message="matches the reference"))
        pd.DataFrame(rows).to_csv(out_dir / "reference_check.csv", index=False)
        summary["metadata_check"] = dict(files_not_matching=n_bad, table=str(out_dir / "reference_check.csv"))
        if n_bad and policy == "stop":
            raise RunStopped(f"The reference image does not fit {n_bad} input file(s) (see above and "
                             f"{out_dir / 'reference_check.csv'}). Use a matching reference, or set 'Check the "
                             f"reference against the images' to 'warn'.")
        if n_bad:
            summary["errors"].append(f"reference image does not fit {n_bad} file(s) -- corrected anyway (check = warn)")

    # ---- the corrections ----
    try:
        got = fields_from_reference_image(
            ref, parse_int_list(cfg["background_channels"]), out_dir, kind=cfg["background_image_type"],
            channel_map=parse_channel_map(cfg["background_image_channel_map"]),
            scene_include=cfg["background_image_scenes"], dark_path=cfg["background_dark_path"] or None,
            offset=cfg["background_offset"], downsample=cfg["background_downsample"],
            z_projection=cfg["background_z_projection"], smooth_sigma=cfg["background_smooth_sigma"])
    except Exception as e:
        msg = f"Background from {ref.name} failed: {type(e).__name__}: {e}"
        print("✗ " + msg + " -- continuing WITHOUT correction", flush=True)
        summary["errors"].append(msg)
        return result, summary
    for d in datasets:
        result[d["stem"]] = dict(got)
    summary["fields"] = {str(k): v for k, v in got.items()}
    return result, summary


def stage_background(cfg, datasets):
    """Returns {dataset_stem: {channel: npy_path}} and a stage summary."""
    from .stages.background import estimate_background_field

    mode = cfg["background_mode"]
    result = {d["stem"]: {} for d in datasets}
    summary = dict(mode=mode, fields={}, errors=[])
    if mode == "none":
        return result, summary

    if mode == "image":
        return _background_from_image(cfg, datasets, result, summary)

    if mode == "existing":
        paths = parse_channel_paths(cfg["existing_background_paths"])
        for d in datasets:
            result[d["stem"]] = dict(paths)
        summary["fields"] = {str(k): v for k, v in paths.items()}
        return result, summary

    channels = parse_int_list(cfg["background_channels"])
    kw = dict(downsample=cfg["background_downsample"], z_projection=cfg["background_z_projection"],
              smooth_sigma=cfg["background_smooth_sigma"], n_workers=_n_workers(cfg),
              max_scenes=cfg["background_max_scenes"] or None, timepoint=cfg["timepoint"])

    def run_one(sources, out_dir, prefix):
        got = {}
        for ch in channels:
            npy = Path(out_dir) / f"{prefix}_ch{ch}_background.npy"
            png = Path(out_dir) / f"{prefix}_ch{ch}_background.png"
            if npy.exists() and (not cfg["run_background"] or not cfg["overwrite_background"]):
                print(f"↷ Using existing background ch{ch}: {npy}", flush=True)
                got[ch] = str(npy)
                continue
            if not cfg["run_background"]:
                print(f"⚠ Background ch{ch} not found ({npy}) and stage 1 is off -- no correction for ch{ch}",
                      flush=True)
                continue
            try:
                estimate_background_field(sources, channel=ch, save_path=npy, plot_path=png, **kw)
                got[ch] = str(npy)
            except Exception as e:
                msg = f"Background ch{ch} for {prefix} failed: {e}"
                print("✗ " + msg, flush=True)
                summary["errors"].append(msg)
        return got

    if mode == "pooled":
        sources = [(d["file"], s) for d in datasets for s in d["scenes"]]
        out_dir = output_paths(cfg)["pooled_background"]
        print(f"\n--- Pooled background from {len(sources)} scene(s) of {len(datasets)} file(s) ---", flush=True)
        got = run_one(sources, out_dir, "pooled")
        for d in datasets:
            result[d["stem"]] = dict(got)
        summary["fields"] = {str(k): v for k, v in got.items()}
    else:  # per_file
        for i, d in enumerate(datasets, 1):
            print(f"\n--- Background [{i}/{len(datasets)}] {d['file'].name} ---", flush=True)
            if len(d["scenes"]) < 5:
                print(f"⚠ only {len(d['scenes'])} scene(s) in this file -- a per-file background estimate "
                      f"will contain real signal. Consider background mode 'pooled'.", flush=True)
            got = run_one([(d["file"], s) for s in d["scenes"]], _dirs(cfg, d["stem"])["background"], d["stem"])
            result[d["stem"]] = got
            summary["fields"][d["stem"]] = {str(k): v for k, v in got.items()}
            progress("background", i, len(datasets))
    return result, summary


def stage_overview(cfg, datasets):
    from .stages.overview import overview_scene

    jobs = []
    limit = int(cfg["overview_max_scenes"] or 0)
    for d in datasets:
        scenes = list(d["scenes"])
        if limit and len(scenes) > limit:
            scenes = [scenes[i] for i in sorted(set(np.linspace(0, len(scenes) - 1, limit).round().astype(int)))]
        out = _dirs(cfg, d["stem"])["overview"]
        out.mkdir(parents=True, exist_ok=True)
        jobs += [(d, s, out) for s in scenes]
    total = len(jobs)
    print(f"\n=== Overview pictures: {total} scene(s), {_n_workers(cfg)} workers ===", flush=True)
    counts, errors, done = dict(ok=0, skipped_exists=0, error=0), [], 0
    with ProcessPoolExecutor(max_workers=_n_workers(cfg)) as ex:
        futs = {ex.submit(overview_scene, str(d["file"]), s, d["stem"], str(out),
                          {**cfg, "timepoint": d["timepoints"][0]}): (d["stem"], s) for d, s, out in jobs}
        for f in as_completed(futs):
            stem, scene = futs[f]
            done += 1
            try:
                r = f.result()
                counts[r["status"]] += 1
                print(f"[{stem}] " + ("✓ overview: " if r["status"] == "ok" else "↷ exists: ") + Path(r["file"]).name, flush=True)
            except Exception as e:
                counts["error"] += 1
                errors.append(dict(dataset=stem, scene=scene, error=f"{type(e).__name__}: {e}"))
                print(f"[{stem}] ✗ overview failed for {scene}: {type(e).__name__}: {e}", flush=True)
            progress("overview", done, total)
    if jobs:
        print(f"✓ Overview pictures in {output_paths(cfg)['datasets']}/<file>/{cfg['overview_subdir']}/", flush=True)
    return dict(n_scenes=total, **counts, errors=errors)


def stage_segmentation(cfg, datasets, backgrounds):
    from .stages.segmentation import (segment_scene, seg_params_from_config, write_segmentation_summary,
                                      write_object_statistics, write_adjacency)

    params = seg_params_from_config(cfg)
    jobs = []
    for d in datasets:
        dirs = _dirs(cfg, d["stem"])
        dirs["seg"].mkdir(parents=True, exist_ok=True)
        bg = backgrounds.get(d["stem"], {}) if cfg["correct_in_segmentation"] else {}
        for s in d["scenes"]:
            for t in (d["timepoints"] if d["series"] else [None]):
                jobs.append((d, s, t, dirs, bg))

    total = len(jobs)
    use_nuclei = cfg["seed_source"] != "junctions"
    print(f"\n=== Segmentation: {total} {'image(s) (scenes × timepoints)' if any(d['series'] for d in datasets) else 'scene(s)'} "
          f"in {len(datasets)} dataset(s), "
          f"{_n_workers(cfg)} workers ===", flush=True)
    what = ("Only nuclei are segmented (no cell borders)" if cfg["segment_objects"] == "nuclei_only" else
            "Cells are found from " + ("the nuclei channel" if use_nuclei else
                                       "the junction channel alone (no nuclei channel)"))
    print(what + f"; size settings are in "
          f"{'µm' if cfg['size_units'] == 'um' else 'pixels'}, empty ones are derived from a "
          f"{cfg['typical_nucleus_diameter_um']:g} µm nucleus and a {cfg['typical_cell_area_um2']:g} µm² cell.", flush=True)
    sizes_shown, sizes_used = set(), {}
    per_dataset = {d["stem"]: [] for d in datasets}
    counts = dict(ok=0, skipped_exists=0, skipped_sparse=0, error=0)
    errors = []
    done = 0
    with ProcessPoolExecutor(max_workers=_n_workers(cfg)) as ex:
        futs = {ex.submit(segment_scene, str(d["file"]), s, d["stem"], str(dirs["seg"]),
                          str(dirs["qc"]) if cfg["save_qc_plot"] else None, params,
                          bg.get(cfg["nuclei_channel"]) if use_nuclei else None, bg.get(cfg["edge_channel"]),
                          str(dirs["adj"]) if cfg["save_adjacency"] else None, t): d["stem"]
                for d, s, t, dirs, bg in jobs}
        for f in as_completed(futs):
            stem = futs[f]
            try:
                r = f.result()
            except Exception as e:  # worker crashed (e.g. out of memory)
                r = dict(scene="?", status="error", message=f"✗ worker crashed: {e}")
            done += 1
            if r.get("sizes_px") and stem not in sizes_used:
                sizes_used[stem] = r["sizes_px"]
                key = json.dumps(r["sizes_px"], sort_keys=True, default=str)
                if key not in sizes_shown:        # the full table only once per distinct pixel size / image size
                    sizes_shown.add(key)
                    print(f"[{stem}] sizes used (settings converted to pixels for this image):\n{r['sizes_text']}", flush=True)
            print(f"[{stem}] {r['message']}", flush=True)
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            if r["status"] == "error":
                errors.append(f"{stem}: {r['message']}")
            else:
                per_dataset[stem].append(r)
            progress("segmentation", done, total)

    n_nonconf = n_multi = 0
    for d in datasets:
        if per_dataset[d["stem"]]:
            write_segmentation_summary(per_dataset[d["stem"]], _dirs(cfg, d["stem"])["seg"], d["stem"])
            write_object_statistics(per_dataset[d["stem"]], _dirs(cfg, d["stem"])["seg"], d["stem"])
            if cfg["save_adjacency"]:
                write_adjacency(per_dataset[d["stem"]], _dirs(cfg, d["stem"])["adj"], d["stem"])
        n_nonconf += sum(1 for r in per_dataset[d["stem"]] if r.get("confluent") is False)
        n_multi += sum(int(r["n_multinucleated"]) for r in per_dataset[d["stem"]]
                       if r.get("n_multinucleated") == r.get("n_multinucleated") and r.get("n_multinucleated"))
    result = dict(n_scenes=total, **counts, n_nonconfluent_scenes=n_nonconf, n_multinucleated_cells=n_multi,
                  seed_source=cfg["seed_source"], sizes_used_px=sizes_used, errors=errors)

    if cfg["save_size_statistics"]:
        from .stages.diagnostics import run_size_statistics
        print("\n--- Size statistics (for choosing the segmentation settings) ---", flush=True)
        try:
            result["size_statistics"] = run_size_statistics(cfg, [d["stem"] for d in datasets])
        except Exception as e:
            print(f"✗ size statistics failed: {type(e).__name__}: {e}", flush=True)
    return result


def add_timepoints(cfg, datasets):
    """Adds to every dataset: 'series' (bool), 'timepoints' (list) and 'frame_interval_min' (or None)."""
    from .config import parse_timepoints
    for d in datasets:
        if cfg["time_mode"] != "series":
            d.update(series=False, timepoints=[int(cfg["timepoint"])], frame_interval_min=None)
            continue
        n_t = io_utils.n_timepoints(d["file"], d["scenes"][0])
        tps = parse_timepoints(cfg["timepoints"], n_t)
        interval = cfg["frame_interval_min"] or io_utils.frame_interval_min(d["file"], d["scenes"][0])
        d.update(series=True, timepoints=tps, frame_interval_min=interval)
        if not tps:
            print(f"⚠ {d['file'].name}: none of the timepoints '{cfg['timepoints']}' exists "
                  f"(the file has {n_t}, numbered 0-{n_t - 1})", flush=True)
        else:
            iv = f", {interval:g} min apart" if interval else ", time between frames unknown"
            rng = f"{tps[0]}-{tps[-1]}" if tps == list(range(tps[0], tps[-1] + 1)) else ", ".join(map(str, tps[:8])) + ("…" if len(tps) > 8 else "")
            print(f"  • {d['file'].name}: {len(tps)} of {n_t} timepoint(s) ({rng}){iv}", flush=True)
    return datasets


def stage_tracking(cfg, datasets):
    from .stages.tracking import track_dataset
    todo = [d for d in datasets if d["series"] and d["timepoints"]]
    print(f"\n=== Tracking: {sum(len(d['scenes']) for d in todo)} scene(s) over time, "
          f"min. overlap {cfg['track_min_overlap']:g}, bridging up to {cfg['track_max_gap']} missing frame(s) ===",
          flush=True)
    out, errors = {}, []
    for i, d in enumerate(todo, 1):
        dirs = _dirs(cfg, d["stem"])
        try:
            tracks, summ = track_dataset(d["stem"], d["scenes"], d["timepoints"], dirs["seg"], dirs["track"],
                                         cfg["track_min_overlap"], cfg["track_max_gap"], cfg["track_divisions"],
                                         cfg["track_split_fraction"], cfg["save_tracked_labels"],
                                         log=lambda m: print(m, flush=True),
                                         label_suffix="nuclei_labels" if cfg["segment_objects"] == "nuclei_only"
                                         else "cell_labels")
            out[d["stem"]] = dict(n_tracks=int(len(summ)),
                                  n_whole_series=int(summ["whole_series"].sum()) if len(summ) else 0)
        except Exception as e:
            msg = f"tracking {d['stem']} failed: {type(e).__name__}: {e}"
            print("✗ " + msg, flush=True)
            errors.append(msg)
        progress("tracking", i, len(todo))
    return dict(datasets=out, errors=errors)


def stage_measurement(cfg, datasets, backgrounds):
    from .stages.measurement import (measure_scene, meas_params_from_config, save_measurements,
                                         load_scene_info)

    params = meas_params_from_config(cfg)
    jobs, todo = [], []
    for d in datasets:
        dirs = _dirs(cfg, d["stem"])
        csv = dirs["meas"] / f"{d['stem']}_cell_measurements.csv"
        if csv.exists() and not cfg["overwrite_measurement"]:
            print(f"↷ Skipping measurement (exists): {csv}", flush=True)
            continue
        todo.append(d)
        info = load_scene_info(dirs["seg"], d["stem"], dirs["adj"], series=d["series"])
        bg = backgrounds.get(d["stem"], {}) if cfg["correct_in_measurement"] else {}
        for s in d["scenes"]:
            for t in (d["timepoints"] if d["series"] else [None]):
                jobs.append((d, s, t, dirs, bg, info.get((s, t) if d["series"] else s)))

    total = len(jobs)
    print(f"\n=== Measurement: {total} scene(s) in {len(todo)} dataset(s) ===", flush=True)
    rows = {d["stem"]: [] for d in todo}
    extras = {d["stem"]: {} for d in todo}
    errors, done = [], 0
    with ProcessPoolExecutor(max_workers=_n_workers(cfg)) as ex:
        futs = {ex.submit(measure_scene, str(d["file"]), s, d["stem"], str(dirs["seg"]), params, bg, info,
                          t, d.get("frame_interval_min")):
                (d["stem"], s if t is None else f"{s}|t{t}") for d, s, t, dirs, bg, info in jobs}
        for f in as_completed(futs):
            stem, s = futs[f]
            done += 1
            try:
                r, e, msg = f.result()
                rows[stem].extend(r)
                extras[stem][s] = e
            except Exception as err:
                msg = f"✗ Error measuring {s}: {type(err).__name__}: {err}"
                errors.append(f"{stem}: {msg}")
            print(f"[{stem}] {msg}", flush=True)
            progress("measurement", done, total)

    n_cells = {}
    for d in todo:
        if rows[d["stem"]]:
            track_dir = _dirs(cfg, d["stem"])["track"] if (d["series"] and cfg["track_cells"]) else None
            save_measurements(rows[d["stem"]], extras[d["stem"]], _dirs(cfg, d["stem"])["meas"],
                              d["stem"], cfg["save_pickle"], track_dir)
        else:
            print(f"⚠ No cells measured for {d['stem']}", flush=True)
        n_cells[d["stem"]] = len(rows[d["stem"]])
    return dict(n_scenes=total, cells_per_dataset=n_cells, total_cells=sum(n_cells.values()), errors=errors)


# -------------------------------------------------
# Main entry
# -------------------------------------------------
def run_pipeline(cfg):
    cfg = normalize_config(cfg)
    problems = validate_config(cfg)
    if problems:
        for p in problems:
            print(f"✗ {p}")
        raise SystemExit("Configuration invalid -- nothing was run.")

    out_root = Path(cfg["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    log = AnalysisLog(cfg, output_paths(cfg)["logs"])

    console = open(log.console_path, "w", encoding="utf-8")
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = Tee(old_out, console)
    sys.stderr = Tee(old_err, console)

    status = "failed"
    try:
        print(f"Analysis log: {log.base}.json", flush=True)
        print(f"Input: {cfg['input_path']}  (mode: {io_utils.resolve_input_mode(cfg)})", flush=True)

        needs_images = (cfg["run_background"] or cfg["run_segmentation"] or cfg["run_measurement"]
                        or cfg["run_overview"])
        datasets = io_utils.resolve_jobs(cfg) if needs_images else []
        if datasets and cfg["time_mode"] == "series":
            print("Time-lapse: every selected timepoint is analysed as its own image", flush=True)
        add_timepoints(cfg, datasets)
        log.data["inputs"] = [dict(file=str(d["file"]), dataset=d["stem"], format=d["format"],
                                   n_scenes_total=d["n_scenes_total"], n_scenes_used=len(d["scenes"]),
                                   scenes=d["scenes"], timepoints=d["timepoints"] if d["series"] else None,
                                   frame_interval_min=d["frame_interval_min"]) for d in datasets]
        log.save()
        n_sc = sum(len(d["scenes"]) for d in datasets)
        print(f"Found {len(datasets)} file(s) with {n_sc} scene(s) after filtering.", flush=True)
        for d in datasets:
            print(f"  • {d['file'].name}: {len(d['scenes'])}/{d['n_scenes_total']} scenes", flush=True)

        if cfg["write_dataset_params"]:
            for d in datasets:
                root = _dirs(cfg, d["stem"])["root"]
                root.mkdir(parents=True, exist_ok=True)
                with open(root / "analysis_parameters.json", "w") as f:
                    json.dump(dict(run_id=log.stamp, pipeline_version=PIPELINE_VERSION, source_file=str(d["file"]),
                                   scenes=d["scenes"], config=cfg), f, indent=2, default=str)

        if datasets:
            if cfg["run_overview"]:
                log.data["stages"]["overview"] = stage_overview(cfg, datasets)
                log.save()
            backgrounds, s = stage_background(cfg, datasets)
            log.data["stages"]["background"] = s
            log.save()

            if cfg["run_segmentation"]:
                log.data["stages"]["segmentation"] = stage_segmentation(cfg, datasets, backgrounds)
                log.save()
            if cfg["time_mode"] == "series" and cfg["track_cells"] and (cfg["run_segmentation"] or cfg["run_measurement"]):
                log.data["stages"]["tracking"] = stage_tracking(cfg, datasets)
                log.save()
            if cfg["run_measurement"]:
                log.data["stages"]["measurement"] = stage_measurement(cfg, datasets, backgrounds)
                log.save()
        elif needs_images:
            print("⚠ Nothing to process -- check format and name filters.", flush=True)

        if cfg["run_aggregation"]:
            from .stages.aggregation import aggregate_measurements
            print("\n=== Aggregation ===", flush=True)
            path = aggregate_measurements(cfg)
            log.data["stages"]["aggregation"] = dict(output=str(path) if path else None)
            log.save()

        if cfg["run_analysis"]:
            from .stages.analysis import run_analysis
            print("\n=== Analysis plots ===", flush=True)
            try:
                log.data["stages"]["analysis"] = run_analysis(cfg)
            except Exception as e:
                traceback.print_exc()
                log.data["stages"]["analysis"] = dict(errors=[f"analysis failed: {e}"])
            log.save()

        n_err = sum(len(v.get("errors", [])) for v in log.data["stages"].values() if isinstance(v, dict))
        status = "completed" if n_err == 0 else f"completed_with_{n_err}_errors"
        print(f"\nAll done ({status}). Outputs in {out_root}", flush=True)
        print(f"Analysis log: {log.base}.json", flush=True)
    except RunStopped as e:
        status = "stopped"
        print(f"\n✗ Stopped: {e}", flush=True)
        raise SystemExit(2)
    except KeyboardInterrupt:
        status = "cancelled"
        print("\n✗ Cancelled.", flush=True)
        raise
    except BaseException as e:
        print(f"\n✗ Pipeline failed: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        log.finish(status)
        sys.stdout, sys.stderr = old_out, old_err
        console.close()
    return log.base


def _ask(prompt, default=""):
    try:
        v = input(f"{prompt} [{default}]: ").strip()
    except EOFError:
        return default
    return v or default


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="settings JSON (or a previous analysis_log_*.json)")
    ap.add_argument("--input", help="input file or folder (overrides config)")
    ap.add_argument("--output", help="output folder (overrides config)")
    ap.add_argument("--format", choices=list(SUPPORTED_FORMATS), help="file format when input is a folder")
    ap.add_argument("--workers", type=int, help="number of worker processes")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any parameter")
    ap.add_argument("--no-prompt", action="store_true", help="never ask interactively")
    a = ap.parse_args(argv)

    cfg = load_config_file(a.config) if a.config else normalize_config({})
    if a.input:
        cfg["input_path"] = a.input
    if a.output:
        cfg["output_root"] = a.output
    if a.workers is not None:
        cfg["n_workers"] = a.workers
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = v
    if a.format:
        cfg["file_format"] = a.format

    interactive = sys.stdin.isatty() and not a.no_prompt
    if interactive and cfg["input_path"] and Path(cfg["input_path"]).is_dir() and not (a.config or a.format):
        counts = io_utils.detect_formats_in_folder(cfg["input_path"], cfg["recursive"])
        print("Formats found in folder:", ", ".join(f"{k} ({v})" for k, v in counts.items()) or "none")
        cfg["file_format"] = _ask("Which file format should be analysed?",
                                  max(counts, key=counts.get) if counts else cfg["file_format"])
        cfg["filename_include"] = _ask("File name must contain (comma-separated, blank = all)",
                                       cfg["filename_include"])
        cfg["filename_exclude"] = _ask("File name must NOT contain", cfg["filename_exclude"])
        cfg["scene_include"] = _ask("Scene name must contain (multi-scene files)", cfg["scene_include"])
        cfg["scene_exclude"] = _ask("Scene name must NOT contain", cfg["scene_exclude"])

    run_pipeline(cfg)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
