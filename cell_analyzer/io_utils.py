"""
cell_analyzer/io_utils.py

Format-independent input handling for the pipeline:
  * finding input files (single file, or a folder filtered by format + name)
  * listing / filtering scenes (multi-scene .lif, .czi, ...)
  * lazily loading single channels / MIP stacks from any file AICSImage reads
  * parsing well IDs (if there are any) from scene or file names

Everything that touches the image reader lives here, so switching e.g. from
aicsimageio to bioio later is a change in `open_image` only.

Reader notes
------------
  .lif  -> needs `readlif`           (pip install readlif)
  .czi  -> needs `aicspylibczi`      (pip install aicspylibczi fsspec)
           mosaic CZIs are stitched automatically (reconstruct_mosaic=True)
  .nd2  -> needs `nd2`               (pip install nd2)
  .tif / .ome.tif work out of the box.
"""

import logging
import re
from pathlib import Path

# aicsimageio (conda build) pulls in bfio, which warns at import time that its optional Java
# (Bio-Formats) backend is missing. The pipeline never uses that backend -- .lif/.czi/.tif are read
# by readlif / aicspylibczi / tifffile -- so the message is only noise. Silence bfio below ERROR.
logging.getLogger("bfio").setLevel(logging.ERROR)

import numpy as np

from .config import SUPPORTED_FORMATS, parse_str_list


# -------------------------------------------------
# Reader
# -------------------------------------------------
def open_image(path):
    from aicsimageio import AICSImage
    return AICSImage(str(path))


def format_of(path):
    name = Path(path).name.lower()
    # check longer (more specific) extensions first, e.g. .ome.tif before .tif
    for fmt, exts in sorted(SUPPORTED_FORMATS.items(), key=lambda kv: -max(len(e) for e in kv[1])):
        if any(name.endswith(e) for e in exts):
            return fmt
    return None


def file_stem(path):
    """Stem without (multi-part) image extension: 'a.ome.tif' -> 'a'."""
    name = Path(path).name
    low = name.lower()
    for exts in SUPPORTED_FORMATS.values():
        for e in sorted(exts, key=len, reverse=True):
            if low.endswith(e):
                return name[: -len(e)]
    return Path(path).stem


def safe_name(text):
    return re.sub(r'[\\/:*?"<>|]', "_", str(text))


def scene_stem(dataset_stem, scene):
    return f"{dataset_stem}_{safe_name(scene)}"


# -------------------------------------------------
# Name filters
# -------------------------------------------------
def name_matches(name, include, exclude, case_sensitive=False, use_regex=False):
    include = parse_str_list(include)
    exclude = parse_str_list(exclude)

    def hit(pattern):
        if use_regex:
            return re.search(pattern, name, 0 if case_sensitive else re.IGNORECASE) is not None
        if case_sensitive:
            return pattern in name
        return pattern.lower() in name.lower()

    if include and not any(hit(p) for p in include):
        return False
    if exclude and any(hit(p) for p in exclude):
        return False
    return True


# -------------------------------------------------
# File discovery
# -------------------------------------------------
def _iter_files(folder, recursive):
    it = Path(folder).rglob("*") if recursive else Path(folder).iterdir()
    for p in it:
        # skip macOS AppleDouble junk and hidden files
        if p.is_file() and not p.name.startswith("."):
            yield p


def detect_formats_in_folder(folder, recursive=False):
    """{format: n_files} for every supported format found in `folder`."""
    counts = {}
    for p in _iter_files(folder, recursive):
        fmt = format_of(p)
        if fmt:
            counts[fmt] = counts.get(fmt, 0) + 1
    return counts


def resolve_input_mode(cfg):
    mode = cfg["input_mode"]
    if mode == "auto":
        mode = "folder" if Path(cfg["input_path"]).is_dir() else "file"
    return mode


def find_input_files(cfg):
    """Return the list of input files according to input mode, format and file-name filter."""
    path = Path(cfg["input_path"])
    mode = resolve_input_mode(cfg)

    if mode == "file":
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        return [path]

    if not path.is_dir():
        raise NotADirectoryError(f"Input folder not found: {path}")

    fmt = cfg["file_format"]
    files = [p for p in _iter_files(path, cfg["recursive"]) if format_of(p) == fmt]
    files = [p for p in files if name_matches(
        p.name, cfg["filename_include"], cfg["filename_exclude"],
        cfg["filter_case_sensitive"], cfg["filter_use_regex"])]
    files = sorted(files)

    skip = int(cfg.get("skip_files") or 0)
    files = files[skip:]
    max_files = int(cfg.get("max_files") or 0)
    if max_files > 0:
        files = files[:max_files]
    return files


def list_scenes(path):
    img = open_image(path)
    scenes = list(img.scenes)
    del img
    return scenes


def filter_scenes(scenes, cfg):
    if len(scenes) <= 1 and not cfg["scene_filter_single_scene_files"]:
        kept = list(scenes)
    else:
        kept = [s for s in scenes if name_matches(
            s, cfg["scene_include"], cfg["scene_exclude"],
            cfg["filter_case_sensitive"], cfg["filter_use_regex"])]
    max_s = int(cfg.get("max_scenes_per_file") or 0)
    if max_s > 0:
        kept = kept[:max_s]
    return kept


def resolve_jobs(cfg, log=print):
    """
    Resolve the configuration into a list of datasets:
        [{"file": Path, "stem": str, "format": str, "scenes": [...], "n_scenes_total": int}, ...]
    One dataset == one input file == one output sub-folder.
    """
    files = find_input_files(cfg)
    datasets = []
    stems_seen = {}
    for f in files:
        try:
            all_scenes = list_scenes(f)
        except Exception as e:
            log(f"✗ Cannot open {f.name}: {e}")
            continue
        scenes = filter_scenes(all_scenes, cfg)
        stem = file_stem(f)
        # disambiguate identical stems (recursive search, same name in different folders)
        if stem in stems_seen:
            stem = f"{safe_name(f.parent.name)}_{stem}"
        stems_seen[stem] = f
        if not scenes:
            log(f"⚠ {f.name}: 0 of {len(all_scenes)} scenes pass the scene filter -- skipped")
            continue
        datasets.append(dict(file=f, stem=stem, format=format_of(f),
                             scenes=scenes, n_scenes_total=len(all_scenes)))
    return datasets


# -------------------------------------------------
# Metadata
# -------------------------------------------------
def inspect_file(path):
    """Quick metadata summary of one file (first scene)."""
    img = open_image(path)
    info = dict(
        file=str(path),
        n_scenes=len(img.scenes),
        first_scenes=list(img.scenes[:10]),
        dims=str(img.dims),
        channel_names=[str(c) for c in (img.channel_names or [])],
        pixel_size_um=_pixel_size(img),
        dtype=str(img.dtype),
    )
    del img
    return info


def _pixel_size(img):
    try:
        ps = img.physical_pixel_sizes
        v = ps.Y if ps.Y else ps.X
        return float(v) if v else None
    except Exception:
        return None


# -------------------------------------------------
# Pixel loading (lazy: only the requested channel ever leaves the disk)
# -------------------------------------------------
def load_channel(path, scene, channel, mode="mip", z_index=None, timepoint=0):
    img = open_image(path)
    img.set_scene(scene)
    n_c = img.dims.C
    if channel >= n_c:
        raise IndexError(f"Channel {channel} requested but '{scene}' only has {n_c} channel(s)")
    ch = img.get_image_dask_data("TCZYX")[timepoint, channel]  # lazy ZYX

    if mode == "mip":
        arr = ch.max(axis=0).compute()
    elif mode == "per_slice":
        arr = ch[0 if z_index is None else z_index].compute()
    elif mode == "3d":
        arr = ch.compute()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return np.asarray(arr, dtype=np.float32)


def n_z(path, scene):
    img = open_image(path)
    img.set_scene(scene)
    return img.dims.Z


def load_mip_stack(path, scene, timepoint=0):
    """All channels, max-projected over Z -> (C, Y, X) float32, pixel size (µm or None), channel names."""
    img = open_image(path)
    img.set_scene(scene)
    stack = img.get_image_dask_data("CZYX", T=timepoint).max(axis=1).compute()
    names = [str(c) for c in (img.channel_names or [])]
    return np.asarray(stack, dtype=np.float32), _pixel_size(img), names


# -------------------------------------------------
# Well parsing
# -------------------------------------------------
_GENERIC_WELL_RE = re.compile(r"(?:^|[^A-Za-z0-9])(?P<row>[A-Pa-p])(?P<col>0?[1-9]|[12][0-9]|3[0-2])(?=$|[^0-9])")


def parse_well(scene, filename, well_regex, fallback=True):
    """
    Return (row, col) as strings, e.g. ('B', '5'), or (None, None).
    Tries the configured regex on the scene name, then on the file name;
    optionally falls back to a generic stand-alone 'B5' / 'B05' token.
    """
    candidates = [str(scene or ""), str(filename or "")]
    if well_regex:
        rx = re.compile(well_regex)
        for text in candidates:
            m = rx.search(text)
            if m and m.groupdict().get("row") and m.groupdict().get("col"):
                return m.group("row").upper(), str(int(m.group("col")))
    if fallback:
        for text in candidates:
            m = _GENERIC_WELL_RE.search(text)
            if m:
                return m.group("row").upper(), str(int(m.group("col")))
    return None, None
