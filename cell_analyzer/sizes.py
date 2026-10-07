"""
cell_analyzer/sizes.py -- physical (µm) size settings -> pixels.

Every size setting of the segmentation can be

  * left EMPTY  -> "automatic": derived from two physical scales the user knows,
        typical nucleus diameter   (default 10 µm)
        typical cell area          (default 1000 µm², e.g. a 20 x 50 µm HUVEC)
  * given as a number -> in µm / µm² when `size_units` is "um" (default),
                         or in pixels when `size_units` is "px" (old settings files).

Per image the values are converted to pixels with that image's pixel size (metadata, or the
"Pixel size (µm)" override). So the same settings work for a 10x and a 40x objective.

Automatic values (d = nucleus diameter, A = cell area, w = sqrt(A / 2.5) = cell width):

    nuclei_sigma            d / 40        nuclei smoothing
    min_nucleus_size        pi d² / 12    1/3 of a nucleus area
    min_distance            0.35 d        between nucleus seeds (0.7 x radius)
    edge_sigma              d / 20        junction smoothing
    min_label_size          A / 10        smallest cell
    max_cell_area           4 A           largest cell made by merging (junction-only mode)
    seed_sigma              w / 8         smoothing to find cell centres (junction-only mode)
    multinuc_max_distance   2 d           nuclei of one cell
    multinuc_band_px        d / 30        junction search width around a boundary
    clear_border_buffer     d / 6
    nuc_dilation_rad        d / 9         non-confluent foreground mask
    min_hole_size           A / 20        non-confluent foreground mask
    boundary_band_px        d / 60        measurement ring at the cell border (>= 1 px)
    min_nuclei_confluent    0.3 x image area / A      (a count, needs the image size)
"""

import math

# key -> (kind, label)   kind: "length" (µm), "area" (µm²), "sigma" (µm, stays fractional in px)
SIZE_KEYS = {
    "nuclei_sigma": ("sigma", "Nuclei smoothing"),
    "min_nucleus_size": ("area", "Min. nucleus size"),
    "min_distance": ("length", "Min. distance between nucleus seeds"),
    "edge_sigma": ("sigma", "Junction smoothing"),
    "min_label_size": ("area", "Min. cell size"),
    "max_cell_area": ("area", "Max. cell size after merging"),
    "seed_sigma": ("sigma", "Smoothing to find cell centres"),
    "multinuc_max_distance": ("length", "Max. distance between the nuclei of one cell"),
    "multinuc_band_px": ("length", "Junction search width"),
    "clear_border_buffer": ("length", "Border buffer"),
    "nuc_dilation_rad": ("length", "Nuclei dilation radius"),
    "min_hole_size": ("area", "Fill holes smaller than"),
    "boundary_band_px": ("length", "Boundary ring width"),
}
MIN_PX = {"min_distance": 1, "min_nucleus_size": 1, "min_label_size": 1, "nuc_dilation_rad": 1,
          "boundary_band_px": 1, "multinuc_band_px": 0, "clear_border_buffer": 0, "min_hole_size": 0}


def auto_sizes_um(cfg):
    """Automatic values in µm / µm² from the two typical scales."""
    d = float(cfg.get("typical_nucleus_diameter_um") or 10.0)
    A = float(cfg.get("typical_cell_area_um2") or 1000.0)
    w = math.sqrt(A / 2.5)
    return {
        "nuclei_sigma": d / 40,
        "min_nucleus_size": math.pi * d * d / 12,
        "min_distance": 0.35 * d,
        "edge_sigma": d / 20,
        "min_label_size": A / 10,
        "max_cell_area": 4 * A,
        "seed_sigma": w / 8,
        "multinuc_max_distance": 2 * d,
        "multinuc_band_px": d / 30,
        "clear_border_buffer": d / 6,
        "nuc_dilation_rad": d / 9,
        "min_hole_size": A / 20,
        "boundary_band_px": d / 60,
    }


def _is_blank(v):
    return v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, float) and math.isnan(v))


def needs_pixel_size(cfg):
    """True if any size has to be converted from µm (i.e. not everything is given in pixels)."""
    if cfg.get("size_units", "um") == "um":
        return True
    return any(_is_blank(cfg.get(k)) for k in SIZE_KEYS)


def resolve_sizes(cfg, pixel_size_um, image_shape=None):
    """
    Convert all size settings to pixels for one image.

    Returns (px, report):
      px      {key: value in pixels}  -- lengths/areas as int, sigmas as float;
              plus "min_nuclei_confluent" (count) and "pixel_size_um"
      report  list of dicts (key, label, kind, value_um, value_px, source) for logs / the GUI
    """
    units = cfg.get("size_units", "um")
    ps = float(pixel_size_um) if pixel_size_um else None
    if ps is None and needs_pixel_size(cfg):
        raise ValueError("this image has no pixel size in its metadata -- enter 'Pixel size (µm)' "
                         "(Stages & Channels tab), or give every size setting in pixels")
    auto = auto_sizes_um(cfg)
    px, report = {"pixel_size_um": ps}, []
    for key, (kind, label) in SIZE_KEYS.items():
        raw = cfg.get(key)
        if _is_blank(raw) or (key == "multinuc_max_distance" and float(raw) == 0):
            v_um, source = auto[key], "auto"
        elif units == "um":
            v_um, source = float(raw), "set"
        else:
            v_um, source = None, "set"
        if v_um is not None:
            v_px = v_um / (ps * ps) if kind == "area" else v_um / ps
        else:
            v_px = float(raw)
            v_um = (v_px * ps * ps if kind == "area" else v_px * ps) if ps else None
        if kind != "sigma":
            v_px = int(max(MIN_PX.get(key, 0), round(v_px)))
        else:
            v_px = float(max(v_px, 0.0))
        px[key] = v_px
        report.append(dict(key=key, label=label, kind=kind, value_um=v_um, value_px=v_px, source=source))

    # confluence threshold: a number of nuclei/cells per image
    raw = cfg.get("min_nuclei_confluent")
    if _is_blank(raw):
        if image_shape is not None and ps:
            area_um2 = float(image_shape[-1]) * float(image_shape[-2]) * ps * ps
            px["min_nuclei_confluent"] = int(round(0.3 * area_um2 / float(cfg.get("typical_cell_area_um2") or 1000.0)))
            src = "auto"
        else:
            px["min_nuclei_confluent"], src = 0, "auto"
    else:
        px["min_nuclei_confluent"], src = int(float(raw)), "set"
    report.append(dict(key="min_nuclei_confluent", label="Min. nuclei/cells for a confluent image", kind="count",
                       value_um=None, value_px=px["min_nuclei_confluent"], source=src))
    return px, report


def resolve_one(cfg, key, pixel_size_um):
    """Pixel value of a single size setting (used by the measurement stage)."""
    kind = SIZE_KEYS[key][0]
    raw = cfg.get(key)
    ps = float(pixel_size_um) if pixel_size_um else None
    if _is_blank(raw):
        v = auto_sizes_um(cfg)[key]
        v = (v / (ps * ps) if kind == "area" else v / ps) if ps else float(MIN_PX.get(key, 1))
    elif cfg.get("size_units", "um") == "um":
        v = float(raw)
        v = (v / (ps * ps) if kind == "area" else v / ps) if ps else float(MIN_PX.get(key, 1))
    else:
        v = float(raw)
    return float(max(v, 0.0)) if kind == "sigma" else int(max(MIN_PX.get(key, 0), round(v)))


def report_text(report, pixel_size_um, indent="  "):
    """Readable table of the resolved sizes."""
    ps = f"{pixel_size_um:.4g} µm" if pixel_size_um else "unknown"
    lines = [f"{indent}pixel size: {ps}"]
    for r in report:
        if r["kind"] == "count":
            lines.append(f"{indent}{r['label']:<46} {r['value_px']:>9}            ({r['source']})")
            continue
        unit = "µm²" if r["kind"] == "area" else "µm"
        um = f"{r['value_um']:.3g} {unit}" if r["value_um"] is not None else "?"
        pxv = f"{r['value_px']:.2f}" if r["kind"] == "sigma" else f"{r['value_px']}"
        lines.append(f"{indent}{r['label']:<46} {um:>10} = {pxv:>7} px  ({r['source']})")
    return "\n".join(lines)
