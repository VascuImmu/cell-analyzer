"""
cell_analyzer/config.py

Single source of truth for EVERY adjustable parameter of the pipeline.

The GUI (gui.py) builds its widgets directly from PARAM_SCHEMA, the
orchestrator (pipeline.py) reads the same keys, and the analysis log
stores the full resolved dict -- so a parameter that exists here is
automatically exposed in the GUI, saved in settings files and recorded in
the log. To add a new parameter: add it here, then read it where it's used.

Field types
-----------
  str, int, float, bool, choice
  float_opt   -- float or blank (None)
  int_list    -- comma-separated ints            ("1, 3")
  str_list    -- comma-separated strings         ("_merged, stitched")
  path_in     -- input file OR folder
  dir         -- folder
  file_open   -- existing file
  file_save   -- file to be written
  channel_paths -- "0=/path/a.npy; 1=/path/b.npy"
"""

import json
from copy import deepcopy
from pathlib import Path

APP_NAME = "Cell Analyzer"
PIPELINE_VERSION = "3.2"

# format key -> file extensions (lower case)
SUPPORTED_FORMATS = {
    "lif": [".lif"],
    "czi": [".czi"],
    "ome.tif": [".ome.tif", ".ome.tiff"],
    "tif": [".tif", ".tiff"],
    "nd2": [".nd2"],
}


def F(key, default, type_, label, help_="", choices=None, enabled_if=None, advanced=False):
    """Field descriptor. enabled_if = (other_key, allowed_values) -> widget greyed out otherwise."""
    return dict(key=key, default=default, type=type_, label=label, help=help_,
                choices=choices, enabled_if=enabled_if, advanced=advanced)


PARAM_SCHEMA = [
    # ------------------------------------------------------------------
    ("Input", [
        F("input_path", "", "path_in", "Input file or folder",
          "A single image file (.lif/.czi/...) or a folder containing several."),
        F("input_mode", "auto", "choice", "Input mode",
          "auto = decide from the path (file vs folder).", ["auto", "file", "folder"]),
        F("file_format", "lif", "choice", "File format (folder mode)",
          "Which files to pick up when a folder is given.", list(SUPPORTED_FORMATS)),
        F("recursive", False, "bool", "Search sub-folders", "Also look for files in sub-folders."),
        F("filename_include", "", "str_list", "File name must contain",
          "Comma-separated; a file is kept if its name contains ANY of these (blank = all). "
          "e.g. _merged, stitched"),
        F("filename_exclude", "", "str_list", "File name must NOT contain",
          "Comma-separated; files containing ANY of these are dropped."),
        F("scene_include", "_merged", "str_list", "Scene name must contain",
          "For multi-scene files (e.g. .lif). Comma-separated, ANY match keeps the scene "
          "(blank = all scenes). e.g. _merged, stitched"),
        F("scene_exclude", "", "str_list", "Scene name must NOT contain",
          "Comma-separated; scenes containing ANY of these are dropped."),
        F("filter_case_sensitive", False, "bool", "Filters are case-sensitive"),
        F("filter_use_regex", False, "bool", "Filters are regular expressions",
          "Treat each filter entry as a Python regex (re.search) instead of plain text."),
        F("scene_filter_single_scene_files", False, "bool", "Apply scene filter to single-scene files",
          "Single-scene files (typical .czi) usually have generic scene names like 'Image:0'; "
          "by default the scene filter is only applied to files with >1 scene."),
        F("skip_files", 0, "int", "Skip first N files", "Useful for resuming a batch.", advanced=True),
        F("max_files", 0, "int", "Max. number of files (0 = all)", advanced=True),
        F("max_scenes_per_file", 0, "int", "Max. scenes per file (0 = all)",
          "Handy for a quick test run.", advanced=True),
    ]),
    # ------------------------------------------------------------------
    ("Output", [
        F("output_root", "", "dir", "Output folder",
          "Everything goes in here: logs/, per_file/, results/ (see the user guide)."),
        F("log_dir", "", "dir", "Log folder",
          "Analysis logs (parameters, inputs, results, console output) are written here automatically. "
          "Blank = <output folder>/logs"),
        F("datasets_subdir", "per_file", "str", "Sub-folder: results per input file",
          "One folder per input file is created inside it."),
        F("results_subdir", "results", "str", "Sub-folder: combined results",
          "Holds the combined table (stage 4) and the plots (stage 5)."),
        F("background_subdir", "background", "str", "Per file: background fields"),
        F("segmentation_subdir", "segmentation", "str", "Per file: label images"),
        F("qc_subdir", "segmentation_qc", "str", "Per file: QC plots"),
        F("measurements_subdir", "measurements", "str", "Per file: measurements"),
        F("adjacency_subdir", "adjacency", "str", "Per file: cell contacts (adjacency)"),
        F("write_dataset_params", True, "bool", "Copy parameters into every dataset folder",
          "Writes <dataset>/analysis_parameters.json so each result folder is self-describing."),
    ]),
    # ------------------------------------------------------------------
    ("General", [
        F("run_background", False, "bool", "Stage 1: estimate background fields",
          "Only needed for background mode 'per_file' or 'pooled'."),
        F("run_segmentation", True, "bool", "Stage 2: segmentation"),
        F("run_measurement", True, "bool", "Stage 3: measurement"),
        F("run_aggregation", True, "bool", "Stage 4: aggregate all measurements"),
        F("run_analysis", True, "bool", "Stage 5: analysis plots per condition",
          "Violin/box/replicate plots, ECDFs, plate heatmaps and summary statistics from the aggregated table."),
        F("n_workers", 0, "int", "Parallel worker processes (0 = auto)",
          "Scenes are processed in parallel. On a 16 GB machine keep this at 2-6."),
        F("timepoint", 0, "int", "Timepoint index", "Which T to analyse for time-lapse data.", advanced=True),
    ]),
    # ------------------------------------------------------------------
    ("Channels", [
        F("seed_source", "nuclei", "choice", "Cells are found from",
          "nuclei = a nuclear stain starts one cell per nucleus (best). "
          "junctions = NO nuclear stain: cells are found from the junction channel alone -- works for "
          "confluent cells with (nearly) complete junctions.", ["nuclei", "junctions"]),
        F("nuclei_channel", 0, "int", "Nuclei channel index", "Channel used to seed the watershed (DNA/DAPI).",
          enabled_if=("seed_source", ["nuclei"])),
        F("edge_channel", 1, "int", "Edge/junction channel index",
          "Channel used as the whole-cell watershed surface (e.g. VE-Cadherin)."),
        F("channel_names", "DNA, VE-Cadherin, ICAM1, PECAM1", "str_list", "Channel names",
          "Comma-separated, in channel order. Used for measurement column names. "
          "Blank = use names from the file metadata."),
        F("edge_channels", "1, 3", "int_list", "Junction channels for boundary intensity",
          "Channels whose intensity is sampled in a thin ring at the cell boundary."),
    ]),
    # ------------------------------------------------------------------
    ("Physical sizes", [
        F("pixel_size_um", None, "float_opt", "Pixel size (µm)",
          "Empty = read from each image's metadata (normal case). Enter it only if the metadata is "
          "missing or wrong."),
        F("typical_nucleus_diameter_um", 10.0, "float", "Typical nucleus diameter (µm)",
          "All size settings left empty are derived from this and the cell area below, converted to "
          "pixels with each image's pixel size."),
        F("typical_cell_area_um2", 1000.0, "float", "Typical cell area (µm²)",
          "e.g. 1000 µm² for HUVEC (about 20 × 50 µm)."),
        F("size_units", "um", "choice", "Units of the size settings",
          "um = sizes you type on the other tabs are in µm / µm² (recommended). "
          "px = they are in pixels (settings files from before version 3.2).", ["um", "px"]),
    ]),
    # ------------------------------------------------------------------
    ("Background", [
        F("background_mode", "none", "choice", "Background / flatfield correction",
          "none = no correction; per_file = estimate one field per input file; "
          "pooled = one field from ALL input files (best for folders of single-scene .czi); "
          "existing = use .npy files you already have.",
          ["none", "per_file", "pooled", "existing"]),
        F("background_channels", "0, 1", "int_list", "Channels to estimate",
          enabled_if=("background_mode", ["per_file", "pooled"])),
        F("background_downsample", 4, "int", "Downsample factor",
          enabled_if=("background_mode", ["per_file", "pooled"])),
        F("background_smooth_sigma", 50.0, "float", "Smoothing sigma (downsampled px)",
          enabled_if=("background_mode", ["per_file", "pooled"])),
        F("background_z_projection", "max", "choice", "Z projection", "", ["max", "mean", "middle"],
          enabled_if=("background_mode", ["per_file", "pooled"])),
        F("background_max_scenes", 0, "int", "Max scenes used (0 = all)",
          enabled_if=("background_mode", ["per_file", "pooled"])),
        F("overwrite_background", False, "bool", "Overwrite existing background fields",
          enabled_if=("background_mode", ["per_file", "pooled"])),
        F("existing_background_paths", "", "channel_paths", "Existing background files",
          "channel=path pairs separated by ';', e.g. 0=/data/ch0_background.npy; 1=/data/ch1_background.npy",
          enabled_if=("background_mode", ["existing"])),
        F("correct_in_segmentation", True, "bool", "Apply correction before segmentation",
          enabled_if=("background_mode", ["per_file", "pooled", "existing"])),
        F("correct_in_measurement", True, "bool", "Apply correction before measurement",
          enabled_if=("background_mode", ["per_file", "pooled", "existing"])),
    ]),
    # ------------------------------------------------------------------
    ("Segmentation", [
        F("seg_mode", "mip", "choice", "Z handling",
          "mip = max projection (required for measurement); per_slice; 3d.", ["mip", "per_slice", "3d"]),
        F("nuclei_sigma", None, "float_opt", "Nuclei smoothing sigma",
          "Empty = automatic: nucleus diameter ÷ 40 (0.25 µm).", enabled_if=("seed_source", ["nuclei"])),
        F("min_nucleus_size", None, "float_opt", "Min. nucleus size (area)",
          "Smaller objects are removed before seeding. Empty = automatic: ⅓ of a nucleus area (26 µm²).",
          enabled_if=("seed_source", ["nuclei"])),
        F("min_distance", None, "float_opt", "Min. distance between nucleus seeds",
          "Lower = more aggressive splitting of touching nuclei. Empty = automatic: 0.35 × nucleus diameter (3.5 µm).",
          enabled_if=("seed_source", ["nuclei"])),
        F("split_touching_nuclei", True, "bool", "Split touching nuclei",
          "On = touching nuclei stay separate after the nuclei watershed. Off = reproduces the original "
          "script, where a re-labelling step merged them back into one object (use only to compare with "
          "old results).", enabled_if=("seed_source", ["nuclei"])),
        F("edge_sigma", None, "float_opt", "Junction smoothing sigma",
          "Empty = automatic: nucleus diameter ÷ 20 (0.5 µm)."),
        F("edge_mode", "intensity", "choice", "Watershed surface",
          "The landscape the cells grow on. intensity (recommended) = the smoothed junction image; cell "
          "borders run exactly along the middle of the junctions. gradient = Sobel gradient of that image "
          "(called 'LoG' in the old scripts); borders are less precise.", ["intensity", "gradient"]),
        F("min_label_size", None, "float_opt", "Min. cell size (area)",
          "Smaller cells are removed after segmentation. Empty = automatic: 1/10 of a cell area (100 µm²)."),
        F("merge_multinucleated", True, "bool", "Keep multi-nucleated cells together",
          "Every nucleus starts its own cell, so a cell with two nuclei is first cut in two. With this on, "
          "two neighbouring cells are merged back into one when their nuclei are close AND there is no "
          "junction signal on the boundary between them. Merged cells are marked in cyan in the QC pictures.",
          enabled_if=("seed_source", ["nuclei"])),
        F("multinuc_max_distance", None, "float_opt", "Max. distance between the nuclei of one cell",
          "Only nuclei whose centres are closer than this can belong to one cell. "
          "Empty = automatic: two nucleus diameters (20 µm).", enabled_if=("seed_source", ["nuclei"])),
        F("multinuc_junction_ratio", 0.5, "float", "Junction strength needed to keep two cells apart",
          "0 = like the signal under the nuclei (no junction), 1 = like a typical cell-cell boundary of the "
          "same image. Boundaries weaker than this value count as 'no junction'. Lower = merges less often.",
          enabled_if=("seed_source", ["nuclei"])),
        F("multinuc_max_nuclei", 2, "int", "Max. nuclei per cell",
          "Stops chains of merges in regions with weak junction staining.",
          enabled_if=("seed_source", ["nuclei"])),
        F("multinuc_band_px", None, "float_opt", "Junction search width",
          "The junction is looked for within this distance of the boundary between two cells. "
          "Empty = automatic: nucleus diameter ÷ 30 (0.33 µm).", advanced=True),
        F("seed_sigma", None, "float_opt", "No nuclei: smoothing to find cell centres",
          "The junction image is blurred this much; every dark basin that remains starts one cell. "
          "Larger = fewer starting points. Empty = automatic: cell width ÷ 8 (2.5 µm).",
          enabled_if=("seed_source", ["junctions"])),
        F("seed_min_depth", 0.02, "float", "No nuclei: min. depth of a cell centre",
          "How much darker than its surroundings a basin must be (fraction of the image contrast). "
          "Lower = more starting points (extra ones are merged again by the next setting).",
          enabled_if=("seed_source", ["junctions"])),
        F("junction_merge_ratio", 0.4, "float", "No nuclei: junction strength needed between two cells",
          "Two neighbouring regions are merged when the boundary between them is weaker than this: "
          "0 = as dark as a cell centre, 1 = as bright as a clear junction of the same image. "
          "Lower (0.3) if real cells get merged where junctions have gaps; higher (0.5) if cells stay cut in pieces.",
          enabled_if=("seed_source", ["junctions"])),
        F("max_cell_area", None, "float_opt", "No nuclei: max. cell size after merging (area)",
          "Regions are never merged into something larger than this. Empty = automatic: 4 × cell area (4000 µm²).",
          enabled_if=("seed_source", ["junctions"])),
        F("clear_border_labels", True, "bool", "Remove cells touching the image border"),
        F("clear_border_buffer", None, "float_opt", "Border buffer",
          "Empty = automatic: nucleus diameter ÷ 6 (1.7 µm).", enabled_if=("clear_border_labels", [True])),
        F("overwrite_segmentation", True, "bool", "Overwrite existing label images"),
        F("save_nuclei_mask", True, "bool", "Save binary nuclei mask", enabled_if=("seed_source", ["nuclei"])),
        F("save_qc_plot", True, "bool", "Save QC overlay plots"),
        F("qc_dpi", 100, "int", "QC plot DPI", enabled_if=("save_qc_plot", [True])),
        F("save_size_statistics", True, "bool", "Save size-statistics plots",
          "After segmentation: histograms of nucleus/cell area, diameter, eccentricity, distance between "
          "nuclei and nuclei per cell (in µm), with your current thresholds drawn in and suggested "
          "values. Saved in each file's segmentation_qc folder and in results/size_statistics."),
        F("save_adjacency", True, "bool", "Save which cells touch each other (adjacency)",
          "For spatial analysis: every pair of cells that share a boundary, with the length of the contact. "
          "Saved per file in the adjacency folder (table + sparse matrix per image); the measurements get a "
          "column n_neighbours."),
    ]),
    # ------------------------------------------------------------------
    ("Non-confluent", [
        F("nonconfluent_enabled", True, "bool", "Enable non-confluent analysis",
          "If a scene has fewer nuclei than the threshold below it is treated as NON-confluent and the "
          "cell watershed is restricted to a foreground mask (instead of flooding the whole field)."),
        F("min_nuclei_confluent", None, "float_opt", "Min. nuclei (cells) for a scene to count as confluent",
          "Scenes with at least this many nuclei are segmented as a confluent monolayer. "
          "Empty = automatic: 30 % of the cells that would fit into the image at the typical cell area. "
          "Set it very high to always use the foreground mask."),
        F("foreground_mask", "union", "choice", "Foreground mask (non-confluent scenes)",
          "union = dilated nuclei OR bright edge signal; nuclei = dilated nuclei only; "
          "edge = thresholded edge channel only. Without a nuclei channel the area enclosed by junctions is used.",
          ["union", "nuclei", "edge"], enabled_if=("nonconfluent_enabled", [True])),
        F("edge_threshold", None, "float_opt", "Edge intensity threshold",
          "Absolute intensity threshold on the smoothed edge channel. Empty = use the percentile below.",
          enabled_if=("nonconfluent_enabled", [True])),
        F("edge_threshold_percentile", 20.0, "float", "Edge threshold percentile",
          "Only used when the absolute threshold is blank.", enabled_if=("nonconfluent_enabled", [True])),
        F("nuc_dilation_rad", None, "float_opt", "Nuclei dilation radius",
          "Empty = automatic: nucleus diameter ÷ 9 (1.1 µm).", enabled_if=("nonconfluent_enabled", [True])),
        F("nuc_dilation_iterations", 3, "int", "Nuclei dilation iterations", enabled_if=("nonconfluent_enabled", [True])),
        F("min_hole_size", None, "float_opt", "Fill holes smaller than (area)",
          "Empty = automatic: 1/20 of a cell area (50 µm²).", enabled_if=("nonconfluent_enabled", [True])),
        F("sparse_policy", "confluent", "choice", "If disabled: scenes below the threshold are...",
          "confluent = segment them like every other scene (whole field flooded); "
          "skip = do not segment/measure them at all.", ["confluent", "skip"],
          enabled_if=("nonconfluent_enabled", [False])),
    ]),
    # ------------------------------------------------------------------
    ("Measurement", [
        F("overwrite_measurement", True, "bool", "Overwrite existing measurement tables"),
        F("measure_nuclei", True, "bool", "Also measure nuclei (nuc_* columns)"),
        F("boundary_band_px", None, "float_opt", "Boundary ring width",
          "Width of the ring along the cell border in which junction intensity is measured. "
          "Empty = automatic: nucleus diameter ÷ 60, at least 1 pixel."),
        F("ruggedness_smooth_frac", 0.02, "float", "Ruggedness smoothing (fraction of contour length)"),
        F("ruggedness_min_sigma", 1.5, "float", "Ruggedness min. smoothing sigma"),
        F("save_pickle", True, "bool", "Also save .pkl with per-cell boundary intensity arrays"),
        F("well_regex", r"(?P<row>[A-Za-z]{1,2})/(?P<col>\d{1,2})(?:_|$)", "str", "Well regex",
          "Regex with named groups 'row' and 'col', searched in the scene name, then the file name. "
          "Default matches LIF scenes like '.../B/5_Merged'."),
        F("well_regex_fallback", True, "bool", "Fallback: generic well pattern (B5 / B05)",
          "If the regex above doesn't match, look for a stand-alone token like 'B5' or 'B05'."),
    ]),
    # ------------------------------------------------------------------
    ("Aggregation", [
        F("aggregation_input_dir", "", "dir", "Folder with dataset results",
          "Blank = the output folder above (finds per_file/ and folders from older versions)."),
        F("aggregation_output_path", "", "file_save", "Aggregated output file",
          "Blank = <output folder>/results/aggregated_cell_measurements.csv"),
        F("aggregation_format", "csv", "choice", "Output format", "", ["csv", "parquet"]),
        F("dataset_regex", r"Plate(?P<plate>\d+)_Sample(?P<sample>\d+)(?:_(?P<rescan>\d+))?$", "str",
          "Dataset-name regex", "Named groups plate / sample / rescan (all optional). Datasets that don't "
          "match are still included, just without plate/sample numbers."),
        F("keep_latest_rescan", True, "bool", "Keep only the latest rescan per plate/sample"),
        F("exclude_nonconfluent", False, "bool", "Exclude non-confluent scenes"),
        F("plate_map_path", "", "file_open", "Plate map (.xlsx, optional)"),
        F("plate_map_sheet", "Treatment_Map", "str", "Plate map sheet"),
        F("compound_sheet", "Compound_IDs", "str", "Compound sheet (blank = none)"),
        F("compound_columns", "Source Name, Class/Target", "str_list", "Compound columns to add"),
        F("plate_column", "Plate", "str", "Plate column in plate map"),
        F("treatment_column", "Treatment", "str", "Treatment column in plate map"),
        F("zscore_columns", "mean_ICAM1", "str_list", "Columns to Z-score",
          "Z-scored relative to the control cells of the reference group below."),
        F("zscore_reference", "dataset", "choice", "Z-score reference group",
          "dataset = each file/scan against its own controls (one plate per .lif); "
          "plate = all datasets with the same plate number (e.g. one .czi per well); "
          "all = one pooled control reference.", ["dataset", "plate", "all"]),
        F("control_treatments", "240", "str_list", "Control treatment(s)", "Values of the treatment column."),
        F("control_wells", "", "str_list", "Control wells (no plate map)",
          "e.g. A1, A2 -- used when no plate map is given (or in addition to it)."),
    ]),
    # ------------------------------------------------------------------
    ("Analysis", [
        F("analysis_input_path", "", "file_open", "Aggregated table to analyse",
          "Blank = the aggregated output of stage 4. Can also point to an older aggregated .csv/.parquet."),
        F("analysis_output_dir", "", "dir", "Plot folder",
          "Blank = <output folder>/results/plots"),
        F("group_by", "auto", "str_list", "Group cells by (conditions)",
          "One or more columns, each gives its own set of plots, e.g. 'Treatment, source_dataset'. "
          "auto = Treatment if a plate map was used, otherwise well_id if wells are known, otherwise "
          "source_dataset (= one group per input file)."),
        F("label_column", "Source Name", "str", "Extra label column",
          "Shown next to the group name on the axis (e.g. compound name for a treatment code). Blank = none."),
        F("plot_columns", "mean_*, edge_mean_*, Zscore_*", "str_list", "Columns to plot",
          "Comma-separated column names; * wildcards allowed, e.g. mean_ICAM1, mean_*, edge_cv_*, area_um2."),
        F("plot_violin", True, "bool", "Violin plots"),
        F("plot_box", True, "bool", "Box plots"),
        F("plot_superplot", True, "bool", "Replicate plots (violin + dots)",
          "Shows the per-replicate medians on top of the cell distribution -- the honest view for "
          "comparing conditions."),
        F("plot_ecdf", True, "bool", "Cumulative distribution (ECDF) plots"),
        F("plot_plate_heatmap", True, "bool", "Plate heatmaps (per-well median)",
          "Only drawn when well information exists."),
        F("plot_overview", True, "bool", "Overview heatmap (all conditions × all columns)",
          "Median shift of every condition vs. control, in units of the control's robust SD."),
        F("replicate_level", "scene", "choice", "Replicate unit",
          "What one dot in the replicate plot / one replicate in the statistics is: scene (field of view), "
          "well, or dataset (input file).", ["scene", "well", "dataset"]),
        F("sort_groups", "name", "choice", "Group order", "", ["name", "median", "none"]),
        F("controls_first", True, "bool", "Put control group(s) first",
          "Controls are the cells marked by the aggregation (control treatments / control wells); "
          "they are drawn in grey."),
        F("min_cells_per_group", 20, "int", "Min. cells per group", "Smaller groups are left out of the plots."),
        F("max_points_per_group", 20000, "int", "Max. cells drawn per group",
          "Random subsample used only for drawing (violins/ECDF); statistics use all cells."),
        F("max_groups_per_figure", 40, "int", "Max. groups per figure", "More groups are split over several pages."),
        F("y_percentile_clip", 0.5, "float", "Hide extreme outliers (percentile)",
          "Axis range is set to [p, 100-p] percentiles of the data. 0 = full range."),
        F("log_scale", False, "bool", "Logarithmic value axis"),
        F("analysis_exclude_nonconfluent", False, "bool", "Exclude non-confluent scenes"),
        F("compute_stats", True, "bool", "Write summary statistics",
          "Per group: n, mean, median, SD, quartiles, change vs. control, Mann-Whitney p (cell level) and "
          "Welch t-test p on replicate medians, Benjamini-Hochberg corrected."),
        F("figure_format", "png", "choice", "Figure format", "", ["png", "pdf", "svg"]),
        F("figure_dpi", 150, "int", "Figure DPI"),
        F("make_html_index", True, "bool", "Write an HTML overview page (index.html) of all plots"),
    ]),
]

ALL_FIELDS = {f["key"]: f for _, fields in PARAM_SCHEMA for f in fields}


# size settings that existed (in pixels) before version 3.2
LEGACY_PIXEL_KEYS = ("nuclei_sigma", "min_nucleus_size", "min_distance", "edge_sigma", "min_label_size",
                     "clear_border_buffer", "nuc_dilation_rad", "min_hole_size", "boundary_band_px",
                     "multinuc_band_px", "min_nuclei_confluent")


def default_config():
    return {k: deepcopy(f["default"]) for k, f in ALL_FIELDS.items()}


# -------------------------------------------------
# Parsing helpers (config values may be strings from the GUI or JSON)
# -------------------------------------------------
def parse_str_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [s.strip() for s in str(value).split(",") if s.strip()]


def parse_int_list(value):
    return [int(v) for v in parse_str_list(value)]


def parse_channel_paths(value):
    """'0=/a.npy; 1=/b.npy' (or dict) -> {0: '/a.npy', 1: '/b.npy'}"""
    if not value:
        return {}
    if isinstance(value, dict):
        return {int(k): str(v) for k, v in value.items() if v}
    out = {}
    for part in str(value).replace("\n", ";").split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Background entry '{part}' must look like 'channel=path'")
        ch, path = part.split("=", 1)
        out[int(ch.strip())] = path.strip()
    return out


def coerce(key, value):
    """Convert a raw value (e.g. text from a GUI entry) into the field's type."""
    f = ALL_FIELDS[key]
    t = f["type"]
    if t == "int":
        return int(float(value)) if str(value).strip() != "" else 0
    if t == "float":
        return float(value)
    if t == "float_opt":
        return None if value is None or str(value).strip() == "" else float(value)
    if t == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if t == "choice":
        if value not in f["choices"]:
            raise ValueError(f"'{f['label']}': '{value}' is not one of {f['choices']}")
        return value
    if t == "int_list":
        parse_int_list(value)  # validate
        return value if isinstance(value, str) else ", ".join(map(str, value))
    if t == "channel_paths":
        parse_channel_paths(value)  # validate
        if isinstance(value, dict):
            return "; ".join(f"{k}={v}" for k, v in value.items())
        return value
    if t == "str_list" and isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return "" if value is None else str(value)


def normalize_config(cfg):
    """Fill in defaults for missing keys, coerce types, keep unknown keys out."""
    out = default_config()
    unknown = []
    cfg = dict(cfg or {})
    # Settings saved before version 3.2 have no 'size_units' and give all sizes in PIXELS.
    if "size_units" not in cfg and any(k in cfg for k in LEGACY_PIXEL_KEYS):
        cfg["size_units"] = "px"
        if cfg.get("multinuc_max_distance") in (0, "0", 0.0):
            cfg["multinuc_max_distance"] = None          # 0 meant "automatic"
    for k, v in cfg.items():
        if k in ALL_FIELDS:
            out[k] = coerce(k, v)
        else:
            unknown.append(k)
    # backwards-compatible alias from the old code
    if out.get("edge_mode") in ("LoG", "log"):
        out["edge_mode"] = "gradient"
    if unknown:
        print(f"⚠ Ignoring unknown config keys: {unknown}")
    return out


def validate_config(cfg):
    """Return a list of human-readable problems (empty list = OK)."""
    problems = []
    needs_images = cfg["run_background"] or cfg["run_segmentation"] or cfg["run_measurement"]
    if not cfg["input_path"]:
        if needs_images:
            problems.append("No input file/folder chosen.")
    elif not Path(cfg["input_path"]).exists():
        problems.append(f"Input path does not exist: {cfg['input_path']}")
    if not cfg["output_root"]:
        problems.append("No output folder chosen.")
    if cfg["seg_mode"] != "mip" and cfg["run_measurement"]:
        problems.append("Measurement needs 2D labels: set segmentation 'Z handling' to 'mip' "
                        "or disable the measurement stage.")
    if cfg["run_background"] and cfg["background_mode"] not in ("per_file", "pooled"):
        problems.append("Stage 1 (background estimation) is on, but background mode is "
                        f"'{cfg['background_mode']}' -- choose 'per_file' or 'pooled', or switch stage 1 off.")
    if cfg["background_mode"] == "existing":
        try:
            paths = parse_channel_paths(cfg["existing_background_paths"])
            if not paths:
                problems.append("Background mode 'existing' but no background files given.")
            for ch, p in paths.items():
                if not Path(p).exists():
                    problems.append(f"Background file for channel {ch} not found: {p}")
        except ValueError as e:
            problems.append(str(e))
    if cfg["seed_source"] == "nuclei" and cfg["nuclei_channel"] == cfg["edge_channel"]:
        problems.append("Nuclei and edge channel are identical. (No nuclear stain? Set 'Cells are found from' "
                        "to 'junctions' on the Stages & Channels tab.)")
    for key in ("typical_nucleus_diameter_um", "typical_cell_area_um2"):
        if not cfg[key] or cfg[key] <= 0:
            problems.append(f"'{ALL_FIELDS[key]['label']}' must be a positive number.")
    for key in LEGACY_PIXEL_KEYS + ("seed_sigma", "max_cell_area", "multinuc_max_distance", "pixel_size_um"):
        if cfg.get(key) is not None and cfg[key] < 0:
            problems.append(f"'{ALL_FIELDS[key]['label']}' cannot be negative.")
    if cfg["run_aggregation"] and cfg["plate_map_path"] and not Path(cfg["plate_map_path"]).exists():
        problems.append(f"Plate map not found: {cfg['plate_map_path']}")
    import re
    for key in ("well_regex", "dataset_regex"):
        try:
            re.compile(cfg[key])
        except re.error as e:
            problems.append(f"Invalid regex in '{ALL_FIELDS[key]['label']}': {e}")
    if cfg["filter_use_regex"]:
        for key in ("filename_include", "filename_exclude", "scene_include", "scene_exclude"):
            for pat in parse_str_list(cfg[key]):
                try:
                    re.compile(pat)
                except re.error as e:
                    problems.append(f"Invalid regex '{pat}' in '{ALL_FIELDS[key]['label']}': {e}")
    return problems


def output_paths(cfg):
    """
    Where everything goes (single source of truth for the output layout):

      <output_root>/
        logs/                     analysis logs
        per_file/<dataset>/       background/ segmentation/ segmentation_qc/ measurements/ analysis_parameters.json
        pooled_background/        background mode 'pooled'
        results/
          aggregated_cell_measurements.csv
          plots/                  index.html, by_<group>/, plate_heatmaps/
    """
    root = Path(cfg["output_root"]) if cfg.get("output_root") else None
    results = root / cfg.get("results_subdir", "results") if root else None
    fmt = cfg.get("aggregation_format", "csv")
    aggregated = (Path(cfg["aggregation_output_path"]) if cfg.get("aggregation_output_path")
                  else results / f"aggregated_cell_measurements.{fmt}" if results else None)
    return dict(
        root=root,
        logs=Path(cfg["log_dir"]) if cfg.get("log_dir") else (root / "logs" if root else None),
        datasets=root / cfg.get("datasets_subdir", "per_file") if root else None,
        pooled_background=root / "pooled_background" if root else None,
        results=results,
        aggregated=aggregated,
        size_statistics=results / "size_statistics" if results else None,
        plots=Path(cfg["analysis_output_dir"]) if cfg.get("analysis_output_dir") else (results / "plots" if results else None),
    )


def load_config_file(path):
    """Load a settings JSON or a previous analysis log (uses its 'config' section)."""
    with open(path) as f:
        data = json.load(f)
    if "config" in data and isinstance(data["config"], dict):
        data = data["config"]
    return normalize_config(data)


def save_config_file(cfg, path):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def config_as_text(cfg):
    """Human-readable, section-grouped parameter listing (for the .txt log)."""
    lines = []
    for section, fields in PARAM_SCHEMA:
        lines.append(f"[{section}]")
        for f in fields:
            v = cfg.get(f["key"])
            lines.append(f"  {f['label']:<55} ({f['key']}) = {v!r}")
        lines.append("")
    return "\n".join(lines)
