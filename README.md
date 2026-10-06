# Cell Analyzer (v3.1.1)

Cell Analyzer segments nuclei and cells in microscopy images (`.lif`, `.czi`, `.ome.tif`/`.tif`, `.nd2`) and measures every cell. It then combines the results of all files and makes plots and statistics per condition. All of this runs from one window.

> **Not a Python user? Open [`USER_GUIDE.html`](https://github.com/VascuImmu/cell-analyzer/blob/main/USER_GUIDE.html) in your web browser.** It covers installation, a first analysis step by step, and what every setting means.

## Quick start

| | macOS | Windows | Linux |
|---|---|---|---|
| 1. Install [Miniforge](https://github.com/conda-forge/miniforge) | once | once | once |
| 2. Install Cell Analyzer | in Terminal: `bash install_mac.command` | double-click `install_windows.bat` | `bash install_linux.sh` |
| 3. Start | double-click **Cell Analyzer.app** (or `start_mac.command`) | double-click `start_windows.bat` | `bash start_linux.sh` |

The installer creates a conda environment called `cell-analyzer` from `environment.yml`. Every package comes from conda-forge and none from pip, because pip once replaced numpy 1.x with 2.x underneath scikit-image. The installer then imports every package (`tools/check_env.py`) and rebuilds the environment by itself if something is broken. Add `--fresh` to force a clean rebuild.

What the installer does on each system:

- **macOS:** it removes the quarantine flag that blocks double-clicking downloaded scripts, restores the files' execute permission, and builds a local `Cell Analyzer.app`.
- **Windows:** the start file puts the environment's DLL folders first on `PATH`, which is the same thing `conda activate` does. Without this, Windows can load another program's DLLs and crash with code 0xC06D007F.

The start files look for the environment in every usual Miniforge, Miniconda or Anaconda location. They also still accept the environment name `cellpipeline` used before the rename.

## Folder structure

```
cell-analyzer/
├── USER_GUIDE.html           step-by-step guide with screenshots, every setting explained
├── README.md                 this file
├── install_mac.command       install / update / repair      (Windows: install_windows.bat, Linux: install_linux.sh)
├── start_mac.command         start the window               (Windows: start_windows.bat,   Linux: start_linux.sh)
├── Cell Analyzer.app         created by install_mac.command
├── environment.yml           the conda environment (conda-forge only)
├── tools/
│   ├── find_conda.sh         used by the macOS/Linux scripts: finds conda and the environment
│   └── check_env.py          used by the installers: checks that every package loads
└── cell_analyzer/            the program (a Python package)
    ├── __main__.py           python -m cell_analyzer  →  the window
    ├── gui.py                the window (built automatically from config.py)
    ├── config.py             EVERY parameter: default, label, help text, validation; output layout
    ├── pipeline.py           runs the stages in order, writes the analysis log; command line
    ├── io_utils.py           finding files, name filters, scenes, image loading, well parsing
    └── stages/
        ├── background.py     1  flatfield / background estimation
        ├── flatfield.py         applies a background field
        ├── segmentation.py   2  nuclei + cell watershed, confluence decision, multi-nucleated cells, QC plots
        ├── diagnostics.py       size statistics of nuclei and cells, suggested settings
        ├── measurement.py    3  per-cell shape, intensity and junction measurements
        ├── aggregation.py    4  one combined table, plate map, Z-scores
        └── analysis.py       5  plots and statistics per condition
```

## Output structure

```
<output folder>/
├── results/                                  start here
│   ├── aggregated_cell_measurements.csv      every cell of every file (stage 4)
│   ├── plots/                                stage 5: index.html, by_<group>/…, plate_heatmaps/
│   └── size_statistics/                      sizes of nuclei and cells (px), suggested settings
├── per_file/<dataset>/                       one folder per input file
│   ├── analysis_parameters.json
│   ├── segmentation_qc/  segmentation/  measurements/  background/
├── pooled_background/                        background mode "pooled"
└── logs/                                     analysis_log_<time>.json / .txt / _console.txt
```

The layout is defined in one place, `config.output_paths()`. The folder names can be changed on the Input / Output tab. Output folders from older versions, which had dataset folders directly in the output folder, can still be combined and plotted.

## Command line

```bash
conda activate cell-analyzer
cd path/to/cell-analyzer
python -m cell_analyzer                                            # the window
python -m cell_analyzer.pipeline --config settings.json            # run without the window
python -m cell_analyzer.pipeline --input /data/plate.lif --output /data/out --set n_workers=4 min_distance=15
python -m cell_analyzer.pipeline --input /data/czi_folder --output /data/out   # asks for format + filters
python -m cell_analyzer.stages.analysis results/aggregated_cell_measurements.csv results/plots --group-by Treatment
```

`--config` also accepts a previous `logs/analysis_log_*.json`, which repeats that run exactly.

## Pipeline stages

1. **Background** (optional). Averages many scenes into an illumination field. It can be made per file, pooled across all files, or loaded from existing `.npy` files.
2. **Segmentation.**
   - The nuclei channel goes through Otsu thresholding, then a distance transform, then a watershed that splits touching nuclei.
   - Cells are then grown from the nuclei over the edge channel, using its gradient or its intensity.
   - Cells with two nuclei are kept together. Two neighbouring cells are merged when their nuclei are close and the boundary between them has no junction signal, compared with a typical boundary in the same image. Merged cells are marked in cyan in the QC pictures.
   - Size statistics are plotted after segmentation: nucleus and cell area, diameter, eccentricity, distance between nuclei and nuclei per cell, all in pixels, with the current thresholds and suggested starting values drawn in.
   - Scenes below the confluence threshold are masked to the tissue area if non-confluent analysis is on. Otherwise they are treated as confluent or skipped. The decision for each scene is saved in `segmentation_summary.csv`.
3. **Measurement.**
   - Shape: area, axes, orientation, solidity, ruggedness and more.
   - Per-channel intensity in the cell and in the nucleus.
   - Intensity in a ring along the cell boundary, for the junction channels.
   - `n_nuclei`, the number of nuclei in the cell.
   - Well ID, read from the scene or file name with a regex you can change.
4. **Aggregation.**
   - Combines the tables of all files and keeps only the latest rescan of each plate.
   - Adds plate-map metadata.
   - Computes Z-scores against the controls of the same dataset, the same plate, or all data.
5. **Analysis.**
   - Plots per condition: violin, box, replicate plots (superplots), ECDFs, plate heatmaps and an overview heatmap.
   - Summary statistics with two tests against the control. The cell-level Mann-Whitney test is only indicative. Use the replicate-level Welch t-test on per-replicate medians. Both are Benjamini-Hochberg corrected.

## Changes in 3.1

- **Size statistics** (`stages/diagnostics.py`). After segmentation the program writes `results/size_statistics/all_files_size_statistics.png`, one figure per file in `segmentation_qc/`, and `size_statistics_summary.csv`. The Run log prints suggested values for min. nucleus size, min. seed distance and min. cell size.
- **Multi-nucleated cells.** New settings on the Segmentation tab: keep multi-nucleated cells together (on by default), max. distance between the nuclei, junction strength, max. nuclei per cell. There is a new measurement column `n_nuclei`, and the QC pictures mark merged cells in cyan. Switching the option off gives the previous behaviour.
- New per-file tables in `segmentation/`: `*_nuclei_statistics.csv`, `*_cell_statistics.csv`, `*_raw_object_areas.csv`.

## Changes in 3.0

- Renamed to **Cell Analyzer**. The code is now a package (`cell_analyzer/` with `stages/`), and the helper scripts are in `tools/`.
- New output layout: `results/`, `per_file/`, `pooled_background/` and `logs/`.
- Conda environment `cell-analyzer`. The old `cellpipeline` environment is still found by the start files and can be removed once the new one is installed.
- Settings saved by the previous version are loaded automatically. Saved settings files and analysis logs keep working.
