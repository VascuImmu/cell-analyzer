"""
Cell Analyzer -- segmentation and measurement of cells in microscopy images (.lif, .czi, .tif, .nd2).

    python -m cell_analyzer                    start the window (GUI)
    python -m cell_analyzer.pipeline --help    run without window

Layout
    config.py      every parameter (defaults, labels, help, validation); the GUI is built from it
    gui.py         the window
    pipeline.py    runs the stages in order, writes the analysis log; command-line entry point
    io_utils.py    finding files, name filters, scenes, image loading, well parsing
    stages/        background, flatfield, segmentation, measurement, aggregation, analysis
"""
from .config import APP_NAME, PIPELINE_VERSION as __version__  # noqa: F401
