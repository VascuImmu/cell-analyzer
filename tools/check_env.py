"""Run by the installers: verifies that every package Cell Analyzer needs imports cleanly."""
import logging
import sys
logging.getLogger("bfio").setLevel(logging.ERROR)  # harmless "Java backend not available" notice
failed = []
for mod in ["numpy", "scipy.ndimage", "skimage.measure", "skimage.segmentation", "skimage.morphology",
            "skimage.feature", "pandas", "tifffile", "matplotlib", "dask.array", "aicsimageio",
            "readlif.reader", "aicspylibczi", "openpyxl", "cmap", "tkinter"]:
    try:
        __import__(mod)
    except Exception as e:  # noqa
        failed.append(f"{mod}: {type(e).__name__}: {str(e).splitlines()[-1] if str(e) else ''}")
import numpy
print(f"  python {sys.version.split()[0]}, numpy {numpy.__version__}")
if failed:
    print("  ✗ these packages do not load:")
    for f in failed:
        print("     -", f)
    sys.exit(1)
print("  ✓ all packages load")
