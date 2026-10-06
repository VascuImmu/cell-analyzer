"""The analysis stages, in the order the pipeline runs them:

    background.py    1  flatfield / background estimation   (flatfield.py applies it)
    segmentation.py  2  nuclei + cell watershed, confluence decision, QC plots
    measurement.py   3  per-cell shape / intensity / junction measurements
    aggregation.py   4  one combined table, plate map, Z-scores
    analysis.py      5  plots and statistics per condition
"""
