# Deltas_NatComm

Code and data behind our Matters Arising on Wang et al. (2025), "Response time of global deltas to changes in
fluvial sediment supply", *Nature Communications* 16, 5573.

## Re-analysis (`reanalysis_2026/`)

This folder reproduces every number and Figure 2 of the Matters Arising, using the data and MATLAB logic released
by the authors.

Requirements: Python 3 with `numpy`, `pandas`, `scipy`, `matplotlib`, `openpyxl`.

Run from inside `reanalysis_2026/`, in this order:

| Script | What it does |
|---|---|
| `analysis.py` | Cross-correlation between sediment load and global land change rate for the published (smoothed) data, the unsmoothed data, the data with gap years removed (without and with the authors' smoothing), the 27 deltas without gaps, and 2000-2020 only. Significance from 10,000 phase-randomised surrogates. Writes `output/lag_results.csv`, `output/xcorr_curves.csv` and `output/sensitivity_threshold.csv` (sensitivity to the "no data" threshold). |
| `check_authors_code.py` | Checks of the authors' released files and code: "no data" statistics, their Figure 3 lag and P-values, growth rates, delta-scale lags, and the land series assigned to the wrong deltas. Prints the numbers and writes `output/per_delta_check.csv`. |
| `figure2.py` | Figure 2 of the Matters Arising (`output/Figure2.png`). Needs the output of `analysis.py`. |

`load.py` and `xc.py` hold the data loaders and the cross-correlation (Eq. 3 of the paper, MATLAB `xcorr(...,'coeff')`).
`matlab_smooth.py` is a Python implementation of the authors' land processing (zeros set to missing, `fillmissing(...,'movmean',5)`,
`smoothdata(...,'rlowess',5)`); run on its own, it compares the result with their smoothed land (52 of 60 delta series identical).

### Data

- `GSW_YearlyHistory_nodata_60deltas.csv` (repository root): area of each JRC Global Surface Water class, including
  "no data", per delta polygon and year (1984-2021). Extracted with the authors' polygons and their Earth Engine code,
  modified only to export the "no data" class: https://code.earthengine.google.com/14e092eb0d11b226c5357a3b4c814822
- `reanalysis_2026/data/wang2025/`: files released by Wang et al. on Zenodo under CC BY 4.0, copied here unchanged so
  that the scripts run without downloads.
  - `1_Annual_sediment_load.xlsx`, `2_Annual_active_land_area.xlsx`: https://doi.org/10.5281/zenodo.15965855
  - `GSW_100.xlsx`, `original_land_19902020.mat`, `smooth_land_19902020.mat`, `smooth_sediment_load_19902020.mat`, `original_sediment_load_19902020.mat`,
    `delta_timelag_regression_results.mat`, `Data2_lag_times_hydro_forcings.xlsx`: https://doi.org/10.5281/zenodo.20281243

## Earlier analysis (2025, repository root)

The files in the repository root are our first analysis from October 2025, made before the authors released their
unsmoothed data and code. It is kept for reference and is superseded by `reanalysis_2026/`.

Prerequisite: run https://code.earthengine.google.com/a8507f4e65d43a1b8a33c06462bc9a20 to get the p1 and p2 yearly
delta area changes. `data_analysis.ipynb` uses `GSW_YearlyHistory_nodata_60deltas.csv` (p1 and p2 combined) and
correlates the "no data" change in the JRC GSW data with a land change smoothed by our own reconstruction of the
authors' method (`reanalysis_matlab.mlx`). The R2 = 0.83 obtained there refers to that reconstruction, not to the
authors' published series; with their released data the corresponding result is R2 = 0.60 for the unsmoothed series
(`check_authors_code.py`).
