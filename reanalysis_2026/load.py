"""Shared loaders: data released by Wang et al. (Zenodo 10.5281/zenodo.15965855 and 10.5281/zenodo.20281243)
and our Earth Engine extraction of all JRC GSW classes, including "no data", in their polygons."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data' / 'wang2025'
OUT = HERE / 'output'
GSW = HERE.parent / 'GSW_YearlyHistory_nodata_60deltas.csv'
YEARS = np.arange(1990, 2021)


def authors():
    """Annual sediment load (t/yr) and smoothed 'active delta land area' (km2) as released by Wang et al."""
    Q = pd.read_excel(DATA / '1_Annual_sediment_load.xlsx', index_col=0)
    A = pd.read_excel(DATA / '2_Annual_active_land_area.xlsx', index_col=0)
    Q.columns = A.columns = [c.strip() for c in A.columns]
    return Q, A


def mat(name):
    """A 31 x 60 matrix (1990-2020, 60 deltas) from the authors' MATLAB files, e.g. 'original_land_19902020'."""
    return sio.loadmat(DATA / f'{name}.mat')[name]


def gsw_full():
    """Per-class areas (km2), 1984-2021, by polygon, with the authors' delta names (38 x 60 DataFrames)."""
    g = pd.read_csv(GSW)
    names = authors()[1].columns
    return {c: pd.DataFrame({n: json.loads(v) for n, v in zip(names, g[c])}, index=np.arange(1984, 2022))
            for c in ['nodata', 'notwater', 'seasonal_water', 'water']}


def gsw():
    """As gsw_full() for 1990-2020, plus land (= not water + seasonal water, as in their code), total and no-data fraction."""
    out = gsw_full()
    out['land'] = out['notwater'] + out['seasonal_water']
    out['total'] = out['land'] + out['water'] + out['nodata']
    out['nodata_frac'] = out['nodata'] / out['total']
    return {k: v.loc[1990:2020] for k, v in out.items()}
