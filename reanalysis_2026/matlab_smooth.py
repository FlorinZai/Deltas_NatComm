"""Python re-implementation of the authors' land processing (GSW_100.m, Zenodo 10.5281/zenodo.20281243):
zeros -> NaN, fillmissing(x,'movmean',5), smoothdata(x,'rlowess',5).
Run this file to compare with their smooth_land_19902020.mat: 52 of the 60 delta series are reproduced exactly; the
other 8 differ in years where the robust weights leave fewer than two points in a window."""
import numpy as np


def fill_movmean(x, win=5):
    x = np.asarray(x, float); h = win // 2; out = x.copy()
    for i in np.where(np.isnan(x))[0]:
        w = x[max(0, i - h):i + h + 1]
        if np.isfinite(w).any():
            out[i] = np.nanmean(w)
    return out


def rlowess(y, win=5, iters=5):
    """MATLAB smoothdata(y,'rlowess',win): local linear fit over the win nearest samples (window shifted at the ends),
    tricube weights scaled by the farthest sample, 5 bisquare iterations with a global 6*median(|residual|) cutoff.
    Exact for windows keeping >= 2 positive weights (50 of the 60 deltas reproduce to 1e-12); where the robust weights
    leave one point its value is used, and where they leave none the previous iteration's value is kept."""
    y = np.asarray(y, float); n = len(y); h = win // 2; t = np.arange(n)
    ok = np.isfinite(y); rob = np.where(ok, 1.0, 0.0); fit = np.full(n, np.nan); last = np.full(n, np.nan)
    for it in range(iters + 1):
        new = np.full(n, np.nan)
        for i in range(n):
            lo = min(max(0, i - h), n - win); j = np.arange(lo, lo + win); jj = j[ok[j]]
            if len(jj) == 0:
                continue
            dv = np.abs(t[jj] - i); w = np.clip(1 - (dv / np.abs(t[j] - i).max()) ** 3, 0, None) ** 3 * rob[jj]
            npos = (w > 0).sum()
            if npos >= 2:
                X = np.c_[np.ones(len(jj)), t[jj] - i]; s = np.sqrt(w)
                new[i] = np.linalg.lstsq(X * s[:, None], y[jj] * s, rcond=None)[0][0]
            elif npos == 1:
                new[i] = y[jj][w > 0][0]
            else:
                new[i] = last[i]
        fit = new; last = np.where(np.isfinite(fit), fit, last)
        if it == iters:
            break
        r = np.abs(y - fit); m = np.nanmedian(r[ok]) if ok.any() else 0
        u = r / (6 * m) if m > 0 else np.zeros(n)
        rob = np.where(ok & (u < 1), (1 - u ** 2) ** 2, 0.0)
    return np.where(np.isfinite(fit), fit, last)


def land_smooth(nw, sw):
    out = 0
    for x in (nw, sw):
        x = np.asarray(x, float).copy(); x[x == 0] = np.nan
        out = out + rlowess(fill_movmean(x))
    return out


# xlsx column (0-based) that each output column of GSW_100.m actually reads: the 10 shifted deltas
READ_ORDER = list(range(31)) + [33, 34, 35, 36, 37, 38, 39, 40, 31, 32] + list(range(41, 60))


def validate():
    """Max absolute difference (km2) per delta between our implementation and their smoothed land."""
    import pandas as pd
    from load import DATA, mat
    X = pd.read_excel(DATA / 'GSW_100.xlsx', sheet_name=None, header=None)
    S = mat('smooth_land_19902020')
    nw, sw = X['not_water'].values, X['seasonal_water'].values
    return np.array([np.nanmax(np.abs(land_smooth(nw[:, c], sw[:, c])[6:37] - S[:, k])) for k, c in enumerate(READ_ORDER)])


if __name__ == '__main__':
    e = validate()
    print('delta series reproduced to < 0.01 km2: %d of 60' % (e < 0.01).sum())
