"""Checks of the data and MATLAB logic released by Wang et al. (Zenodo 10.5281/zenodo.20281243).
Each block prints the numbers quoted in our Matters Arising. The cross-correlation and regression follow their
Figure_3.m (global) and Extra1_lag_time_all.m (per delta) line by line."""
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from load import DATA, OUT, SRC, authors, authors_own, gsw, gsw_full, mat

Q, A = authors(); G = gsw(); names = list(A.columns)
L, O, S = mat('smooth_land_19902020'), mat('original_land_19902020'), mat('smooth_sediment_load_19902020')
RES = sio.loadmat(DATA / "delta_timelag_regression_results.mat")["results"]
info = pd.read_excel(DATA / 'Data2_lag_times_hydro_forcings.xlsx')


def matlab_xcorr(y, x, maxlag):
    """MATLAB xcorr(y, x, maxlag, 'coeff') on anomalies: c(k) = sum y(n+k) x(n) / sqrt(sum x^2 sum y^2)."""
    y = y - y.mean(); x = x - x.mean(); n = len(y); den = np.sqrt((x ** 2).sum() * (y ** 2).sum())
    lags = np.arange(-maxlag, maxlag + 1)
    c = np.array([(y[k:] * x[:n - k]).sum() if k >= 0 else (y[:n + k] * x[-k:]).sum() for k in lags]) / den
    return lags, c


def figure3(land, sed):
    """Their Figure_3.m: xcorr(diff(A), Q(2:end)); regression of dA(lag+1:end) on Q(1:end-lag-1)."""
    q = sed.sum(1); d = np.diff(land.sum(1))
    lags, c = matlab_xcorr(d, q[1:], 30); k = lags[np.argmax(c)]
    p_theirs = stats.linregress(q[:-k - 1], d[k:]).pvalue if k > 0 else np.nan          # Q one year earlier than in xcorr
    p_consistent = stats.linregress(q[1:][:len(d) - k], d[k:]).pvalue if k > 0 else np.nan
    return k, c.max(), c[lags == 6][0], p_theirs, p_consistent


def delta_lag(land, sed):
    """Their Extra1_lag_time_all.m for one delta: xcorr(diff(A), Q(1:end-1)), max over +-28 years."""
    s = sed[:-1]; d = np.diff(land); n = len(s)
    lags, c = matlab_xcorr(d, s, 28); k = lags[np.argmax(c)]
    xs, ys = (s[:n - k], d[k:]) if k > 0 else ((s[-k:], d[:n + k]) if k < 0 else (s, d))
    return int(k), stats.linregress(xs, ys).pvalue


print('== 1. Released files')
print('smoothed land .mat equals the released land table: max diff %.3f km2' % np.abs(L - A.values).max())
# their unsmoothed file is in the same (shifted) column order as the smoothed one: undo the shift before comparing
own = {n: O[:, names.index(SRC.get(n, n))] for n in names}
print('unsmoothed land .mat equals our GSW extraction (not water + seasonal) once the column shift of section 6 is undone: '
      'max diff %.3f km2' % max(np.abs(own[n] - G['land'][n].values).max() for n in names))

print('\n== 2. "No data" in the 60 polygons')
f = G['nodata_frac']
print('deltas with >10%% no data in at least one year: %d; >50%%: %d' % ((f.max() > 0.1).sum(), (f.max() > 0.5).sum()))
print('delta-years >10%% no data: 1990-2000 %.0f%%, 2001-2020 %.0f%%' % (100 * (f.loc[:2000] > 0.1).values.mean(), 100 * (f.loc[2001:] > 0.1).values.mean()))
flag = (G['notwater'] == 0) | (G['seasonal_water'] == 0)          # the only years their code treats as missing
gap = f > 0.1
print('delta-years with >10%% no data: %d; flagged missing by their code (area exactly zero): %d, all of them 100%% unobserved: %s'
      % (gap.values.sum(), (gap & flag).values.sum(), bool(f.values[flag.values].min() > 0.999)))
ref = G['land'].where(f < 0.01).loc[2001:].median()               # typical land of each delta in gap-free years
kept = gap & ~flag
print('kept as data: %d delta-years; land relative to the 2001-2020 median: %.0f%% before smoothing, %.0f%% after (medians)'
      % (kept.values.sum(), 100 * (G['land'] / ref).where(kept).stack().median(), 100 * (authors_own() / ref).where(kept).stack().median()))
d = 'Ganges-Brahmaputra'
print('Ganges 1994-1999: no data %s %%' % np.round(100 * f.loc[1994:1999, d].values).astype(int))
print('   unsmoothed land %s km2' % np.round(O[4:10, names.index(d)]).astype(int))
print('   published land  %s km2' % np.round(L[4:10, names.index(d)]).astype(int))
r = stats.linregress(np.diff(G['nodata'].sum(1).values), np.diff(O.sum(1)))
print('unsmoothed annual land change vs annual no-data change: R2 = %.2f (n = %d)' % (r.rvalue ** 2, len(O) - 1))

print('\n== 3. Global lag (their Figure_3.m)')
k, cm, c6, p1, p2 = figure3(L, S)
print('published (smoothed): lag %d, peak r = %.3f (paper: 0.218); P = %.3f with their regression pairing, %.3f with the xcorr pairing' % (k, cm, p1, p2))
k, cm, c6, _, _ = figure3(O, S)
print('their unsmoothed land: highest peak at lag %d (r = %.3f); r at 6 years = %.3f' % (k, cm, c6))
k, cm, c6, _, _ = figure3(O, mat('original_sediment_load_19902020'))
print('their unsmoothed land and unsmoothed sediment: highest peak at lag %d (r = %.3f); r at 6 years = %.3f' % (k, cm, c6))
ch = info.iloc[:, 5].values
for lab, cols in (('24 deltas, increasing load', ch > 0), ('36 deltas, decreasing load', ch < 0)):
    k, cm, _, p1, _ = figure3(L[:, cols], S[:, cols]); print('%s: lag %d, r = %.3f, P = %.3f' % (lab, k, cm, p1))

print('\n== 4. Land growth rates (published series)')
tr = lambda y, t0: np.polyfit(np.arange(t0, 2021), y[t0 - 1990:], 1)[0]
tot = A.sum(1).values
print('global trend 1990-2020: %.0f km2/yr; 2000-2020: %.0f km2/yr; change 1996 -> 2001: %+.0f km2' % (tr(tot, 1990), tr(tot, 2000), tot[11] - tot[6]))
i = names.index('Danube'); print('Danube trend 1990-2020: %.1f km2/yr; 2000-2020: %.1f km2/yr' % (tr(L[:, i], 1990), tr(L[:, i], 2000)))

print('\n== 5. Delta-scale lags (their Extra1_lag_time_all.m)')
rep = np.array([delta_lag(L[:, j], S[:, j]) for j in range(60)])
sel = (RES[:, 0] >= 0) & (RES[:, 3] <= 0.10)
print('lags reproduced: %d/60' % (rep[:, 0] == RES[:, 0]).sum())
print('mean lag, all 60 deltas: %.1f yr (paper: 2.9); %d deltas with lag >= 0 and P <= 0.10: %.1f yr (paper: 6.8)' % (RES[:, 0].mean(), sel.sum(), RES[sel, 0].mean()))

print('\n== 6. Land series assigned to the wrong delta')
# GSW_100.m reads Liaohe, Minjiang from columns 32-33 and Zambezi...Brahmani from 34-41, but GSW_100.xlsx is in polygon order
nw = pd.read_excel(DATA / 'GSW_100.xlsx', sheet_name='not_water', header=None).values
ours = gsw_full()['notwater'].values
exact = [np.abs(ours - nw[:, [j]]).max(0).argmin() == j for j in range(60)]
print('GSW_100.xlsx columns in polygon order: %d/60' % sum(exact))
land = G['land'].values
holds = [int(np.abs(land - O[:, [j]]).max(0).argmin()) for j in range(60)]     # polygon whose series sits in column j
shifted = {names[j]: names[p] for j, p in enumerate(holds) if p != j}
print("columns of their land files that hold another delta's series: %d" % len(shifted))
for lab, poly in shifted.items():
    print('   published as %-16s = polygon %-16s (published mean %7.1f km2; own polygon %7.1f km2)'
          % (lab, poly, L[:, names.index(lab)].mean(), land[:, names.index(lab)].mean()))
assert shifted == {v: k for k, v in SRC.items()}
mean_pub, mean_own = L.mean(0), land.mean(0)
Lc = L.copy()
for delta, col in SRC.items():          # the correct series of each delta sits in the column of the delta that read it
    Lc[:, names.index(delta)] = L[:, names.index(col)]
cor = np.array([delta_lag(Lc[:, j], S[:, j]) for j in range(60)])
for delta in ('Liaohe', 'Murray', 'Paraiba Do Sul'):
    j = names.index(delta); print('%-15s published lag %3d yr (P = %.2f) -> with its own polygon %3d yr (P = %.2f)' % (delta, RES[j, 0], RES[j, 3], cor[j, 0], cor[j, 1]))
selc = (cor[:, 0] >= 0) & (cor[:, 1] <= 0.10)
print('selected deltas with corrected series: %d, mean lag %.1f yr' % (selc.sum(), cor[selc, 0].mean()))
pd.DataFrame({'published_mean_km2': mean_pub, 'own_polygon_mean_km2': mean_own, 'published_lag': RES[:, 0], 'corrected_lag': cor[:, 0]},
             index=names).round(2).to_csv(OUT / 'per_delta_check.csv')
