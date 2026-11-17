"""Re-analysis of the global 6-year lag of Wang et al. (2025, Nat Commun 16:5573).

Cross-correlation between sediment load and the global land change rate, as in the authors' Figure_3.m
(xcorr(diff(A), Q(2:end), 'coeff')), for these treatments of the data:
  published        : the authors' smoothed land and smoothed sediment load (reproduces their Fig. 3)
  unsmoothed       : their unsmoothed land, smoothed sediment load
  unsmoothed_both  : their unsmoothed land and unsmoothed sediment load
  gaps_removed     : delta-years with more than THR "no data" removed and interpolated, not smoothed
  gaps_removed_smo : as gaps_removed, then the authors' smoothing (rlowess, 5 points)
  gapfree_deltas   : published land of the deltas that never exceed THR "no data", with the sediment load of the same deltas
  post2000         : published data, 2000-2020 only
Significance: phase-randomised surrogates of the land change series, which keep its autocorrelation.
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from load import OUT, authors, authors_own, gsw, gsw_full, mat
from matlab_smooth import rlowess
from xc import xcorr

THR = 0.10
N_SURR = 10000
LAGS = np.arange(-25, 26)
rng = np.random.default_rng(1)


def phase_surrogates(x, n):
    """Fourier phase-randomised surrogates with the same power spectrum (autocorrelation) as x."""
    x = np.asarray(x, float); m = len(x)
    X = np.fft.rfft(x - x.mean())
    ph = np.exp(1j * rng.uniform(0, 2 * np.pi, (n, len(X)))); ph[:, 0] = 1
    if m % 2 == 0:
        ph[:, -1] = rng.choice([-1.0, 1.0], n)      # the Nyquist term must stay real
    return np.fft.irfft(np.abs(X) * ph, m) + x.mean()


def lag_test(q, a, n_surr=N_SURR):
    """Peak over lags 1..15, r at lag 6, and P-values: surrogate (lag 6 alone; maximum over lags 1..15) and Pearson at lag 6."""
    q = np.asarray(q, float)[1:]; dadt = np.diff(np.asarray(a, float))     # dA(t) = A(t) - A(t-1) paired with Q(t)
    maxlag = min(LAGS.max(), len(dadt) - 3)
    lags, r = xcorr(q, dadt, maxlag, 'coeff')
    pos = (lags >= 1) & (lags <= 15)
    r6, rmax = r[lags == 6][0], np.nanmax(r[pos])
    sr = np.array([xcorr(q, s, maxlag, 'coeff')[1] for s in phase_surrogates(dadt, n_surr)])
    res = dict(peak_lag=int(lags[pos][np.nanargmax(r[pos])]), r_peak=round(rmax, 3), r_lag6=round(r6, 3),
               p_surr_lag6=round(np.mean(sr[:, lags == 6][:, 0] >= r6), 3),
               p_surr_maxlag=round(np.mean(np.nanmax(sr[:, pos], axis=1) >= rmax), 3),
               p_pearson_lag6=round(pearsonr(q[:-6], dadt[6:])[1], 3))
    return res, pd.Series(r, lags).reindex(LAGS)


def gaps_removed(thr, smooth):
    """Global land (1990-2020) after removing delta-years with more than thr "no data".
    Per class, as in their code: zeros are missing; here gap years are missing too; then linear interpolation."""
    F = gsw_full()
    frac = F['nodata'] / (F['nodata'] + F['notwater'] + F['seasonal_water'] + F['water'])
    tot = 0
    for c in ('notwater', 'seasonal_water'):
        x = F[c].where(F[c] != 0).mask(frac > thr).interpolate(limit_direction='both')
        tot = tot + (x.apply(lambda s: pd.Series(rlowess(s.values), index=s.index)) if smooth else x)
    return tot.loc[1990:2020].sum(1).values


def gapfree(thr):
    """Sediment load and published land (attached to the right polygons) of the deltas that never exceed thr "no data"."""
    Q = authors()[0]; keep = gsw()['nodata_frac'].max() <= thr
    return Q.loc[:, keep].sum(1).values, authors_own().loc[:, keep].sum(1).values, int(keep.sum())


def variants():
    Q, A = authors()
    q, a = Q.sum(1).values, A.sum(1).values
    raw = mat('original_land_19902020').sum(1)
    qg, ag, n = gapfree(THR)
    return {
        'published': (q, a, 60),
        'unsmoothed': (q, raw, 60),
        'unsmoothed_both': (mat('original_sediment_load_19902020').sum(1), raw, 60),
        'gaps_removed': (q, gaps_removed(THR, False), 60),
        'gaps_removed_smo': (q, gaps_removed(THR, True), 60),
        'gapfree_deltas': (qg, ag, n),
        'post2000': (q[10:], a[10:], 60),
    }


if __name__ == '__main__':
    V = variants()
    rows, curves = {}, {}
    for k, (q, a, n) in V.items():
        rows[k], curves[k] = lag_test(q, a)
        rows[k]['n_deltas'] = n
    out = pd.DataFrame(rows).T
    pd.set_option('display.width', 200)
    print(out)

    # sensitivity to the "no data" threshold (correlation at the 6-year lag and peak lag over 1..15 years)
    q = authors()[0].sum(1).values
    sens = []
    for thr in (0.05, 0.10, 0.20, 0.50):
        qg, ag, n = gapfree(thr)
        for name, (qq, aa) in {'gaps_removed': (q, gaps_removed(thr, False)), 'gaps_removed_smo': (q, gaps_removed(thr, True)),
                               'gapfree_deltas': (qg, ag)}.items():
            r, _ = lag_test(qq, aa, n_surr=1)
            sens.append(dict(threshold=thr, treatment=name, r_lag6=r['r_lag6'], peak_lag=r['peak_lag'], r_peak=r['r_peak'],
                             n_deltas=n if name == 'gapfree_deltas' else 60))
    sens = pd.DataFrame(sens)
    print('\nSensitivity to the "no data" threshold:\n', sens.pivot(index='threshold', columns='treatment', values='r_lag6'))

    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / 'lag_results.csv')
    pd.DataFrame(curves).to_csv(OUT / 'xcorr_curves.csv')
    sens.to_csv(OUT / 'sensitivity_threshold.csv', index=False)
    pd.DataFrame({k: v[1] for k, v in V.items() if len(v[1]) == 31}, index=np.arange(1990, 2021)).to_csv(OUT / 'global_land_series.csv')
