"""Re-analysis of Wang et al. (2025, Nat Commun 16:5573) global 6-year lag.

Variants of the global land series (1990-2020, km2):
  authors      : sum of the authors' released 'active land area' (Zenodo 15965855) -> reproduces their Fig. 3a
  gsw_raw      : our GEE re-extraction with their polygons/classes (notwater + seasonal), no cleaning
  gsw_masked   : as gsw_raw, delta-years with no-data > THR set missing and linearly interpolated
  authors_clean: authors' series summed over deltas that never exceed THR no-data in 1990-2020
For each: cross-correlation (their Eq. 3, MATLAB 'coeff' normalisation) between total sediment load
and global dA/dt, and a surrogate test that keeps the autocorrelation of dA/dt.
"""
import numpy as np
import pandas as pd
from load import OUT, authors, gsw
from xc import xcorr

THR = 0.10
N_SURR = 10000
LAGS = np.arange(-25, 26)
rng = np.random.default_rng(1)


def phase_surrogates(x, n):
    """Fourier phase-randomised surrogates with the same power spectrum (autocorrelation) as x."""
    x = np.asarray(x, float); m = len(x)
    X = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, (n, len(X))); ph[:, 0] = 0
    if m % 2 == 0:
        ph[:, -1] = 0
    return np.fft.irfft(np.abs(X) * np.exp(1j * ph), m) + x.mean()


def lag_test(q, a):
    """Peak r over lags 1..15 and surrogate p-values (for r(6) and for the max over 1..15)."""
    # Global pairing of their Figure_3.m (Zenodo 20281243): xcorr(diff(A), Q(2:end)), i.e. dA(t) = A(t) - A(t-1) with Q(t).
    # (Their per-delta Extra1_lag_time_all.m pairs dA(t) with Q(t-1) instead.)
    q = q[1:]; dadt = np.diff(a)
    maxlag = min(LAGS.max(), len(dadt) - 3)
    lags, r = xcorr(q, dadt, maxlag, "coeff")
    pos = (lags >= 1) & (lags <= 15)
    k = lags[pos][np.nanargmax(r[pos])]
    r6, rmax = r[lags == 6][0], np.nanmax(r[pos])
    S = phase_surrogates(dadt, N_SURR)
    sr = np.array([xcorr(q, s, maxlag, "coeff")[1] for s in S])
    p6 = np.mean(sr[:, lags == 6][:, 0] >= r6)
    pmax = np.mean(np.nanmax(sr[:, pos], axis=1) >= rmax)
    # naive Pearson on the lagged pairs at lag 6 (as in their Fig. 3c)
    from scipy.stats import pearsonr
    rp, pp = pearsonr(q[:-6], dadt[6:])
    return dict(peak_lag=int(k), r_peak=round(rmax, 3), r_lag6=round(r6, 3), r_lag0=round(r[lags == 0][0], 3),
                p_surr_lag6=round(p6, 3), p_surr_maxlag=round(pmax, 3), pearson_lag6=round(rp, 3),
                p_naive_lag6=round(pp, 3)), lags, r


def series():
    Q, A = authors(); G = gsw()
    f = G['nodata_frac']
    land = G['land']
    masked = land.mask(f > THR).interpolate(limit_direction='both')
    clean = f.max() <= THR
    return Q.sum(1).values, {
        'authors': A.sum(1),
        'gsw_raw': land.sum(1),
        'gsw_masked': masked.sum(1),
        'authors_clean': A.loc[:, clean].sum(1),
        'authors_post2000': A.sum(1),
    }, clean


if __name__ == '__main__':
    q, S, clean = series()
    rows, curves = {}, {}
    for k, a in S.items():
        qq, aa = (q, a.values)
        if k == 'authors_post2000':
            qq, aa = q[a.index >= 2000], a.loc[2000:].values
        rows[k], lags, r = lag_test(qq, aa)
        curves[k] = pd.Series(r, lags).reindex(LAGS)
    out = pd.DataFrame(rows).T
    out['n_deltas'] = [60, 60, 60, int(clean.sum()), 60]
    pd.set_option('display.width', 200)
    print(out)
    OUT.mkdir(exist_ok=True)
    out.to_csv(OUT / 'lag_results.csv')
    pd.DataFrame(curves).to_csv(OUT / 'xcorr_curves.csv')
    pd.DataFrame(S).to_csv(OUT / 'global_land_series.csv')
