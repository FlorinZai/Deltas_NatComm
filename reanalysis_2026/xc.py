"""Cross-correlation variants matching MATLAB xcorr/crosscorr conventions."""
import numpy as np


def xcorr(q, a, maxlag=30, kind='coeff'):
    """r[k] = corr(q[t], a[t+k]); k>0 means land change lags sediment by k years.
    kind='coeff'  : MATLAB xcorr(...,'coeff') on demeaned series (1/N normalisation, as in their Eq. 2-3)
    kind='pearson': Pearson r on the overlapping segment only."""
    q = np.asarray(q, float); a = np.asarray(a, float)
    n = len(q); lags = np.arange(-maxlag, maxlag + 1); r = np.full(len(lags), np.nan)
    qd, ad = q - q.mean(), a - a.mean()
    den = np.sqrt((qd ** 2).sum() * (ad ** 2).sum())
    for i, k in enumerate(lags):
        x, y = (q[:n - k], a[k:]) if k >= 0 else (q[-k:], a[:n + k])
        xd, yd = (qd[:n - k], ad[k:]) if k >= 0 else (qd[-k:], ad[:n + k])
        if len(x) < 3 or abs(k) >= n:
            continue
        r[i] = (xd * yd).sum() / den if kind == 'coeff' else np.corrcoef(x, y)[0, 1]
    return lags, r
