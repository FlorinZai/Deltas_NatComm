"""Figure 2: data gaps -> raw land change -> published (smoothed) signal -> lag. Run analysis.py first.
Left column: time series with the 'compromised data' box. Right column: scatter, cross-correlation, lag-6 summary."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy import stats
from analysis import phase_surrogates
from load import OUT, authors, gsw, mat
from xc import xcorr

plt.rcParams.update({'font.size': 7.5, 'axes.titlesize': 8, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'axes.spines.top': False, 'axes.spines.right': False})
GREY, BLUE, RED, ORANGE = '0.55', '#2b7bba', '#d62728', '#e08214'

Q, A = authors(); G = gsw()
yrs = A.index.values
raw_land = mat('original_land_19902020').sum(1)   # ref. 1, unsmoothed
pub_land = A.sum(1).values                                                                                    # ref. 1, smoothed
nodata = G['nodata'].sum(1).values
d_nodata, d_raw, d_pub = np.diff(nodata), np.diff(raw_land), np.diff(pub_land)
qsum = Q.sum(1).values
C = pd.read_csv(OUT / 'xcorr_curves.csv', index_col=0)
R = pd.read_csv(OUT / 'lag_results.csv', index_col=0)

fig = plt.figure(figsize=(7.2, 8.3))
outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.36, left=0.085, right=0.985, top=0.965, bottom=0.055)
left = outer[0].subgridspec(4, 1, hspace=0.42)
right = outer[1].subgridspec(3, 1, hspace=0.42, height_ratios=[1.15, 1, 1])


def box(ax, label=False, x1=2001.5):
    """Red frame around the years affected by data gaps (as in the original figure)."""
    lo, hi = ax.get_ylim()
    ax.add_patch(Rectangle((1989.4, lo), x1 - 1989.4, hi - lo, fill=False, ec=RED, lw=1.3, zorder=6, clip_on=False))
    if label:
        ax.text(x1 + 0.7, hi - 0.05 * (hi - lo), 'compromised\ndata', color=RED, fontsize=7.5, va='top', fontweight='bold')
    ax.set_xlim(1988.3, 2021.2)


# ---- a: no-data area
a = fig.add_subplot(left[0])
a.bar(yrs, nodata / 1e3, color=GREY, width=0.75)
a.set_ylabel('Area (10$^3$ km$^2$)'); a.set_title('a  “No data” area, all 60 deltas'); box(a, label=True)

# ---- b: annual change in no-data
a = fig.add_subplot(left[1])
a.bar(yrs[1:], d_nodata / 1e3, color=GREY, width=0.75); a.axhline(0, color='k', lw=0.5)
a.set_ylabel('Change (10$^3$ km$^2$ yr$^{-1}$)'); a.set_title('b  “No data” year-to-year change'); box(a)

# ---- c: annual change in raw land (authors' unsmoothed data)
a = fig.add_subplot(left[2])
a.bar(yrs[1:], d_raw / 1e3, color=BLUE, width=0.75); a.axhline(0, color='k', lw=0.5)
a.set_ylabel('Change (10$^3$ km$^2$ yr$^{-1}$)'); a.set_title('c  Land change, unsmoothed data of ref. 1'); box(a)

# ---- d: published smoothed land change + sediment load (= ref. 1 Fig. 3a)
a = fig.add_subplot(left[3]); b = a.twinx(); b.spines['right'].set_visible(True)
a.bar(yrs[1:], d_pub / 1e3, color=BLUE, width=0.75, label='Land change rate (dA/dt)'); a.axhline(0, color='k', lw=0.5)
b.plot(yrs, qsum / 1e9, 'o-', color='k', mfc='w', ms=3, lw=0.8, label='Fluvial sediment load')
a.set_ylim(-2.1, 2.6); b.set_ylim(2.65, 3.65)
a.set_ylabel('Change (10$^3$ km$^2$ yr$^{-1}$)'); b.set_ylabel('Sediment load (10$^9$ t yr$^{-1}$)')
a.set_title('d  Published (smoothed) land change, ref. 1'); a.set_xlabel('Year')
h1, l1 = a.get_legend_handles_labels(); h2, l2 = b.get_legend_handles_labels()
a.legend(h2 + h1, l2 + l1, fontsize=6, loc='upper right', frameon=False, handlelength=1.4, borderaxespad=0.1)
box(a)

# ---- e: raw land change vs no-data change
a = fig.add_subplot(right[0])
early = yrs[1:] <= 2000
fit = stats.linregress(d_nodata, d_raw)
xx = np.array([d_nodata.min(), d_nodata.max()])
a.plot(xx / 1e3, (fit.intercept + fit.slope * xx) / 1e3, '--', color=RED, lw=1.2, zorder=1)
a.scatter(d_nodata[~early] / 1e3, d_raw[~early] / 1e3, s=16, color='0.6', label='2001–2020', zorder=2)
a.scatter(d_nodata[early] / 1e3, d_raw[early] / 1e3, s=20, color=RED, label='1991–2000', zorder=3)
for y, off in ((1997, (-24, -2)), (1999, (5, 3))):
    i = y - 1991
    a.annotate(str(y), (d_nodata[i] / 1e3, d_raw[i] / 1e3), xytext=off, textcoords='offset points', fontsize=6, color=RED)
a.text(0.97, 0.95, 'R$^2$ = %.2f\nn = %d years' % (fit.rvalue ** 2, len(d_raw)), transform=a.transAxes, ha='right', va='top',
       fontsize=7, bbox=dict(fc='#f6ecd9', ec='0.6', lw=0.5, boxstyle='round,pad=0.3'))
a.axhline(0, color='k', lw=0.4); a.axvline(0, color='k', lw=0.4)
a.set_xlabel('“No data” change (10$^3$ km$^2$ yr$^{-1}$)'); a.set_ylabel('Land change (10$^3$ km$^2$ yr$^{-1}$)')
a.set_title('e  Land change follows data gaps (c vs b)')
a.legend(fontsize=6, loc='lower left', frameon=False, handletextpad=0.2, borderaxespad=0.2)

# ---- f: cross-correlation, published vs unsmoothed, with chance range
a = fig.add_subplot(right[1])
q = qsum[1:]
sr = np.array([xcorr(q, s, 25, 'coeff')[1] for s in phase_surrogates(d_pub, 10000)])
lags = C.index.values; keep = (lags >= -15) & (lags <= 15)
lo95, hi95 = np.nanpercentile(sr, 2.5, 0), np.nanpercentile(sr, 97.5, 0)
a.fill_between(lags[keep], lo95[keep], hi95[keep], color='0.88', lw=0, label='95% chance range')
for key, col, off, lab in [('authors', 'k', -0.17, 'Published (smoothed)'), ('gsw_raw', ORANGE, 0.17, 'Unsmoothed data of ref. 1')]:
    y = C[key].values[keep]; x = lags[keep] + off
    a.vlines(x, 0, y, color=col, lw=1.0); a.plot(x, y, 'o', color=col, ms=2.6, label=lab)
r6 = C.loc[6, 'authors']
a.plot(6 - 0.17, r6, 'o', color='k', ms=5.5, zorder=5)
a.annotate('reported lag:\n6 years', (6 - 0.17, r6), xytext=(9.3, 0.50), fontsize=6.3, ha='left', va='center',
           arrowprops=dict(arrowstyle='-', lw=0.6, color='k'))
a.axhline(0, color='k', lw=0.5); a.set_xlim(-15.6, 15.6); a.set_ylim(-0.5, 0.62)
a.set_xlabel('Lag time (years)'); a.set_ylabel('Cross-correlation coefficient')
a.set_title('f  Sediment load vs land change rate')
a.legend(fontsize=6, loc='upper left', frameon=False, handlelength=1.2, borderaxespad=0.1, labelspacing=0.25)

# ---- g: correlation at the 6-year lag for each treatment of the data
a = fig.add_subplot(right[2])
rows = [('authors', 'Published\n(smoothed)', 'k'), ('gsw_raw', 'Unsmoothed\ndata of ref. 1', ORANGE),
        ('gsw_masked', 'Gap years\nremoved', BLUE), ('authors_clean', '27 deltas\nwithout gaps', BLUE),
        ('authors_post2000', '2000–2020\nonly', BLUE)]
ypos = np.arange(len(rows))[::-1]
i6 = np.where(lags == 6)[0][0]
a.axvspan(lo95[i6], hi95[i6], color='0.88', lw=0)
a.text(lo95[i6] + 0.01, ypos[0] + 0.5, 'chance range', fontsize=6, ha='left', va='bottom', color='0.35')
for yy, (key, lab, col) in zip(ypos, rows):
    v = R.loc[key, 'r_lag6']
    a.barh(yy, v, color=col, height=0.55)
    a.text(v + (0.012 if v > 0 else -0.012), yy, ('%+.2f' % v).replace('-', '−'), ha='left' if v > 0 else 'right',
           va='center', fontsize=6.3)
    a.text(0.50, yy, '%d yr' % R.loc[key, 'peak_lag'], ha='right', va='center', fontsize=6.3)
a.text(0.50, ypos[0] + 0.5, 'peak lag', fontsize=6, ha='right', va='bottom', color='0.35')
a.axvline(0, color='k', lw=0.5); a.set_xlim(-0.47, 0.50); a.set_ylim(-0.55, len(rows) - 0.05)
a.set_yticks(ypos); a.set_yticklabels([r[1] for r in rows], fontsize=6.3); a.tick_params(axis='y', length=0)
a.spines['left'].set_visible(False); a.set_xticks([-0.2, 0, 0.2])
a.set_xlabel('Cross-correlation at the 6-year lag'); a.set_title('g  Correlation at the reported 6-year lag', x=-0.22)
pos = a.get_position(); a.set_position([pos.x0 + 0.075, pos.y0, pos.width - 0.075, pos.height])  # room for the row labels

fig.savefig(OUT / 'Figure2.png', dpi=300)
print('R2 raw land vs no-data: %.3f slope %.3f' % (fit.rvalue ** 2, fit.slope))
print('95%% chance range at lag 6: %.2f..%.2f' % (lo95[i6], hi95[i6]))
