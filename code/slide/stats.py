"""Non-parametric comparison of detectors (Demsar-style protocol)."""
import numpy as np
from scipy import stats


def holm(pvals):
    p = np.asarray(pvals, float)
    o = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def wilcoxon_rb(a, b):
    """two-sided Wilcoxon signed-rank test of a vs b, with matched-pairs rank-biserial effect size."""
    d = np.asarray(a) - np.asarray(b)
    d = d[d != 0]
    if len(d) == 0:
        return 1.0, 0.0, 0
    res = stats.wilcoxon(d, alternative="two-sided", method="exact" if len(d) <= 25 else "approx")
    r = stats.rankdata(np.abs(d))
    rb = (r[d > 0].sum() - r[d < 0].sum()) / r.sum()
    return float(res.pvalue), float(rb), len(d)


def friedman_nemenyi(M, alpha=0.05):
    """M: N blocks x k methods (higher is better). returns chi2, p, mean ranks, critical difference."""
    N, k = M.shape
    chi, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.vstack([stats.rankdata(-row) for row in M]).mean(0)
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * N))
    return float(chi), float(p), ranks, float(cd)


def mean_ci(x, level=0.95):
    x = np.asarray(x, float)
    m = x.mean()
    if len(x) < 2:
        return m, m, m
    h = stats.t.ppf(0.5 + level / 2, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h
