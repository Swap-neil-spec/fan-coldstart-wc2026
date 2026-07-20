"""Statistical primitives used by the pre-registered tests, implemented on
numpy + stdlib so the analysis does not require scipy.stats. Validated in
tests against known values.
"""
import math
import numpy as np


def rankdata(x):
    """Average-rank of x (ties share the mean of their positions)."""
    x = np.asarray(x, float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def pearson(a, b):
    a = np.asarray(a, float) - np.mean(a)
    b = np.asarray(b, float) - np.mean(b)
    denom = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denom if denom > 0 else 0.0


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def normal_sf(z):
    """Upper tail P(Z > z) of the standard normal."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def chi2_sf_1df(stat):
    """Survival function of chi-square with one degree of freedom."""
    if stat <= 0:
        return 1.0
    return 2.0 * normal_sf(math.sqrt(stat))


def wilcoxon_signed_rank(diffs):
    """Two-sided paired Wilcoxon signed-rank test, normal approximation with a
    continuity correction. Returns (p_value, z). Zeros are dropped."""
    d = np.asarray(diffs, float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0.0
    r = rankdata(np.abs(d))
    w_plus = float(np.sum(r[d > 0]))
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if sigma == 0:
        return 1.0, 0.0
    z = (w_plus - mu) / sigma
    zc = (abs(w_plus - mu) - 0.5) / sigma
    return min(1.0, 2.0 * normal_sf(zc)), z


def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg step-up. Returns (reject_mask, adjusted_pvalues)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    reject = np.zeros(m, bool)
    passed = p[order] <= q * (np.arange(1, m + 1) / m)
    idx = np.where(passed)[0]
    if len(idx) > 0:
        reject[order[: idx.max() + 1]] = True
    padj = np.empty(m, float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        prev = min(prev, p[order[i]] * m / (i + 1))
        padj[order[i]] = prev
    return reject, padj
