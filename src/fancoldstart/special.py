"""Special functions on numpy arrays, stdlib-backed so the core needs no scipy.

Everything here is validated in tests/test_special.py against closed-form
identities, so the pipeline does not depend on a scipy wheel being available.
"""
import math
import numpy as np

_lgamma = np.frompyfunc(math.lgamma, 1, 1)


def gammaln(x):
    """Log gamma, vectorized over numpy arrays via math.lgamma.

    Always returns a numpy float array (0-d for scalar input), so callers can
    mix scalar and array arguments freely."""
    return np.asarray(_lgamma(np.asarray(x, dtype=float)), dtype=float)


def betaln(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def logsumexp2(a, b):
    """Stable log(exp(a) + exp(b)) elementwise, tolerating -inf entries."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.maximum(a, b)
    m_safe = np.where(np.isfinite(m), m, 0.0)
    ea = np.where(np.isfinite(a), np.exp(a - m_safe), 0.0)
    eb = np.where(np.isfinite(b), np.exp(b - m_safe), 0.0)
    with np.errstate(divide="ignore"):
        out = m_safe + np.log(ea + eb)
    # if both were -inf, result is -inf
    both_ninf = ~np.isfinite(a) & ~np.isfinite(b)
    return np.where(both_ninf, -np.inf, out)


def hyp2f1(a, b, c, z, tol=1e-13, max_terms=5000):
    """Gauss hypergeometric 2F1(a, b; c; z) by power series.

    Convergent for |z| < 1, which is the regime used by the BG/NBD conditional
    expectation, where z = t / (alpha + T + t) lies in (0, 1). Vectorized over z.
    Validated in tests against the identities
        2F1(a, b; b; z) = (1 - z) ** (-a)
        2F1(1, 1; 2; z) = -ln(1 - z) / z .
    """
    z = np.asarray(z, dtype=float)
    term = np.ones_like(z)
    total = np.ones_like(z)
    for n in range(0, max_terms):
        term = term * ((a + n) * (b + n) / ((c + n) * (1.0 + n))) * z
        total = total + term
        if np.all(np.abs(term) <= tol * (np.abs(total) + tol)):
            break
    return total
