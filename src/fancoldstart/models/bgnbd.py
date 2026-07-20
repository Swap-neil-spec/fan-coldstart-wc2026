"""BG/NBD latent-attrition model (Fader, Hardie and Lee 2005).

This is the plain baseline backbone. It implements the closed-form individual
log-likelihood, the maximum-likelihood fit of the population parameters
(r, alpha, a, b), the probability a fan is still active at the end of the
calibration window, and the conditional expected number of future events over
a horizon tau. The expected future engagement is the value estimate used
throughout, following the value-proxy definition in the paper (Section 3.2).

Reference: Fader, Hardie and Lee (2005), "Counting Your Customers the Easy Way:
An Alternative to the Pareto/NBD Model," Marketing Science 24(2):275-284.
"""
import numpy as np

from ..special import gammaln, logsumexp2, hyp2f1
from ..optimize import minimize_nelder_mead


def _neg_log_likelihood(params, x, t_x, T):
    r, alpha, a, b = params
    if min(r, alpha, a, b) <= 0:
        return 1e12
    x = np.asarray(x, float)
    t_x = np.asarray(t_x, float)
    T = np.asarray(T, float)

    ln_A1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    ln_A2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
    ln_A3 = -(r + x) * np.log(alpha + T)
    with np.errstate(divide="ignore", invalid="ignore"):
        ln_A4 = np.where(
            x > 0,
            np.log(a) - np.log(b + x - 1.0) - (r + x) * np.log(alpha + t_x),
            -np.inf,
        )
    ll = ln_A1 + ln_A2 + logsumexp2(ln_A3, ln_A4)
    total = np.sum(ll)
    if not np.isfinite(total):
        return 1e12
    return -total


def fit(x, t_x, T, init=(1.0, 1.0, 1.0, 1.0), reg=1e-3):
    """Maximum-likelihood fit of (r, alpha, a, b). Optimizes in log space so the
    parameters stay positive. A weak ridge penalty on the log-parameters keeps
    the fit from wandering along a flat likelihood ridge when the sample is
    dominated by cold-start fans with little repeat-transaction information; it
    is negligible where the parameters are identified and only pins them where
    the likelihood is flat. Returns a dict of the four parameters."""
    x = np.asarray(x, float)
    t_x = np.asarray(t_x, float)
    T = np.asarray(T, float)

    def objective(log_params):
        return _neg_log_likelihood(np.exp(log_params), x, t_x, T) + reg * float(
            np.sum(log_params ** 2)
        )

    best, _ = minimize_nelder_mead(objective, np.log(np.asarray(init, float)))
    r, alpha, a, b = np.exp(best)
    return {"r": float(r), "alpha": float(alpha), "a": float(a), "b": float(b)}


def prob_alive(params, x, t_x, T):
    r, alpha, a, b = params["r"], params["alpha"], params["a"], params["b"]
    x = np.asarray(x, float)
    t_x = np.asarray(t_x, float)
    T = np.asarray(T, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        extra = np.where(
            x > 0, (a / (b + x - 1.0)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0
        )
    return 1.0 / (1.0 + extra)


def expected_future(params, x, t_x, T, tau):
    """Conditional expected number of events in a future horizon of length tau,
    E[Y(tau) | x, t_x, T], the value estimate for each fan."""
    r, alpha, a, b = params["r"], params["alpha"], params["a"], params["b"]
    a = max(a, 1.0 + 1e-6)  # (a - 1) guard; a > 1 required for a finite mean
    x = np.asarray(x, float)
    t_x = np.asarray(t_x, float)
    T = np.asarray(T, float)

    z = tau / (alpha + T + tau)
    hg = hyp2f1(r + x, b + x, a + b + x - 1.0, z)
    leading = (a + b + x - 1.0) / (a - 1.0)
    bracket = 1.0 - ((alpha + T) / (alpha + T + tau)) ** (r + x) * hg
    return leading * bracket * prob_alive(params, x, t_x, T)
