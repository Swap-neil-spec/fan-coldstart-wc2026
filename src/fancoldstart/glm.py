"""Generalized linear models used by the pre-registered hypothesis tests:
logistic regression (persistence, H3), negative-binomial NB2 regression
(future-engagement counts, H4), and OLS with heteroskedasticity-robust standard
errors (effect-concentration interaction, H2). Each returns coefficients,
standard errors, and two-sided Wald p-values. Implemented on numpy + stdlib.
"""
import math
import numpy as np

from .special import gammaln
from .stats import chi2_sf_1df
from .optimize import minimize_nelder_mead


def _design(X):
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    return np.column_stack([np.ones(len(X)), X])


def _wald(beta, cov):
    se = np.sqrt(np.diag(cov))
    wald = np.where(se > 0, (beta / se) ** 2, 0.0)
    p = np.array([chi2_sf_1df(w) for w in wald])
    return se, p


def logistic_regression(X, y, max_iter=200, tol=1e-9, ridge=1e-8):
    Xd = _design(X)
    y = np.asarray(y, float)
    n, k = Xd.shape
    beta = np.zeros(k)
    for _ in range(max_iter):
        eta = np.clip(Xd @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = mu * (1 - mu) + 1e-12
        z = eta + (y - mu) / w
        H = (Xd.T * w) @ Xd + ridge * np.eye(k)
        beta_new = np.linalg.solve(H, (Xd.T * w) @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = np.clip(Xd @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = mu * (1 - mu) + 1e-12
    cov = np.linalg.inv((Xd.T * w) @ Xd + ridge * np.eye(k))
    se, p = _wald(beta, cov)
    return {"beta": beta, "se": se, "p": p}


def ols(X, y, robust=True):
    Xd = _design(X)
    y = np.asarray(y, float)
    n, k = Xd.shape
    XtX_inv = np.linalg.inv(Xd.T @ Xd + 1e-10 * np.eye(k))
    beta = XtX_inv @ (Xd.T @ y)
    resid = y - Xd @ beta
    if robust:  # HC0 sandwich
        meat = (Xd * resid[:, None]).T @ (Xd * resid[:, None])
        cov = XtX_inv @ meat @ XtX_inv
    else:
        sigma2 = float(resid @ resid) / max(n - k, 1)
        cov = sigma2 * XtX_inv
    se, p = _wald(beta, cov)
    return {"beta": beta, "se": se, "p": p}


def negbin_nb2(X, y, max_iter=200, tol=1e-9):
    """Negative-binomial (NB2) regression with a log link. Dispersion alpha is
    profiled by maximum likelihood; beta is fit by IRLS at each alpha."""
    Xd = _design(X)
    y = np.asarray(y, float)
    n, k = Xd.shape

    def fit_beta(alpha):
        beta = np.zeros(k)
        beta[0] = math.log(max(y.mean(), 1e-3))
        for _ in range(max_iter):
            eta = np.clip(Xd @ beta, -30, 30)
            mu = np.exp(eta)
            w = mu / (1.0 + alpha * mu)
            z = eta + (y - mu) / mu
            H = (Xd.T * w) @ Xd + 1e-8 * np.eye(k)
            beta_new = np.linalg.solve(H, (Xd.T * w) @ z)
            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new
        return beta, np.exp(np.clip(Xd @ beta, -30, 30))

    def nll(log_alpha):
        alpha = math.exp(log_alpha[0])
        beta, mu = fit_beta(alpha)
        r = 1.0 / alpha
        ll = (
            gammaln(y + r) - gammaln(r) - gammaln(y + 1)
            + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu))
        )
        val = -float(np.sum(ll))
        return val if np.isfinite(val) else 1e12

    best, _ = minimize_nelder_mead(nll, np.array([0.0]))
    alpha = math.exp(best[0])
    beta, mu = fit_beta(alpha)
    w = mu / (1.0 + alpha * mu)
    cov = np.linalg.inv((Xd.T * w) @ Xd + 1e-8 * np.eye(k))
    se, p = _wald(beta, cov)
    return {"beta": beta, "se": se, "p": p, "alpha": alpha}


def residualize(feature, controls):
    """Return the part of `feature` orthogonal to `controls` (with intercept),
    used to control for activity in H3 and H4."""
    Xd = _design(controls)
    coef = np.linalg.lstsq(Xd, np.asarray(feature, float), rcond=None)[0]
    return np.asarray(feature, float) - Xd @ coef
