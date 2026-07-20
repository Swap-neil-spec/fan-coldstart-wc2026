"""Validate the GLMs recover known effects on data simulated from each model."""
import numpy as np

from fancoldstart import glm


def test_logistic_recovers_positive_effect():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 1.2 * x)))
    y = (rng.uniform(size=n) < p).astype(float)
    fit = glm.logistic_regression(x, y)
    assert fit["beta"][1] > 0
    assert fit["p"][1] < 0.05


def test_ols_recovers_slope():
    rng = np.random.default_rng(2)
    n = 3000
    x = rng.normal(size=n)
    y = 3.0 * x + rng.normal(scale=1.0, size=n)
    fit = glm.ols(x, y)
    assert abs(fit["beta"][1] - 3.0) < 0.2
    assert fit["p"][1] < 0.05


def test_negbin_recovers_positive_effect():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.normal(size=n)
    mu = np.exp(0.5 + 0.6 * x)
    # NB2 draws via gamma-poisson mixture
    alpha = 0.5
    shape = 1.0 / alpha
    lam = rng.gamma(shape=shape, scale=mu / shape)
    y = rng.poisson(lam).astype(float)
    fit = glm.negbin_nb2(x, y)
    assert fit["beta"][1] > 0
    assert fit["p"][1] < 0.05


def test_residualize_removes_control():
    rng = np.random.default_rng(4)
    n = 1000
    c = rng.normal(size=n)
    f = 2.0 * c + rng.normal(scale=0.5, size=n)
    r = glm.residualize(f, c)
    # residual should be nearly uncorrelated with the control
    corr = np.corrcoef(r, c)[0, 1]
    assert abs(corr) < 0.05
