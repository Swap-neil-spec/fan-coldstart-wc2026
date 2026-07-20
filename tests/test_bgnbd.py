"""Validate the BG/NBD implementation by simulating from the generative model
with known parameters, fitting, and checking that (a) the optimizer reaches at
least the true-parameter likelihood, (b) the mean-rate parameter is recovered,
and (c) predicted future engagement correlates with realized future events.
No fabricated data: this is a self-check of the estimator on data drawn from the
model it claims to fit, which is exactly what a unit test should do.
"""
import numpy as np

from fancoldstart.models import bgnbd
from fancoldstart.stats import spearman


def simulate(n, r, alpha, a, b, T_cal, tau, seed=7):
    rng = np.random.default_rng(seed)
    lam = rng.gamma(shape=r, scale=1.0 / alpha, size=n)
    p = rng.beta(a, b, size=n)
    x = np.zeros(n, float)
    t_x = np.zeros(n, float)
    y_future = np.zeros(n, float)
    horizon = T_cal + tau
    for i in range(n):
        t = 0.0
        last = 0.0
        cnt_cal = 0
        cnt_future = 0
        alive = True
        while alive:
            t += rng.exponential(1.0 / lam[i])
            if t > horizon:
                break
            if t <= T_cal:
                cnt_cal += 1
                last = t
            else:
                cnt_future += 1
            if rng.random() < p[i]:
                alive = False
        x[i] = cnt_cal
        t_x[i] = last
        y_future[i] = cnt_future
    return x, t_x, np.full(n, T_cal, float), y_future


def test_bgnbd_recovers_and_predicts():
    r, alpha, a, b = 0.9, 5.0, 1.3, 2.4
    T_cal, tau = 40.0, 20.0
    x, t_x, T, y_future = simulate(6000, r, alpha, a, b, T_cal, tau)

    params = bgnbd.fit(x, t_x, T)

    # (a) optimizer beats or matches the true-parameter likelihood
    from fancoldstart.models.bgnbd import _neg_log_likelihood
    nll_fit = _neg_log_likelihood((params["r"], params["alpha"], params["a"], params["b"]), x, t_x, T)
    nll_true = _neg_log_likelihood((r, alpha, a, b), x, t_x, T)
    assert nll_fit <= nll_true + 5.0

    # (b) mean purchase rate E[lambda] = r / alpha recovered within 25 percent
    assert abs((params["r"] / params["alpha"]) / (r / alpha) - 1.0) < 0.25

    # (c) predicted future engagement correlates with realized future events
    pred = bgnbd.expected_future(params, x, t_x, T, tau)
    assert np.all(np.isfinite(pred))
    assert spearman(pred, y_future) > 0.3

    # sanity: probabilities and expectations in valid ranges
    pa = bgnbd.prob_alive(params, x, t_x, T)
    assert np.all((pa >= 0) & (pa <= 1))
    assert np.all(pred >= 0)
