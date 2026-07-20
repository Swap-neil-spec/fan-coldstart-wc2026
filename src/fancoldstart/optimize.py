"""A small, dependency-free Nelder-Mead minimizer.

Used for the four-parameter BG/NBD maximum-likelihood fit so the core does not
require scipy.optimize. If scipy is installed, models may use it instead, but
this keeps the minimal stack self-contained and deterministic.
"""
import numpy as np


def minimize_nelder_mead(f, x0, max_iter=8000, tol=1e-10, step=0.5):
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    sim = np.empty((n + 1, n), dtype=float)
    sim[0] = x0
    for i in range(n):
        y = x0.copy()
        y[i] = y[i] + step if y[i] == 0 else y[i] * 1.05
        sim[i + 1] = y
    fv = np.array([f(p) for p in sim], dtype=float)

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = np.argsort(fv)
        sim, fv = sim[order], fv[order]
        if abs(fv[-1] - fv[0]) <= tol * (abs(fv[0]) + tol):
            break
        centroid = sim[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - sim[-1])
        fr = f(xr)
        if fr < fv[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            if fe < fr:
                sim[-1], fv[-1] = xe, fe
            else:
                sim[-1], fv[-1] = xr, fr
        elif fr < fv[-2]:
            sim[-1], fv[-1] = xr, fr
        else:
            xc = centroid + rho * (sim[-1] - centroid)
            fc = f(xc)
            if fc < fv[-1]:
                sim[-1], fv[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    sim[i] = sim[0] + sigma * (sim[i] - sim[0])
                    fv[i] = f(sim[i])
    order = np.argsort(fv)
    return sim[order][0], float(fv[order][0])
