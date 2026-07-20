"""Graph-smoothed latent-attrition estimator (paper Section 4.5).

A hierarchical model with a graph-smoothness prior over a BG/NBD-style backbone.
Each fan carries a latent log purchase-rate. A thin fan's own data barely
constrains that rate, so the estimate is pulled toward a locally structured
prior read off the fan graph rather than toward a single global prior. This is
graph-Laplacian regularization of the per-fan transformed rate:

  minimize   sum_i c_i (theta_i - theta_i_data)^2  +  gamma * sum_{(i,j) in E} w_ij (theta_i - theta_j)^2

with optimality condition  (C + gamma L) theta = C theta_data, where L is the
weighted graph Laplacian, C = diag(c_i), and theta_i_data is the gamma-posterior
mean log-rate log((r + x_i) / (alpha + T_i)). The confidence c_i grows with the
fan's own event count, so data-rich fans keep their own estimate and cold-start
fans borrow from neighbors. This is a predictive prior; no causal claim is made.

The linear system is solved by Jacobi iteration over a sparse edge list rather
than forming the dense Laplacian, so the estimator scales to large graphs. The
system is strictly diagonally dominant because c_i > 0, so Jacobi converges. The
penalty is applied on the transformed (log-rate) parameter, as pre-registered.
"""
import numpy as np

from . import bgnbd


def fit_predict(G, fan_ids, x, t_x, T, params, tau, gamma=0.5, c0=1.0,
                max_iter=1000, tol=1e-7):
    """Degree-normalized (random-walk) graph smoothing of the log-rate.

    The neighbor term is the degree-normalized AVERAGE of neighbor log-rates,
    not their sum, so a high-degree cold-start fan is not steamrolled by the
    number of neighbors. Each fan's own gamma-posterior log-rate carries weight
    c_i = x_i + c0 and the neighbor average carries weight gamma, so a cold-start
    fan (small x) is nudged toward its neighborhood while retaining its own
    evidence, and a data-rich fan mostly keeps its own estimate. This is the
    random-walk-normalized Laplacian prior; it cannot inflate a rate beyond the
    range of the data.
    """
    x = np.asarray(x, float)
    t_x = np.asarray(t_x, float)
    T = np.asarray(T, float)
    n = len(fan_ids)
    index = {f: i for i, f in enumerate(fan_ids)}

    r, alpha = params["r"], params["alpha"]
    theta_data = np.log((r + x) / (alpha + T))  # gamma-posterior mean log-rate
    c = x + c0                                   # own-data confidence

    rows, cols, wts = [], [], []
    for u, v, d in G.edges(data=True):
        if u in index and v in index:
            iu, iv = index[u], index[v]
            w = float(d.get("weight", 1.0))
            rows.append(iu); cols.append(iv); wts.append(w)
            rows.append(iv); cols.append(iu); wts.append(w)
    if rows:
        rows = np.asarray(rows); cols = np.asarray(cols); wts = np.asarray(wts, float)
        deg = np.bincount(rows, weights=wts, minlength=n)
    else:
        deg = np.zeros(n)

    has_nbr = deg > 0
    eff_gamma = np.where(has_nbr, gamma, 0.0)
    denom = c + eff_gamma
    theta = theta_data.copy()
    for _ in range(max_iter):
        if len(rows):
            nb_sum = np.bincount(rows, weights=wts * theta[cols], minlength=n)
            nb_avg = np.where(has_nbr, nb_sum / np.where(has_nbr, deg, 1.0), 0.0)
        else:
            nb_avg = np.zeros(n)
        new_theta = (c * theta_data + eff_gamma * nb_avg) / denom
        if np.max(np.abs(new_theta - theta)) < tol:
            theta = new_theta
            break
        theta = new_theta

    rate = np.exp(theta)
    palive = bgnbd.prob_alive(params, x, t_x, T)
    return rate * tau * palive
