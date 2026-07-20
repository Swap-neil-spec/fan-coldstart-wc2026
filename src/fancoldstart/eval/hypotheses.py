"""The pre-registered hypothesis tests H1 to H7 (paper Section 6).

Each test returns a structured result: the named statistic, its two-sided
p-value, the effect and whether it meets the pre-registered smallest effect size
of interest, and a support decision. The primary family is H1, H3, H4, H5, H6;
H2 and H7 are secondary. p-values across the primary family are corrected by
Benjamini-Hochberg at a false-discovery rate of 0.05. Every number here is a
pipeline output on whatever data is supplied, not a claim asserted in the paper.
"""
import numpy as np

from .. import glm
from ..stats import wilcoxon_signed_rank, spearman, benjamini_hochberg

ALPHA = 0.05
SESOI_MAE = 0.05      # H1, H6: at least a 5 percent reduction in holdout MAE
SESOI_COEF = 0.10     # H3, H4: standardized coefficient at least 0.1 in absolute value


def _std(v):
    v = np.asarray(v, float)
    s = v.std()
    return (v - v.mean()) / (s if s > 0 else 1.0)


def _mae(pred, realized):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(realized, float))))


def _result(name, primary, statistic, p, effect, effect_name, sesoi_met, supported, note=""):
    return {
        "name": name, "primary": primary, "statistic": float(statistic), "p": float(p),
        "effect": float(effect), "effect_name": effect_name, "sesoi_met": bool(sesoi_met),
        "significant": bool(p < ALPHA), "supported": bool(supported), "note": note,
    }


def h1_cold_start_lift(realized, base_pred, graph_pred):
    ae_base = np.abs(base_pred - realized)
    ae_graph = np.abs(graph_pred - realized)
    mae_base, mae_graph = _mae(base_pred, realized), _mae(graph_pred, realized)
    reduction = (mae_base - mae_graph) / mae_base if mae_base > 0 else 0.0
    p, _ = wilcoxon_signed_rank(ae_base - ae_graph)
    sesoi = reduction >= SESOI_MAE
    return _result("H1", True, reduction, p, reduction, "holdout MAE reduction",
                   sesoi, sesoi and p < ALPHA)


def h2_effect_concentration(errors_graph, errors_base, freq_stratum):
    # pooled regression of per-fan error on a graph indicator, the frequency
    # stratum, and their interaction; the interaction is the test.
    err = np.concatenate([errors_graph, errors_base])
    graph_ind = np.concatenate([np.ones_like(errors_graph), np.zeros_like(errors_base)])
    freq = np.concatenate([freq_stratum, freq_stratum]).astype(float)
    X = np.column_stack([graph_ind, freq, graph_ind * freq])
    fit = glm.ols(err, np.zeros(0)) if False else glm.ols(X, err)
    p = fit["p"][3]  # interaction coefficient (intercept, graph, freq, graph*freq)
    return _result("H2", False, fit["beta"][3], p, fit["beta"][3], "graph x frequency interaction",
                   True, False, "secondary; concentration of benefit in the cold-start stratum")


def h3_breadth_persistence(persist, breadth, activity):
    resid = glm.residualize(breadth, activity)
    X = np.column_stack([_std(resid), _std(activity)])
    fit = glm.logistic_regression(X, persist)
    coef, p = fit["beta"][1], fit["p"][1]
    sesoi = abs(coef) >= SESOI_COEF
    return _result("H3", True, coef, p, coef, "standardized logit coefficient",
                   sesoi, coef > 0 and sesoi and p < ALPHA)


def h4_bridging_engagement(y_future, bridging, activity, density):
    resid = glm.residualize(bridging, np.column_stack([activity, density]))
    X = np.column_stack([_std(resid), _std(activity), _std(density)])
    fit = glm.negbin_nb2(X, y_future)
    coef, p = fit["beta"][1], fit["p"][1]
    sesoi = abs(coef) >= SESOI_COEF
    return _result("H4", True, coef, p, coef, "standardized NB2 coefficient",
                   sesoi, coef > 0 and sesoi and p < ALPHA)


def h5_beyond_activity(realized, base_pred, graph_pred_rewired):
    # the load-bearing falsification: the cold-start lift must survive a
    # degree-preserving rewiring of the graph. If a fake (rewired) graph gives
    # the same lift, the features were degree or activity proxies.
    ae_base = np.abs(base_pred - realized)
    ae_rewired = np.abs(graph_pred_rewired - realized)
    mae_base = _mae(base_pred, realized)
    reduction_rewired = (mae_base - _mae(graph_pred_rewired, realized)) / mae_base if mae_base > 0 else 0.0
    p, _ = wilcoxon_signed_rank(ae_base - ae_rewired)
    # support means the REAL lift does NOT reproduce under rewiring, i.e. the
    # rewired lift is not a meaningful improvement.
    survives = reduction_rewired < SESOI_MAE
    return _result("H5", True, reduction_rewired, p, reduction_rewired,
                   "rewired-graph MAE reduction (should be near zero)", True, survives,
                   "supported when the real lift does not reproduce on a rewired graph")


def h5_structure_not_degree(realized_test, inductive_real, inductive_rewired):
    """Load-bearing falsification of the H6 result. The same inductive model is
    trained and evaluated on the real fan graph and on a degree-preserving
    rewired graph (features recomputed on each). If the real graph predicts
    cold-start fans better than the degree-matched random graph, the gain comes
    from genuine structure, not from degree or activity. Supported when the real
    graph's advantage over the rewired graph meets the effect-size floor and is
    significant."""
    ae_real = np.abs(inductive_real - realized_test)
    ae_rw = np.abs(inductive_rewired - realized_test)
    mae_real, mae_rw = _mae(inductive_real, realized_test), _mae(inductive_rewired, realized_test)
    reduction = (mae_rw - mae_real) / mae_rw if mae_rw > 0 else 0.0
    p, _ = wilcoxon_signed_rank(ae_rw - ae_real)
    sesoi = reduction >= SESOI_MAE
    return _result("H5", True, reduction, p, reduction, "real vs degree-matched MAE reduction",
                   sesoi, sesoi and p < ALPHA,
                   "supported when the real graph beats a degree-preserving rewired graph on cold-start fans")


def h6_inductive_transfer(realized_test, inductive_pred, baseline_pred):
    ae_ind = np.abs(inductive_pred - realized_test)
    ae_base = np.abs(baseline_pred - realized_test)
    mae_base = _mae(baseline_pred, realized_test)
    reduction = (mae_base - _mae(inductive_pred, realized_test)) / mae_base if mae_base > 0 else 0.0
    p, _ = wilcoxon_signed_rank(ae_base - ae_ind)
    sesoi = reduction >= SESOI_MAE
    return _result("H6", True, reduction, p, reduction, "holdout MAE reduction vs population average",
                   sesoi, sesoi and p < ALPHA)


def h7_cross_platform(corpus2_h3=None, corpus2_h4=None):
    if corpus2_h3 is None or corpus2_h4 is None:
        return _result("H7", False, 0.0, 1.0, 0.0, "cross-platform replication", False, False,
                       "secondary; not evaluated in this run (no second corpus supplied)")
    same_sign = (corpus2_h3["effect"] > 0) and (corpus2_h4["effect"] > 0)
    sig = (corpus2_h3["p"] < ALPHA) and (corpus2_h4["p"] < ALPHA)
    p = max(corpus2_h3["p"], corpus2_h4["p"])
    return _result("H7", False, 1.0 if same_sign else 0.0, p, 1.0 if same_sign else 0.0,
                   "same-sign same-significance replication", same_sign, same_sign and sig)


def apply_multiplicity(results, q=0.05):
    primary = [r for r in results if r["primary"]]
    if primary:
        reject, padj = benjamini_hochberg([r["p"] for r in primary], q=q)
        for r, rej, pa in zip(primary, reject, padj):
            r["bh_reject"] = bool(rej)
            r["p_bh"] = float(pa)
    for r in results:
        if "bh_reject" not in r:
            r["bh_reject"] = None
            r["p_bh"] = None
    return results
