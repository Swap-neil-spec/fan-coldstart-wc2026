"""End-to-end pipeline: events -> graph -> features -> splits -> models ->
pre-registered hypothesis tests -> report. This is the object the paper refers
to as "the released pipeline." Running it on real collected data produces the
numbers the paper deliberately does not assert in advance.
"""
import numpy as np
import networkx as nx

from .graph.build import build_fan_graph
from .graph.features import compute_features
from .eval.splits import make_splits
from .eval import hypotheses as H
from .eval.report import to_markdown
from .models import bgnbd, graph_smoothed
from .models.inductive import InductiveHead


def _degree_preserving_rewire(G, seed):
    Gr = nx.Graph()
    Gr.add_nodes_from(G.nodes(data=True))
    Gr.add_edges_from(G.edges())
    m = Gr.number_of_edges()
    if m >= 2:
        try:
            nx.double_edge_swap(Gr, nswap=m, max_tries=m * 20, seed=seed)
        except (nx.NetworkXError, nx.NetworkXAlgorithmError):
            pass
    return Gr


def run(events, t_cutoff, tau, provenance="unknown", gamma=0.3,
        cold_threshold=1, seed=13):
    splits = make_splits(events, t_cutoff, tau, cold_threshold)
    G = build_fan_graph(events, t_cutoff, seed=seed)

    fans_g, Phi, names = compute_features(G)
    feat_by_fan = {f: Phi[i] for i, f in enumerate(fans_g)}
    ncol = len(names)
    F = np.array([feat_by_fan.get(f, np.zeros(ncol)) for f in splits.fan_ids])
    breadth, bridging, density, activity = F[:, 0], F[:, 1], F[:, 2], F[:, 3]

    params = bgnbd.fit(splits.x, splits.t_x, splits.T)

    # Both predictors share the same functional form, rate * tau * P(alive), and
    # differ ONLY in the rate: the baseline uses the fan's own gamma-posterior
    # mean rate, the graph model uses the neighbor-smoothed rate. With gamma = 0
    # the two are identical, so H1 isolates the contribution of the graph term.
    palive = bgnbd.prob_alive(params, splits.x, splits.t_x, splits.T)
    base_rate = (params["r"] + splits.x) / (params["alpha"] + splits.T)
    base_pred = base_rate * tau * palive
    graph_pred = graph_smoothed.fit_predict(
        G, splits.fan_ids, splits.x, splits.t_x, splits.T, params, tau, gamma=gamma)

    # Mean-calibrate the smoothed predictor to the baseline scale. Smoothing the
    # log-rate toward more-active neighbors adds a small upward bias; a single
    # global rescaling removes that bias so the H1/H5 comparison reflects whether
    # the graph term improves the RANKING of cold-start fans, not a scale shift.
    def _calibrate(pred, ref):
        m = float(np.mean(pred))
        return pred * (float(np.mean(ref)) / m) if m > 0 else pred

    graph_pred = _calibrate(graph_pred, base_pred)

    cm = splits.cold_mask
    realized_cold = splits.y_future[cm]

    results = []
    results.append(H.h1_cold_start_lift(realized_cold, base_pred[cm], graph_pred[cm]))

    err_graph = np.abs(graph_pred - splits.y_future)
    err_base = np.abs(base_pred - splits.y_future)
    freq_stratum = np.minimum(splits.x, 5)
    results.append(H.h2_effect_concentration(err_graph, err_base, freq_stratum))

    results.append(H.h3_breadth_persistence(splits.persist, breadth, activity))
    results.append(H.h4_bridging_engagement(splits.y_future, bridging, activity, density))

    G_rw = _degree_preserving_rewire(G, seed)
    graph_pred_rw = graph_smoothed.fit_predict(
        G_rw, splits.fan_ids, splits.x, splits.t_x, splits.T, params, tau, gamma=gamma)
    graph_pred_rw = _calibrate(graph_pred_rw, base_pred)
    results.append(H.h5_beyond_activity(realized_cold, base_pred[cm], graph_pred_rw[cm]))

    # H6 inductive transfer: train on a fold, predict cold-start fans held out
    y_by = {f: splits.y_future[i] for i, f in enumerate(splits.fan_ids)}
    x_by = {f: splits.x[i] for i, f in enumerate(splits.fan_ids)}
    rng = np.random.default_rng(seed)
    ids = list(splits.fan_ids)
    perm = rng.permutation(len(ids))
    ntr = int(0.7 * len(ids))
    train_ids = [ids[i] for i in perm[:ntr]]
    test_ids = [ids[i] for i in perm[ntr:]]
    cold_test = [f for f in test_ids if x_by[f] <= cold_threshold]
    if len(cold_test) >= 10:
        head = InductiveHead(ridge=1.0)
        head.fit(G, train_ids, feat_by_fan, [y_by[f] for f in train_ids])
        ind_pred = head.predict(G, cold_test, feat_by_fan)
        base_test = head.baseline(cold_test)
        realized_test = np.array([y_by[f] for f in cold_test], float)
        results.append(H.h6_inductive_transfer(realized_test, ind_pred, base_test))
    else:
        results.append(H._result("H6", True, 0.0, 1.0, 0.0, "holdout MAE reduction",
                                  False, False, "too few cold-start test fans to evaluate"))

    results.append(H.h7_cross_platform(None, None))

    results = H.apply_multiplicity(results)
    report_md = to_markdown(results, params, provenance, len(splits.fan_ids),
                            int(cm.sum()), tau)
    return {"params": params, "results": results, "report_md": report_md,
            "n_fans": len(splits.fan_ids), "n_cold": int(cm.sum())}
