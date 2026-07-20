"""Confound re-analysis prompted by expert review.

Three checks on the real pilot data:
  1. Two-delta decomposition of the H6 cold-start lift. The original H6 baseline
     is a global constant. Here we compare the inductive model against three
     progressively stronger baselines (global mean, community mean, an
     activity+degree model). The network-attributable lift is the advantage over
     the activity+degree baseline, not over the global mean.
  2. Within-community breadth test (H3). Community-demean breadth and activity so
     the coefficient reflects variation within a community, not sorting across
     communities.
  3. Many-draw degree-preserving rewiring null (H5). The original test used one
     rewiring. Here we build a null distribution and report where the real graph
     falls.

Everything runs on the already-collected real sample. Results are printed exactly
as computed.
"""
import sys
import json
import pathlib
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fancoldstart.graph.build import build_fan_graph          # noqa: E402
from fancoldstart.graph.features import compute_features      # noqa: E402
from fancoldstart.eval.splits import make_splits              # noqa: E402
from fancoldstart.models.inductive import InductiveHead       # noqa: E402
from fancoldstart import glm                                  # noqa: E402
from fancoldstart.stats import spearman                       # noqa: E402
from fancoldstart.pipeline import _degree_preserving_rewire   # noqa: E402

CUT, TAU, COLD, SEED = 24.0, 15.0, 1, 7
N_REWIRE = 60


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p, float) - y)))


def main():
    events = [json.loads(l) for l in open(ROOT / "data" / "wc2026_reddit_events.jsonl", encoding="utf-8")]
    splits = make_splits(events, CUT, TAU, COLD)
    G = build_fan_graph(events, CUT, seed=SEED)
    fans_g, Phi, names = compute_features(G)
    feat = {f: Phi[i] for i, f in enumerate(fans_g)}

    home = {f: G.nodes[f].get("home") for f in splits.fan_ids}
    y_by = {f: splits.y_future[i] for i, f in enumerate(splits.fan_ids)}
    x_by = {f: splits.x[i] for i, f in enumerate(splits.fan_ids)}
    deg = {f: float(G.degree(f)) for f in splits.fan_ids}

    rng = np.random.default_rng(SEED)
    ids = list(splits.fan_ids)
    perm = rng.permutation(len(ids))
    ntr = int(0.7 * len(ids))
    train = [ids[i] for i in perm[:ntr]]
    test = [ids[i] for i in perm[ntr:]]
    cold_test = [f for f in test if x_by[f] <= COLD]
    realized = np.array([y_by[f] for f in cold_test], float)
    ytr = np.array([y_by[f] for f in train], float)

    print(f"fans={len(ids)} cold-start={int(splits.cold_mask.sum())} | cold_test={len(cold_test)}")

    # ---- 1. two-delta decomposition ----
    b_global = np.full(len(cold_test), ytr.mean())

    csum = defaultdict(list)
    for f in train:
        csum[home[f]].append(y_by[f])
    cmean = {c: float(np.mean(v)) for c, v in csum.items()}
    b_comm = np.array([cmean.get(home[f], ytr.mean()) for f in cold_test])

    def ad_design(fs):
        Z = np.column_stack([[feat[f][3] for f in fs], [deg[f] for f in fs]])
        return Z.astype(float)
    Ztr, Zte = ad_design(train), ad_design(cold_test)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-9
    Xtr = np.column_stack([np.ones(len(Ztr)), (Ztr - mu) / sd])
    Xte = np.column_stack([np.ones(len(Zte)), (Zte - mu) / sd])
    R = np.eye(Xtr.shape[1]); R[0, 0] = 0
    w = np.linalg.solve(Xtr.T @ Xtr + R, Xtr.T @ ytr)
    b_ad = np.maximum(Xte @ w, 0)

    head = InductiveHead(1.0)
    head.fit(G, train, feat, ytr)
    full = head.predict(G, cold_test, feat)

    print("\n[1] H6 baselines on held-out cold-start fans (lower MAE is better):")
    for nm, p in [("global mean", b_global), ("community mean", b_comm),
                  ("activity+degree", b_ad), ("inductive (full graph)", full)]:
        print(f"    {nm:24s} MAE={mae(p, realized):.4f}  spearman={spearman(p, realized):+.3f}")
    m_full = mae(full, realized)
    for nm, p in [("global", b_global), ("community", b_comm), ("activity+degree", b_ad)]:
        red = (mae(p, realized) - m_full) / mae(p, realized) if mae(p, realized) > 0 else 0.0
        tag = "  <-- network-attributable" if nm == "activity+degree" else ""
        print(f"    lift full vs {nm:16s} = {red*100:+.1f}%{tag}")

    # ---- 2. within-community breadth (H3) ----
    breadth = np.array([feat[f][0] for f in ids])
    activity = np.array([feat[f][3] for f in ids])
    persist = splits.persist
    # community-demean
    grp = defaultdict(list)
    for i, f in enumerate(ids):
        grp[home[f]].append(i)
    br_w = breadth.copy().astype(float)
    ac_w = activity.copy().astype(float)
    for c, idx in grp.items():
        idx = np.array(idx)
        br_w[idx] -= breadth[idx].mean()
        ac_w[idx] -= activity[idx].mean()

    def std(v):
        return (v - v.mean()) / (v.std() + 1e-9)
    fit_naive = glm.logistic_regression(np.column_stack([std(glm.residualize(breadth, activity)), std(activity)]), persist)
    fit_within = glm.logistic_regression(np.column_stack([std(glm.residualize(br_w, ac_w)), std(ac_w)]), persist)
    print("\n[2] H3 breadth -> persistence:")
    print(f"    pooled (original)          coef={fit_naive['beta'][1]:+.3f}  p={fit_naive['p'][1]:.2g}")
    print(f"    within-community (demeaned) coef={fit_within['beta'][1]:+.3f}  p={fit_within['p'][1]:.2g}")

    # ---- 3. many-draw rewiring null (H5) ----
    print(f"\n[3] H5 degree-preserving rewiring null ({N_REWIRE} draws): real-graph advantage over rewired")
    advantages = []
    for d in range(N_REWIRE):
        Grw = _degree_preserving_rewire(G, SEED + 1 + d)
        fr, Pr, _ = compute_features(Grw)
        featr = {f: Pr[i] for i, f in enumerate(fr)}
        h = InductiveHead(1.0)
        h.fit(Grw, train, featr, ytr)
        prw = h.predict(Grw, cold_test, featr)
        m_rw = mae(prw, realized)
        advantages.append((m_rw - m_full) / m_rw if m_rw > 0 else 0.0)
        del Grw, featr, h, prw
    adv = np.array(advantages)
    print(f"    real-vs-rewired MAE reduction: mean={adv.mean()*100:+.2f}%  "
          f"sd={adv.std()*100:.2f}  p5={np.percentile(adv,5)*100:+.2f}%  p95={np.percentile(adv,95)*100:+.2f}%")
    print(f"    draws where real beats rewired: {int((adv>0).sum())}/{N_REWIRE} "
          f"| beats by >=5% floor: {int((adv>=0.05).sum())}/{N_REWIRE}")

    out = {
        "h6_mae": {"global": mae(b_global, realized), "community": mae(b_comm, realized),
                   "activity_degree": mae(b_ad, realized), "inductive": m_full},
        "h6_lift_vs_global": (mae(b_global, realized) - m_full) / mae(b_global, realized),
        "h6_lift_vs_community": (mae(b_comm, realized) - m_full) / mae(b_comm, realized),
        "h6_lift_vs_activity_degree": (mae(b_ad, realized) - m_full) / mae(b_ad, realized),
        "h3_pooled_coef": float(fit_naive['beta'][1]), "h3_pooled_p": float(fit_naive['p'][1]),
        "h3_within_coef": float(fit_within['beta'][1]), "h3_within_p": float(fit_within['p'][1]),
        "h5_rewire_mean": float(adv.mean()), "h5_rewire_p5": float(np.percentile(adv, 5)),
        "h5_rewire_p95": float(np.percentile(adv, 95)), "h5_draws": N_REWIRE,
    }
    (ROOT / "data" / "reanalysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nsaved data/reanalysis.json")


if __name__ == "__main__":
    main()
