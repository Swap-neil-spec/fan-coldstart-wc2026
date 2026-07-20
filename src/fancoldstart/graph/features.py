"""Public-data position features for each fan (paper Section 4.4).

All features use calibration-window data only and are deliberately conservative
operationalizations of the network-diversity ideas the paper motivates:

  breadth   Shannon entropy of the fan's own activity across communities
  bridging  Simpson diversity of the communities of the fan's graph neighbors
            (the probability that two random neighbors sit in different
            communities), a weak-tie bridging measure
  density   local clustering coefficient of the fan in the graph
  activity  calibration-window event count (the control variable)

Features (breadth, bridging, density) and the value target are disjoint
constructs: none of these is used as an outcome anywhere.
"""
import math

import networkx as nx
import numpy as np


def _entropy(counts):
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def community_breadth(G, fan):
    hist = G.nodes[fan].get("communities", {})
    return _entropy(hist.values())


def bridging_score(G, fan):
    """Simpson diversity of neighbor home-communities: 1 - sum_c (n_c/n)^2."""
    neigh = list(G.neighbors(fan))
    if not neigh:
        return 0.0
    counts = {}
    for v in neigh:
        home = G.nodes[v].get("home")
        counts[home] = counts.get(home, 0) + 1
    n = len(neigh)
    simpson = sum((c / n) ** 2 for c in counts.values())
    return 1.0 - simpson


def neighborhood_density(G, fan):
    return float(nx.clustering(G, fan))


def compute_features(G):
    """Return (fan_ids, feature_matrix, feature_names).
    Rows follow fan_ids order; columns are [breadth, bridging, density, activity].
    """
    fans = list(G.nodes)
    clustering = nx.clustering(G)  # computed once for all nodes
    rows = []
    for fan in fans:
        rows.append([
            community_breadth(G, fan),
            bridging_score(G, fan),
            float(clustering.get(fan, 0.0)),
            G.nodes[fan].get("activity", 0.0),
        ])
    return fans, np.asarray(rows, float), ["breadth", "bridging", "density", "activity"]
