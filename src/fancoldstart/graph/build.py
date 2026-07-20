"""Build the fan social graph from an engagement event table.

An event is a dict with keys:
  fan        str   the fan identifier
  community  str   the community the activity occurred in (for example a subreddit)
  t          float event time in days from an origin
  reply_to   str or None   the fan replied to, if this event is a reply

The formal graph combines two edge types, following the paper (Section 4.3):
  - co-participation edges: fans who post in the same community are linked
    (the bipartite fan-community graph projected onto the fan side)
  - reply edges: fan-to-fan replies

Only calibration-window events (t <= t_cutoff) are used, so no holdout
information leaks into the graph. Reddit supplies the formal graph; Bluesky is a
complementary source in the full pipeline.
"""
from collections import defaultdict
import itertools

import numpy as np
import networkx as nx


def calibration_events(events, t_cutoff):
    return [e for e in events if e["t"] <= t_cutoff]


def build_fan_graph(events, t_cutoff, max_community_size=50000,
                    max_pairs_per_community=6000, seed=0):
    """Return an undirected weighted fan graph from calibration events.

    Co-participation edges connect fans sharing a community. A full clique per
    community is O(m^2) in the community size, which is infeasible for large
    communities, so when a community would produce more than
    max_pairs_per_community pairs the co-participation edges are sampled to that
    many random pairs with a fixed seed. Communities above max_community_size are
    skipped entirely. Both are pre-registered, reproducible construction choices,
    not silent truncation; the caller logs them.
    """
    cal = calibration_events(events, t_cutoff)
    G = nx.Graph()

    fan_communities = defaultdict(lambda: defaultdict(int))
    community_fans = defaultdict(set)
    for e in cal:
        G.add_node(e["fan"])
        fan_communities[e["fan"]][e["community"]] += 1
        community_fans[e["community"]].add(e["fan"])

    # Accumulate edge weights in a dict keyed by the sorted pair, then add all
    # edges in one pass. This avoids repeated has_edge lookups and is far faster
    # than incremental insertion on dense co-participation cliques.
    weights = defaultdict(float)
    rng = np.random.default_rng(seed)

    for community, fan_set in community_fans.items():
        fans = sorted(fan_set)
        m = len(fans)
        if m < 2 or m > max_community_size:
            continue
        total_pairs = m * (m - 1) // 2
        if total_pairs <= max_pairs_per_community:
            for u, v in itertools.combinations(fans, 2):
                weights[(u, v)] += 1.0
        else:
            ii = rng.integers(0, m, size=max_pairs_per_community)
            jj = rng.integers(0, m, size=max_pairs_per_community)
            for a, b in zip(ii, jj):
                if a == b:
                    continue
                u, v = (fans[a], fans[b]) if fans[a] < fans[b] else (fans[b], fans[a])
                weights[(u, v)] += 1.0

    for e in cal:
        r = e.get("reply_to")
        if r is None or r == e["fan"]:
            continue
        G.add_node(r)
        key = (e["fan"], r) if e["fan"] < r else (r, e["fan"])
        weights[key] += 1.0

    G.add_weighted_edges_from((u, v, w) for (u, v), w in weights.items())

    # attach per-fan community histogram and home community
    for fan in G.nodes:
        hist = dict(fan_communities.get(fan, {}))
        G.nodes[fan]["communities"] = hist
        G.nodes[fan]["home"] = max(hist, key=hist.get) if hist else None
        G.nodes[fan]["activity"] = float(sum(hist.values()))
    return G
