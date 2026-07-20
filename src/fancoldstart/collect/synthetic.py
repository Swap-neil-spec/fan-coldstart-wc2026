"""SYNTHETIC event generator, for pipeline validation ONLY.

This produces labeled synthetic engagement events with a planted, weak link
between a fan's community position and their future engagement, so that the
pipeline and the hypothesis tests have something to detect when validating that
the code runs end to end. This is NOT real data and its outputs are NOT research
results. The paper reports nothing from this generator. Real runs use the
collectors in this package against public sources.
"""
import numpy as np


def generate(n_fans=1500, n_communities=12, T_cal=40.0, tau=20.0, seed=13):
    rng = np.random.default_rng(seed)
    communities = [f"c{j}" for j in range(n_communities)]
    horizon = T_cal + tau

    # per-community engagement effect: fans who share a home community are
    # somewhat alike, which gives the neighbor-smoothing mechanism (H1) real
    # rate homophily to borrow from.
    community_effect = rng.uniform(0.0, 1.0, n_communities)

    # each fan's community set: breadth_pref controls how many
    breadth_pref = rng.uniform(0.1, 1.0, n_fans)
    fan_comms = []
    home_of = []
    for i in range(n_fans):
        k = 1 + rng.binomial(n_communities - 1, 0.15 + 0.5 * breadth_pref[i])
        cs = list(rng.choice(n_communities, size=min(k, n_communities), replace=False))
        fan_comms.append(cs)
        home_of.append(cs[0])

    # engagement is driven by the fan's own breadth (H3, H4) and by their home
    # community's effect (H1 rate homophily), plus idiosyncratic noise.
    home_eff = np.array([community_effect[h] for h in home_of])
    engagement = np.clip(
        0.40 * breadth_pref + 0.35 * home_eff + 0.25 * rng.uniform(0, 1, n_fans), 0, 1)
    lam = np.exp(np.log(0.30) + 1.1 * (engagement - 0.5))          # events/day
    p_drop = 1.0 / (1.0 + np.exp(-(-0.2 - 1.6 * (engagement - 0.5))))  # dropout prob

    community_members = {j: [] for j in range(n_communities)}
    for i in range(n_fans):
        community_members[home_of[i]].append(i)

    events = []
    for i in range(n_fans):
        t = 0.0
        alive = True
        cs = fan_comms[i]
        while alive:
            t += rng.exponential(1.0 / lam[i])
            if t > horizon:
                break
            # community for this event: home with prob 0.6 else spread
            if len(cs) > 1 and rng.random() > 0.6:
                comm = int(rng.choice(cs[1:]))
            else:
                comm = cs[0]
            reply_to = None
            if rng.random() < 0.3:
                members = community_members.get(comm, [])
                if members:
                    j = int(rng.choice(members))
                    if j != i:
                        reply_to = f"f{j}"
            events.append({"fan": f"f{i}", "community": communities[comm],
                           "t": float(t), "reply_to": reply_to, "platform": "synthetic"})
            if rng.random() < p_drop[i]:
                alive = False
    return events
