"""Validate the graph features on a small hand-built example with known values."""
import math

import networkx as nx

from fancoldstart.graph import features as F


def _tiny_graph():
    G = nx.Graph()
    # A engages evenly in c1 and c2 -> breadth entropy = ln 2
    G.add_node("A", communities={"c1": 2, "c2": 2}, home="c1", activity=4.0)
    G.add_node("B", communities={"c1": 3}, home="c1", activity=3.0)
    G.add_node("C", communities={"c2": 3}, home="c2", activity=3.0)
    # A connects to B (home c1) and C (home c2); B and C not connected
    G.add_edge("A", "B", weight=1.0)
    G.add_edge("A", "C", weight=1.0)
    return G


def test_breadth_entropy():
    G = _tiny_graph()
    assert abs(F.community_breadth(G, "A") - math.log(2)) < 1e-9
    assert abs(F.community_breadth(G, "B") - 0.0) < 1e-9


def test_bridging_simpson():
    G = _tiny_graph()
    # A's neighbors are in two different home communities, one each:
    # Simpson diversity = 1 - (0.5^2 + 0.5^2) = 0.5
    assert abs(F.bridging_score(G, "A") - 0.5) < 1e-9
    # B has a single neighbor (A, home c1): 1 - 1 = 0
    assert abs(F.bridging_score(G, "B") - 0.0) < 1e-9


def test_density_clustering():
    G = _tiny_graph()
    # A's two neighbors are not connected -> clustering 0
    assert abs(F.neighborhood_density(G, "A") - 0.0) < 1e-9
