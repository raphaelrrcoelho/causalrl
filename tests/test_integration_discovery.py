"""Headline: discover the causal model from data, then plan interventions on it (POMIS)."""

from __future__ import annotations

from causalrl.discovery import discover
from causalrl.envs.suite.discovery import sample_discovery_data
from causalrl.identification.intervention_sets import pomis


def test_discovered_graph_feeds_pomis() -> None:
    data = sample_discovery_data(n=10_000, seed=0)
    graph = discover(data, ["X", "Y", "Z", "W"]).to_causal_graph()
    # The agent learned the structure; now plan interventions to optimize the reward W.
    sets = pomis(graph, "W")
    # With no latent confounding, the POMIS is exactly {Z}: intervene on the discovered cause of W.
    assert sets == [frozenset({"Z"})]
