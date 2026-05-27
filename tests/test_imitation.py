"""Causal imitation: the imitability criterion and the two learners."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.envs.suite.imitation import (
    ImitationEnv,
    generate_demonstrations,
    make_imitation_diagram,
    make_unconfounded_observed_diagram,
)
from causalrl.exceptions import CausalGraphError
from causalrl.imitation import (
    BehavioralCloning,
    CausalImitator,
    imitation_backdoor_set,
    is_imitable,
)


def test_imitable_when_confounder_is_observed() -> None:
    graph, observable = make_imitation_diagram()
    z = imitation_backdoor_set(graph, action="A", outcome="Y", observable=observable)
    assert z == frozenset({"W"})
    assert is_imitable(graph, action="A", outcome="Y", observable=observable)


def test_not_imitable_with_latent_confounder() -> None:
    graph, observable = make_unconfounded_observed_diagram()
    assert imitation_backdoor_set(graph, action="A", outcome="Y", observable=observable) is None
    assert not is_imitable(graph, action="A", outcome="Y", observable=observable)


def test_unknown_nodes_raise() -> None:
    graph, observable = make_imitation_diagram()
    with pytest.raises(CausalGraphError):
        imitation_backdoor_set(graph, action="A", outcome="Q", observable=observable)
    with pytest.raises(CausalGraphError):
        imitation_backdoor_set(graph, action="A", outcome="Y", observable={"W", "Q"})


def test_causal_imitator_learns_the_expert_conditional() -> None:
    demos = generate_demonstrations(ImitationEnv(seed=0), n=2000, seed=0)
    agent = CausalImitator(n_actions=2, adjustment=["W"], seed=0)
    agent.fit(demos, action="A")
    # The expert plays A = W deterministically, so P(A=w | W=w) is ~1.
    assert agent.act({"W": 0}) == 0
    assert agent.act({"W": 1}) == 1


def test_behavioral_cloning_learns_the_marginal() -> None:
    demos = generate_demonstrations(ImitationEnv(seed=1), n=4000, seed=1)
    agent = BehavioralCloning(n_actions=2, seed=0)
    agent.fit(demos, action="A")
    draws = [agent.act({}) for _ in range(4000)]
    assert abs(float(np.mean(draws)) - 0.5) < 0.05  # P(A=1) = P(W=1) ~ 0.5
