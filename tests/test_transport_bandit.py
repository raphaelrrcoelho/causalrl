"""TransportableConfoundedBandit: oracle correctness and the orthogonal Gamma/shift knobs."""

from __future__ import annotations

import pytest

from causalrl.envs.suite.transport_bandit import TransportableConfoundedBandit
from causalrl.identification.criteria import backdoor_adjustment_set


def test_safe_arm_is_optimal_in_both_domains() -> None:
    # Arm 1's Z term is balanced and its W term is a penalty, so arm 0 (flat 0.5) is always optimal.
    for gamma in (0.0, 0.5, 1.0):
        for shift in (0.0, 0.4, 1.0):
            env = TransportableConfoundedBandit(gamma=gamma, shift=shift)
            assert env.optimal_action(domain="source") == 0
            assert env.optimal_action(domain="target") == 0
            assert env.true_action_value(0, domain="target") == 0.5


def test_arm1_target_value_matches_closed_form() -> None:
    env = TransportableConfoundedBandit(gamma=0.5, shift=0.6)
    # E[Y|do(1)] = 0.5 - DT * P_target(W=1); DT=0.20, target P(W=1) = 0.5 + shift/2 = 0.8.
    assert env.true_action_value(1, domain="target") == pytest.approx(0.5 - 0.20 * 0.8)
    # The safe arm's value never depends on the domain.
    assert env.true_action_value(1, domain="source") == pytest.approx(0.5 - 0.20 * 0.2)


def test_backdoor_set_is_the_confounder_only() -> None:
    env = TransportableConfoundedBandit(gamma=0.5, shift=0.5)
    assert set(env.graph.nodes) == {"Z", "W", "A", "Y"}
    # Z is the only back-door confounder; W (a selection variable, not into A) is not needed.
    assert set(backdoor_adjustment_set(env.graph, "A", "Y")) == {"Z"}


def test_sampling_matches_the_mechanism_cell_mean() -> None:
    env = TransportableConfoundedBandit(gamma=0.8, shift=0.6, seed=0)
    data = env.sample(200_000, domain="source", seed=0)
    z, w, a, y = data["Z"], data["W"], data["A"], data["Y"]
    cell = (a == 1) & (z == 1) & (w == 0)
    # E[Y | A=1, Z=1, W=0] = 0.5 + DC = 0.75.
    assert y[cell].mean() == pytest.approx(0.75, abs=0.02)


def test_gamma_one_preserves_overlap() -> None:
    # A deterministic policy (A=Z) would break positivity; the 0.4 slope keeps ~0.1 overlap.
    env = TransportableConfoundedBandit(gamma=1.0, shift=0.0, seed=0)
    data = env.sample(50_000, domain="source", seed=0)
    z, a = data["Z"], data["A"]
    assert (a[z == 0] == 1).mean() == pytest.approx(0.1, abs=0.02)


def test_rejects_out_of_range_knobs() -> None:
    with pytest.raises(ValueError):
        TransportableConfoundedBandit(gamma=1.5, shift=0.0)
    with pytest.raises(ValueError):
        TransportableConfoundedBandit(gamma=0.0, shift=-0.1)
