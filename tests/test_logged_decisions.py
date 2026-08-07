"""The certificate layer must accept the action types 3.0 gave the planning layer.

The regression these tests exist to prevent: ``certify_policy`` and ``conformal_action_value``
typed as ``Sequence[int]``, so an :class:`~causalrl.agents.interventional.InterventionalAgent` --
whose whole point is that an action is an assignment, not an arm index -- could not be handed to
the functions the README points it at without a codebook in between.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from causalrl import (
    ConfoundedTrajectoryDataset,
    Intervention,
    InterventionSpace,
    Transition,
    certify_policy,
    conformal_action_value,
)
from causalrl.agents.interventional import InterventionalAgent
from causalrl.data.logged import (
    FeatureDecisionLog,
    LoggedDecision,
    PositivityReport,
    action_key,
)


class DoseAgent(InterventionalAgent):
    """A deterministic interventional policy: give the high dose iff the biomarker is high."""

    def act(
        self, observation: Mapping[str, Any], *, space: InterventionSpace, deadline: Any = None
    ) -> Intervention:
        return {"dose": 1.0 if float(observation["biomarker"]) > 0.0 else 0.0}

    def update(
        self, observation: Mapping[str, Any], intervention: Intervention, reward: float
    ) -> None:
        return None


def _dose_log(seed: int = 0, n: int = 400) -> tuple[FeatureDecisionLog[Intervention], list[float]]:
    """A feature-space log whose actions are interventions, plus the biomarker column."""
    rng = np.random.default_rng(seed)
    biomarkers = rng.normal(size=n)
    decisions: list[LoggedDecision[Intervention]] = []
    for x in biomarkers:
        # Behaviour policy: high dose with probability 0.7 when the biomarker is high.
        p_high = 0.7 if x > 0.0 else 0.3
        high = bool(rng.random() < p_high)
        dose = 1.0 if high else 0.0
        decisions.append(
            LoggedDecision(
                state=np.array([x]),
                action={"dose": dose},
                reward=float(dose * 1.0 + 0.5 * x + rng.normal(scale=0.1)),
                propensity=p_high if high else 1.0 - p_high,
            )
        )
    return FeatureDecisionLog(decisions), [float(x) for x in biomarkers]


def test_an_interventional_agents_actions_certify_without_a_codebook() -> None:
    """The headline: act() -> certify_policy, with no arm index anywhere in between."""
    log, biomarkers = _dose_log()
    agent = DoseAgent()
    space = InterventionSpace.create({"dose": (0.0, 1.0)})
    target_actions = [agent.act({"biomarker": x}, space=space) for x in biomarkers]

    cert = certify_policy(log, target_actions)

    assert cert.decision in {"prefer learned policy", "prefer behavior", "indifferent"}
    assert cert.recommendation in {"act", "abstain"}
    assert isinstance(cert.naive_contrast, float)


def test_interventions_match_regardless_of_key_order() -> None:
    """Two mappings that denote the same assignment are the same action."""
    assert action_key({"a": 1, "b": 2}) == action_key({"b": 2, "a": 1})
    decisions = [
        LoggedDecision(np.array([0.0]), {"a": 1, "b": 2}, 1.0, 0.5),
        LoggedDecision(np.array([0.0]), {"a": 1, "b": 3}, 0.0, 0.5),
    ]
    log = FeatureDecisionLog(decisions)
    assert list(log.matches([{"b": 2, "a": 1}, {"b": 2, "a": 1}])) == [True, False]


def test_unknowable_positivity_hedges_rather_than_passing() -> None:
    """A log with no behaviour model must say it could not check, not that it found nothing."""
    log, biomarkers = _dose_log()
    agent = DoseAgent()
    space = InterventionSpace.create({"dose": (0.0, 1.0)})
    targets = [agent.act({"biomarker": x}, space=space) for x in biomarkers]

    report = log.positivity(targets)
    assert report.checkable is False
    assert report.violated is False  # no gaps found, because none could be looked for

    cert = conformal_action_value(log, targets, alpha=0.1)
    assert cert.hedge is not None
    assert "could not be checked" in cert.hedge.reason
    positivity = next(a for a in cert.assumptions if a.name == "positivity")
    assert positivity.checkable is False


def test_a_behaviour_model_makes_positivity_checkable() -> None:
    """Supplying pi_behavior(a | s) turns the uncheckable hedge into a real check."""
    decisions = [
        LoggedDecision(np.array([0.0]), {"dose": 0.0}, 1.0, 0.5),
        LoggedDecision(np.array([1.0]), {"dose": 0.0}, 1.0, 0.5),
    ]

    def behavior(state: Any, action: Intervention) -> float:
        # The high dose was never available at state 0.
        return 0.0 if float(state[0]) == 0.0 and action["dose"] == 1.0 else 0.5

    log = FeatureDecisionLog(decisions, behavior_propensity=behavior)
    report = log.positivity([{"dose": 1.0}, {"dose": 1.0}])
    assert report.checkable is True
    assert report.violated is True
    assert len(report.gaps) == 1


def test_tabular_dataset_still_satisfies_the_protocol() -> None:
    """The existing tabular log is a LoggedDecisions[int] with no changes at the call site."""
    transitions = [Transition(s, a, float(a), 0, True) for s in (0, 1) for a in (0, 1)] * 25
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)
    targets = [1] * len(transitions)

    assert len(list(dataset.outcomes())) == len(transitions)
    assert all(p > 0.0 for p in dataset.logging_propensities())
    assert dataset.positivity(targets) == PositivityReport(checkable=True, gaps=())
    assert certify_policy(dataset, targets).decision.startswith("prefer")
    assert conformal_action_value(dataset, targets, alpha=0.2).ci is not None


def test_mismatched_lengths_are_refused_by_both_logs() -> None:
    log, _ = _dose_log(n=10)
    with pytest.raises(ValueError, match="one for one"):
        log.matches([{"dose": 1.0}])
    transitions = [Transition(0, 0, 1.0, 0, True)] * 4
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=1)
    with pytest.raises(ValueError, match="one for one"):
        dataset.matches([0])


def test_a_zero_propensity_decision_is_refused_at_construction() -> None:
    """An action that appears in the log had positive probability by construction."""
    with pytest.raises(ValueError, match="propensity"):
        LoggedDecision(np.array([0.0]), 0, 1.0, 0.0)


def test_an_empty_feature_log_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        FeatureDecisionLog([])
