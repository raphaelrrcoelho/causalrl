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


def test_a_policy_matching_no_logged_action_is_refused_not_crashed() -> None:
    """No overlap is no evidence, and must be reported rather than reduced over an empty set.

    Every MSM quantity averages over the decisions the target policy would have taken. With none,
    the bound used to reach a numpy ``zero-size array`` reduction several frames below the call --
    an opaque failure where the library's rule requires a refusal.
    """
    log, _ = _dose_log(n=200)
    unreachable = [{"dose": -1.0}] * len(log)

    assert not any(log.matches(unreachable))
    cert = certify_policy(log, unreachable)

    assert cert.certified is False
    assert cert.recommendation == "abstain"
    assert cert.msm_certified is None, "no sensitivity layer can run without a matched decision"
    assert cert.tipping_gamma is None
    assert "REFUSED" in cert.summary and "none of the 200 logged actions" in cert.summary


def test_the_no_overlap_refusal_names_continuous_actions_as_the_likely_cause() -> None:
    """A continuous target never matches a logged float exactly; the message must say so.

    This is the failure a caller actually hits after 3.0 made action domains continuous: the
    decision layer hands back a real number and the certificate layer asks whether it equals the
    logged one. The refusal has to point at banding rather than leave the caller reading frames.
    """
    rng = np.random.default_rng(3)
    decisions: list[LoggedDecision[Intervention]] = [
        LoggedDecision(
            state=np.array([float(i)]),
            action={"deployment": float(rng.uniform())},
            reward=float(rng.normal()),
            propensity=0.5,
        )
        for i in range(50)
    ]
    log: FeatureDecisionLog[Intervention] = FeatureDecisionLog(decisions)
    # A continuous policy: the same value to nine decimal places is still not the same float.
    nudged = [{"deployment": d.action["deployment"] + 1e-9} for d in decisions]

    cert = certify_policy(log, nudged)

    assert cert.certified is False
    assert "continuous" in cert.summary and "band" in cert.summary


def test_the_alpha_gate_also_refuses_rather_than_crashing_without_overlap() -> None:
    """The conformal gate is downstream of the same empty set, so the guard must precede it."""
    log, _ = _dose_log(n=120)
    unreachable = [{"dose": 7.0}] * len(log)

    cert = certify_policy(log, unreachable, alpha=0.1)

    assert cert.certified is False
    assert cert.conformal_lcb is None, "the gate cannot have run: it had nothing to calibrate on"


def test_check_permitted_reports_a_continuous_violation_as_a_value_error() -> None:
    """The documented error must survive a continuous domain.

    ``check_permitted`` built its message with ``space.values(name)``, which raises ``TypeError``
    on a ``Continuous`` domain -- replacing the documented ``ValueError`` with an unrelated one at
    exactly the moment the caller is being told their intervention is inadmissible.
    """
    from causalrl import Continuous
    from causalrl.agents.interventional import InterventionalAgent

    space = InterventionSpace.create({"dose": Continuous(0.0, 1.0)})

    with pytest.raises(ValueError, match="not admissible"):
        InterventionalAgent.check_permitted({"dose": 5.0}, space)
    assert InterventionalAgent.check_permitted({"dose": 0.5}, space) == {"dose": 0.5}
