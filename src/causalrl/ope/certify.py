"""Certify a learned policy's value improvement against hidden confounding.

The end of the off-policy-evaluation pipeline: given confounded logs and the actions an
already-trained policy would take on them, :func:`certify_policy` bounds whether its value
improvement over the logging/behaviour policy survives hidden confounding, and — with ``alpha`` —
whether it clears the finite-sample downside gate of :func:`causalrl.conformal_action_value`.
causalrl supplies the causal layer, not a new trainer; :mod:`causalrl.scale` re-exports this
function for the "train elsewhere (e.g. ``d3rlpy``), certify here" story.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeVar

from causalrl.conformal.core import conformal_action_value
from causalrl.data.logged import LoggedDecisions
from causalrl.identification.decision import DecisionCertificate, certify_estimate
from causalrl.identification.estimate import PolicyValueContrast

__all__ = ["certify_policy"]

ActionT = TypeVar("ActionT")


def certify_policy(
    dataset: LoggedDecisions[ActionT],
    target_actions: Sequence[ActionT],
    *,
    gamma_max: float = 10.0,
    alpha: float | None = None,
) -> DecisionCertificate:
    """Certify whether a learned policy's value improvement over the behaviour policy is robust
    to hidden confounding.

    ``target_actions[i]`` is the action the learned policy would take at the ``i``-th logged
    decision's state (e.g. from a d3rlpy policy's greedy prediction, using the same observation
    encoding it was trained on). ``dataset`` is any
    :class:`~causalrl.data.logged.LoggedDecisions`: the tabular
    :class:`~causalrl.ConfoundedTrajectoryDataset` whose actions are arm indices, or a
    :class:`~causalrl.data.logged.FeatureDecisionLog` whose states are feature vectors and whose
    actions are :data:`~causalrl.Intervention` assignments -- so an
    :class:`~causalrl.agents.interventional.InterventionalAgent` can be certified directly, with no
    arm codebook in between. The action type is inferred from the log, so a mismatched pair is a
    type error rather than a silent miscount.

    The contrast is the off-policy value ``V(pi) - V(behaviour)`` under
    Tan's marginal sensitivity model: the nominal logging propensities are the dataset's empirical
    ``behavior_propensity``, ``pi``'s per-unit target probability is ``1`` where it matches the
    logged action (greedy), and the behaviour arm's target probability is the logging propensity, so
    its self-normalised value is the logged empirical return. Returns the same
    :class:`~causalrl.DecisionCertificate` as :func:`causalrl.certify_decision`.

    Passing ``alpha`` additionally runs the finite-sample **lower-confidence-bound gate** for safe
    policy improvement: :func:`causalrl.conformal_action_value` calibrates a distribution-free
    lower bound on a fresh return under ``pi`` and under the logging policy from the same logs, and
    ``certified`` then requires the confounding verdict *and* ``lcb(pi) >= lcb(behaviour)``. Too
    few effectively-weighted samples for the level leave ``lcb(pi)`` at ``-inf``, which refuses
    rather than passes — no evidence is not evidence of safety. The gate is on the return of a
    single decision, not on ``V(pi)``: a policy with the better mean but a worse downside fails it.

    Honest scope: the MSM sensitivity is on the logging propensities; the behaviour value is the
    logged empirical return (on-policy for the behaviour policy, hence unconfounded). This is the
    one-step / terminal-return contrast; the per-step cumulative-reward extension uses
    :func:`causalrl.msm_per_step_bounds`.
    """
    outcomes = list(dataset.outcomes())
    e0 = list(dataset.logging_propensities())
    target_on = [1.0 if m else 0.0 for m in dataset.matches(target_actions)]
    contrast = PolicyValueContrast(
        outcomes=outcomes,
        logging_propensities=e0,
        target_on=target_on,
        target_off=list(e0),
    )
    cert = certify_estimate(contrast, gamma_max=gamma_max, labels=("learned policy", "behavior"))
    if alpha is None:
        return cert
    band = conformal_action_value(dataset, target_actions, alpha=alpha).ci
    reference = conformal_action_value(dataset, None, alpha=alpha).ci
    assert band is not None and reference is not None  # conformal_action_value always sets ci
    passed = math.isfinite(band.lower) and band.lower >= reference.lower
    summary = (
        f"{cert.summary} Finite-sample downside gate (conformal, alpha={alpha:g}): "
        f"{'PASS' if passed else 'REFUSE'} — one decision under the learned policy returns at "
        f"least {band.lower:.3f}, vs {reference.lower:.3f} under behavior."
    )
    return cert._replace(
        certified=cert.certified and passed, conformal_lcb=band.lower, summary=summary
    )
