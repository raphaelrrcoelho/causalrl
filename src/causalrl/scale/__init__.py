"""Scale path: certify a policy from an external offline-RL trainer against hidden confounding.

causalrl supplies the causal layer, not a new trainer. Train a policy however you like (e.g. with
``d3rlpy`` — see :func:`causalrl.scale.d3rlpy.to_mdp_dataset`), then hand its chosen actions to
:func:`certify_policy` to bound whether its value improvement over the logging/behaviour policy
survives hidden confounding, and — with ``alpha`` — whether it clears the finite-sample downside
gate of :func:`causalrl.conformal_action_value`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from causalrl.conformal.core import conformal_action_value
from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.identification.decision import DecisionCertificate, certify_estimate
from causalrl.identification.estimate import PolicyValueContrast

__all__ = ["certify_policy"]


def certify_policy(
    dataset: ConfoundedTrajectoryDataset,
    target_actions: Sequence[int],
    *,
    gamma_max: float = 10.0,
    alpha: float | None = None,
) -> DecisionCertificate:
    """Certify whether a learned policy's value improvement over the behaviour policy is robust
    to hidden confounding.

    ``target_actions[i]`` is the action the learned policy would take at the ``i``-th logged
    transition's state (e.g. from a d3rlpy policy's greedy prediction, using the same observation
    encoding it was trained on). The contrast is the off-policy value ``V(pi) - V(behaviour)`` under
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
    transitions = dataset.transitions
    if len(target_actions) != len(transitions):
        raise ValueError("target_actions must have one action per logged transition")
    outcomes = [tr.reward for tr in transitions]
    e0 = [dataset.behavior_propensity(tr.state, tr.action) for tr in transitions]
    target_on = [
        1.0 if int(a) == tr.action else 0.0
        for a, tr in zip(target_actions, transitions, strict=True)
    ]
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
