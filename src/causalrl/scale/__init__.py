"""Scale path: certify a policy from an external offline-RL trainer against hidden confounding.

causalrl supplies the causal layer, not a new trainer. Train a policy however you like (e.g. with
``d3rlpy`` — see :func:`causalrl.scale.d3rlpy.to_mdp_dataset`), then hand its chosen actions to
:func:`certify_policy` to bound whether its value improvement over the logging/behaviour policy
survives hidden confounding.
"""

from __future__ import annotations

from collections.abc import Sequence

from causalrl.data.dataset import ConfoundedTrajectoryDataset
from causalrl.identification.decision import DecisionCertificate, certify_estimate
from causalrl.identification.estimate import PolicyValueContrast

__all__ = ["certify_policy"]


def certify_policy(
    dataset: ConfoundedTrajectoryDataset,
    target_actions: Sequence[int],
    *,
    gamma_max: float = 10.0,
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
    return certify_estimate(contrast, gamma_max=gamma_max, labels=("learned policy", "behavior"))
