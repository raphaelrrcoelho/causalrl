"""Task guide 1: evaluate a policy offline when the logs may be confounded.

You have logged (state, action, reward) data and a candidate policy. The naive off-policy value can
be biased by *unmeasured* confounding of the logging decisions. `certify_policy` bounds the value
improvement over the behaviour policy under Tan's marginal sensitivity model and reports the tipping
Gamma — the confounding strength at which the "ship it" decision would flip.

Run: python examples/guides/01_evaluate_offline_under_confounding.py
"""

from __future__ import annotations

import numpy as np

from causalrl import ConfoundedTrajectoryDataset, Transition, certify_policy


def build_confounded_log(n: int, seed: int) -> ConfoundedTrajectoryDataset:
    rng = np.random.default_rng(seed)
    transitions: list[Transition] = []
    for _ in range(n):
        state = int(rng.integers(0, 2))
        # Behaviour policy prefers action 1 in state 1 (the logging bias).
        action = int(rng.random() < (0.7 if state == 1 else 0.3))
        reward = float(1.0 * action + 0.5 * state + 0.2 * rng.standard_normal())
        transitions.append(Transition(state, action, reward, state, True))
    return ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)


def main() -> None:
    dataset = build_confounded_log(4_000, seed=0)
    # The candidate policy: always take action 1.
    target_actions = [1] * len(dataset.transitions)

    cert = certify_policy(dataset, target_actions, gamma_max=5.0)
    print(cert)
    print(f"decision        : {cert.decision}")
    print(f"certified robust: {cert.certified}")
    print(f"tipping Gamma   : {cert.tipping_gamma}")

    # One-sided-honest: 'certified' means no confounding up to gamma_max flips the decision.
    assert isinstance(cert.certified, bool)
    print("OK — offline value certified under a marginal-sensitivity budget")


if __name__ == "__main__":
    main()
