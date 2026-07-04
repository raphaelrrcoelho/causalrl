"""Confounded offline RL at scale: d3rlpy trains, causalrl certifies.

Train a policy with a d3rlpy offline algorithm on confounded logs, then certify whether its value
improvement over the behaviour policy survives hidden confounding. Run:

    uv run --extra scale python examples/scale_d3rlpy_certify.py

Prints a skip message and exits 0 if d3rlpy is not installed, so it is safe in any environment.
"""

from __future__ import annotations

import numpy as np


def main() -> None:
    try:
        import d3rlpy
        from d3rlpy.algos import DiscreteCQLConfig
    except Exception as exc:  # optional scale stack absent or broken -> skip cleanly
        print(f"[skip] d3rlpy unavailable ({type(exc).__name__}); pip install causalrl[scale].")
        return

    from causalrl import certify_policy
    from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition
    from causalrl.scale.d3rlpy import to_mdp_dataset

    # A confounded 2-state contextual bandit: the paying action equals the state, but the behaviour
    # policy under-explores it, so the naive logged value understates the optimal policy.
    rng = np.random.default_rng(0)
    transitions: list[Transition] = []
    for _ in range(600):
        s = int(rng.random() < 0.5)
        a = int(rng.random() < 0.35)  # behaviour biased toward action 0
        r = 1.0 if a == s else 0.0
        transitions.append(Transition(s, a, r + float(rng.normal(0, 0.05)), s, True))
    dataset = ConfoundedTrajectoryDataset(transitions, n_states=2, n_actions=2)

    d3rlpy.seed(0)
    algo = DiscreteCQLConfig().create()
    algo.fit(to_mdp_dataset(dataset), n_steps=1000, n_steps_per_epoch=500, show_progress=False)

    obs = np.eye(2, dtype=np.float32)[[tr.state for tr in dataset.transitions]]
    target_actions = np.asarray(algo.predict(obs)).ravel().astype(int).tolist()

    cert = certify_policy(dataset, target_actions, gamma_max=20.0)
    print("causalrl certificate for the d3rlpy-learned policy vs. the behaviour policy:")
    print(cert)
    print(f"recommendation: {cert.recommendation}")


if __name__ == "__main__":
    main()
