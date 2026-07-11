"""Task guide 4: certify one agent's action effect inside a fixed population.

A `PopulationAgentView` exposes one ego agent embedded in a population as a per-agent causal
environment (`sample` / `do` -> TrajectoryLog). The ego's logging action is confounded by an
observed context Z that also drives a co-player, but adjusting for Z identifies the ego's action
effect — so a doubly-robust `certify_effect` recovers it and matches the interventional truth.

Run: python examples/guides/04_certify_agent_in_population.py
"""

from __future__ import annotations

import numpy as np

from causalrl import CausalGraph, agent_causal_env_view, certify_effect


def _floats(cells: np.ndarray) -> np.ndarray:
    return np.array([float(v) for v in cells.tolist()], dtype=float)


def main() -> None:
    view = agent_causal_env_view(ego="ego", coplayer="co", ego_effect=1.5, confound=1.0)
    log = view.sample(4_000, seed=0)

    data = {
        "Z": _floats(log.values_by_name("Z")),
        "ego": _floats(log.values_by_name("ego")),
        "Y": _floats(log.values_by_name("Y")),
    }
    # Z confounds the ego action and the reward; adjusting for Z identifies the ego effect.
    graph = CausalGraph([("Z", "ego"), ("Z", "Y"), ("ego", "Y")], [], nodes=["Z", "ego", "Y"])

    cert = certify_effect(graph, "ego", "Y", data, method="aipw")
    print(cert)
    assert cert.value is not None and cert.ci is not None
    print(f"estimated ego effect: {cert.value:.3f}  CI [{cert.ci.lower:.3f}, {cert.ci.upper:.3f}]")

    # Interventional ground truth from the view's do().
    do1 = _floats(view.do({"ego": 1}, 20_000, seed=1).values_by_name("Y")).mean()
    do0 = _floats(view.do({"ego": 0}, 20_000, seed=2).values_by_name("Y")).mean()
    print(f"Monte-Carlo do-effect: {do1 - do0:.3f}  (true ego_effect = 1.5)")

    assert abs(cert.value - (do1 - do0)) < 0.1, "DR estimate and MC do-effect should agree"
    assert abs(cert.value - 1.5) < 0.1, "DR should recover the true ego effect (1.5)"
    print("OK — single-learner-in-population OPE matches the interventional ground truth")


if __name__ == "__main__":
    main()
