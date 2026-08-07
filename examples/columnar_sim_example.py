"""Worked example (plan §10): support causalrl from any simulator by emitting a TrajectoryLog.

A self-contained stand-in for an external simulator — here a tiny "promotion targeting" bandit —
shows the whole contract: emit rows in the columnar schema, wrap them with
``simulator_from_callables``, and the log flows straight into causalrl's streaming off-policy
certificate. No causalrl dependency on the simulator, and no simulator import in causalrl.

Run: python examples/columnar_sim_example.py
"""

from __future__ import annotations

from typing import Any

import numpy as np

from causalrl.interop.columnar_sim import check_conformance, simulator_from_callables
from causalrl.ope.ipw import stream_policy_value


def promotion_simulator_rows(n: int, seed: int | None) -> list[dict[str, Any]]:
    """Emit an off-policy log: a context Z, a logged action under a behaviour policy, a reward, and
    the importance weight of a target policy — one decision (entity) per customer."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)  # customer context
    behaviour_p = 1.0 / (1.0 + np.exp(-0.8 * z))  # logging policy P(promote | z)
    action = rng.binomial(1, behaviour_p).astype(float)
    reward = 1.5 * action + 1.0 * z + 0.5 * rng.standard_normal(n)  # promotion lifts reward

    target_p = 1.0 / (1.0 + np.exp(-0.3 * z))  # a gentler target policy
    pb = action * behaviour_p + (1.0 - action) * (1.0 - behaviour_p)
    pt = action * target_p + (1.0 - action) * (1.0 - target_p)
    weight = pt / pb

    rows: list[dict[str, Any]] = []
    for i in range(n):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "obs", "name": "context", "value": float(z[i])})
        rows.append({**base, "kind": "action", "name": "action", "value": float(action[i])})
        rows.append({**base, "kind": "reward", "name": "reward", "value": float(reward[i])})
        rows.append({**base, "kind": "weight", "name": "weight", "value": float(weight[i])})
    return rows


def main() -> None:
    sim = simulator_from_callables(promotion_simulator_rows, metadata={"source": "promotion-sim"})

    report = check_conformance(sim, n=32)
    print("conformance:", report)

    log = sim.sample(50_000, seed=0)
    cert = stream_policy_value(log, weight="weight", reward="reward")
    print(cert)
    assert cert.value is not None, "expected an identified IS value under positivity"
    assert cert.ci is not None
    lo, hi = cert.ci.lower, cert.ci.upper
    print(f"target-policy value V(pi) = {cert.value:.4f}  CI [{lo:.4f}, {hi:.4f}]")
    print("OK — external simulator certified through the columnar contract")


if __name__ == "__main__":
    main()
