"""Task guide 3: transport a claim across regimes, including a mechanism swap.

A simulation-based-inference or NumPyro posterior over environment parameters gives a *cloud of
calibrated configurations*. `regimes_from_posterior` turns it into a set of Regimes; a Regime can
also mark a **mechanism swap** via a selection node (the target regime replaces one mechanism).
`across_regimes` then reports the [min, max] envelope of a functional over the ensemble — the
worst-case transported value across everything consistent with the data.

Run: python examples/guides/03_transport_across_regimes.py
"""

from __future__ import annotations

import numpy as np

from causalrl import Regime, across_regimes, regimes_from_posterior


def main() -> None:
    rng = np.random.default_rng(0)
    # Posterior over a treatment-effect and a context-shift parameter (calibrated to data).
    posterior = {
        "effect": 1.2 + 0.15 * rng.standard_normal(500),
        "context_shift": 0.3 + 0.1 * rng.standard_normal(500),
    }
    regimes = regimes_from_posterior(
        posterior, selection=["reward_mechanism"], max_regimes=200, seed=0
    )
    print(f"{len(regimes)} calibrated regimes; each marks 'reward_mechanism' as swapped")

    # A transported functional: the policy value under each regime's parameters (a mechanism swap
    # that replaces the reward mechanism with effect*action + context_shift * E[context]).
    def transported_value(regime: Regime) -> float:
        return regime.params["effect"] * 1.0 + regime.params["context_shift"] * 0.5

    envelope = across_regimes(regimes, transported_value)
    print(
        f"transported value envelope across the posterior: "
        f"[{envelope.lower:.4f}, {envelope.upper:.4f}]"
    )

    # The single posterior-mean (target) regime, for a point transport.
    target = Regime.create(
        "target", selection=["reward_mechanism"], parameters={"effect": 1.2, "context_shift": 0.3}
    )
    print(f"target regime selection nodes: {sorted(target.selection)}")

    assert envelope.lower <= transported_value(target) <= envelope.upper
    print("OK — transported across calibrated regimes with a mechanism swap")


if __name__ == "__main__":
    main()
