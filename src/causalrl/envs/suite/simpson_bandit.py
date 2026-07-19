"""Simpson's-paradox bandit: an OBSERVED confounder Z on a back-door path A <- Z -> Y.

A single-decision bandit where action 1 is better within each Z stratum and interventionally, yet the
naive marginal ``E[Y | A]`` *reverses* (action 0 looks better) because Z confounds the logged action.
An agent that back-door-adjusts for the observed Z recovers the interventional optimum; a naive
marginal agent is fooled. This is the M0 (b) substrate — the identifiable causal win that an *active*
deconfounded optimizer can achieve (unlike the certify-gated agent, whose ceiling is the behavior
policy).

Numbers (``P(Y=1 | A, Z)``): [[0.6, 0.2], [0.7, 0.3]] indexed ``[action, z]``; behavior
``P(A=1 | Z) = [0.2, 0.8]``. Interventional: ``E[Y|do(0)] = 0.40``, ``E[Y|do(1)] = 0.50`` (optimum).
Naive marginal reverses to prefer action 0.
"""

from __future__ import annotations

import numpy as np

from causalrl.scm.graph import CausalGraph

# P(Y = 1 | A = a, Z = z), indexed [a, z].
_P_REWARD = np.array([[0.6, 0.2], [0.7, 0.3]])
# Behavior policy P(A = 1 | Z = z), indexed [z].
_P_TREAT = np.array([0.2, 0.8])


class SimpsonBandit:
    """Single-decision bandit with an observed confounder Z on the back-door path A <- Z -> Y."""

    n_states: int = 1
    n_actions: int = 2

    def __init__(self, *, seed: int | None = None) -> None:
        self.graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
        self._rng = np.random.default_rng(seed)

    def sample(self, n: int, *, seed: int | None = None) -> dict[str, np.ndarray]:
        """Draw ``n`` confounded observations as columnar arrays ``{"Z", "A", "Y"}``."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        z = (rng.random(n) < 0.5).astype(int)
        a = (rng.random(n) < _P_TREAT[z]).astype(int)
        y = (rng.random(n) < _P_REWARD[a, z]).astype(float)
        return {"Z": z, "A": a, "Y": y}

    def true_action_value(self, action: int) -> float:
        """Exact interventional value ``E[Y | do(A = action)] = Σ_z P(z) · P(Y=1 | action, z)``."""
        return float(0.5 * _P_REWARD[action, 0] + 0.5 * _P_REWARD[action, 1])

    @property
    def optimal_value(self) -> float:
        return max(self.true_action_value(a) for a in range(self.n_actions))
