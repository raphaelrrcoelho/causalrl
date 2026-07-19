"""Continuous-confounder bandit: the M3 function-approximation tier.

Where M0-M2 back-door-adjust over a handful of discrete strata, M3 keeps a **continuous** observed
confounder ``Z ~ Uniform(0, 1)`` and a **nonlinear** reward, so recovering the interventional value
needs a learned continuous outcome model, not a lookup table -- the "function-approximation scale"
credibility tier, self-contained so it runs in CI.

Graph ``Z -> A, Z -> Y, A -> Y``. The safe arm 0 pays a flat 0.5. Arm 1 pays a narrow reward bump
peaked at high ``z``:  ``q(1, z) = BASE + AMP * exp(-(z - CENTER)^2 / (2 WIDTH^2))``. Averaged over
``Z ~ Uniform`` the bump is thin, so ``E[Y | do(1)] < 0.5`` and **arm 0 is truly optimal**. But the
behavior policy over-samples arm 1 exactly where the bump is high --
``P(A=1 | Z=z) = clip(0.5 + SLOPE*gamma*(2z - 1), 0.03, 0.97)`` (overlap preserved even at
``gamma=1``) -- so the naive marginal ``E[Y | A=1]`` is inflated **above 0.5** and a correlational
agent adopts the harmful arm 1. A function-approximation agent that fits ``q(a, z)`` (ridge on RBF
features) and integrates it over the observed ``Z`` recovers the true low value of arm 1 and keeps
arm 0 -- including at low ``z``, where arm 1 is rarely played and the smooth basis interpolates.

Numbers (BASE=0.10, AMP=1.20, CENTER=0.85, WIDTH=0.10): ``E[Y|do(0)]=0.500``,
``E[Y|do(1)]≈0.381``; at ``gamma=1`` the confounded ``E[Y|A=1]≈0.56 > 0.5`` (naive is fooled).
"""

from __future__ import annotations

import numpy as np

from causalrl.scm.graph import CausalGraph

_BASE, _AMP, _CENTER, _WIDTH = 0.10, 1.20, 0.85, 0.10  # arm-1 reward bump q(1, z)
_SLOPE = 0.45  # confounded-propensity slope; < 0.5 keeps positivity (overlap) at gamma=1
_NOISE = 0.05  # observation noise on the continuous reward


class ContinuousConfoundedBandit:
    """One-decision bandit with a continuous observed confounder ``Z`` and a nonlinear arm-1 reward.

    ``Z`` is the back-door confounder; the reward is continuous (Gaussian observation noise), so the
    outcome model must be a function approximator rather than a per-stratum mean.
    """

    n_states: int = 1
    n_actions: int = 2

    def __init__(self, *, gamma: float = 1.0, seed: int | None = None) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        self.gamma = float(gamma)
        self.graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("A", "Y")])
        self._rng = np.random.default_rng(seed)

    def _reward_mean(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Vectorized ``E[Y | A=a, Z=z]``: flat 0.5 for arm 0, the bump ``q(1, z)`` for arm 1."""
        arm1 = _BASE + _AMP * np.exp(-((z - _CENTER) ** 2) / (2 * _WIDTH**2))
        return np.where(a == 1, arm1, 0.5)

    def sample(self, n: int, *, seed: int | None = None) -> dict[str, np.ndarray]:
        """Draw ``n`` confounded observations ``{"Z", "A", "Y"}`` (``Z``/``Y`` continuous)."""
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        z = rng.random(n)
        a = (rng.random(n) < np.clip(0.5 + _SLOPE * self.gamma * (2 * z - 1), 0.03, 0.97)).astype(
            int
        )
        y = self._reward_mean(a, z) + rng.normal(0.0, _NOISE, n)
        return {"Z": z, "A": a, "Y": y}

    def true_action_value(self, action: int, *, n_mc: int = 200_000) -> float:
        """Exact ``E[Y | do(A=action)] = E_z[E[Y | action, z]]`` over ``Z ~ Uniform(0, 1)``."""
        z = np.linspace(0.0, 1.0, n_mc)
        return float(self._reward_mean(np.full(n_mc, action), z).mean())

    def optimal_action(self) -> int:
        return max(range(self.n_actions), key=self.true_action_value)

    def optimal_value(self) -> float:
        return max(self.true_action_value(a) for a in range(self.n_actions))
