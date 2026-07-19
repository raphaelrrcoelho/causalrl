"""Transportable confounded bandit: the M2 phase-diagram substrate, with two orthogonal knobs.

The failure modes of correlational RL are put on **separate variables** so a 2-D phase diagram over
(confounding strength Gamma) x (covariate-shift magnitude) does not entangle them:

- Confounder ``Z`` drives the **Gamma axis**. The behavior policy conditions on Z as
  ``P(A=1 | Z=z) = 0.5 + 0.4 * gamma * (2z - 1)``: unconfounded at ``gamma=0``, and at ``gamma=1``
  the propensity is ``0.1 / 0.9`` -- strongly confounded but still OVERLAPPING (positivity holds, so
  the effect stays identifiable; a deterministic ``A=Z`` would make the whole grid unidentifiable).
- Shift variable ``W`` drives the **shift axis**. ``W`` affects the reward but not the action, and
  its prevalence spreads symmetrically between domains: ``P(W=1) = 0.5 - shift/2`` in the source and
  ``0.5 + shift/2`` in the target.

Reward mean (INVARIANT across domains; only ``P(W)`` differs) with ``a`` the action:
``E[Y | A=a, Z=z, W=w] = 0.5`` for the safe arm ``a=0``, and for ``a=1``
``0.5 + [+DC if z==1 else -DC] + [-DT if w==1 else 0]``. The Z term is balanced over ``P(Z=1)=0.5``,
so it adds nothing interventionally (``E[Y|do(1)] = 0.5 - DT*P(W=1) < 0.5``): the **safe arm 0
is optimal in both domains**. But the confounded behavior over-samples arm 1 where its Z term is
positive, so the naive marginal ``E[Y|A=1]`` is inflated by ``~DC*gamma`` and, once that overpowers
arm 1's source penalty, a correlational agent adopts the harmful arm 1 — whose target cost
``DT*P_target(W=1)`` grows with the shift. A causal agent that adjusts for Z and transports
W by the target distribution avoids arm 1 in every cell.

The causal-minus-naive gap is therefore monotone nondecreasing in both gamma and shift, with a
diagonal phase boundary ``gamma*(shift) = 0.5 - shift/2`` — the "confounding bites where theory
predicts" signature (verified end-to-end at finite sample, including the tight-overlap gamma=1 edge,
before implementation).
"""

from __future__ import annotations

import numpy as np

from causalrl.scm.graph import CausalGraph

_DC = 0.25  # confounding amplitude on Z (drives the Gamma axis)
_DT = 0.20  # transport/shift amplitude on W (drives the shift axis)
_C = 0.40  # confounded-propensity slope; < 0.5 keeps positivity (overlap) at gamma=1
_P_Z1 = 0.5  # P(Z=1), identical in both domains


class TransportableConfoundedBandit:
    """One-decision bandit with an orthogonal confounder ``Z`` and shift variable ``W``.

    Graph ``Z -> A, Z -> Y, W -> Y, A -> Y``. ``Z`` is the back-door confounder; ``W`` is the
    selection variable whose distribution differs between source and target.
    """

    n_states: int = 1
    n_actions: int = 2

    def __init__(self, *, gamma: float, shift: float, seed: int | None = None) -> None:
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if not 0.0 <= shift <= 1.0:
            raise ValueError("shift must be in [0, 1]")
        self.gamma = float(gamma)
        self.shift = float(shift)
        self.w_source = 0.5 - self.shift / 2.0
        self.w_target = 0.5 + self.shift / 2.0
        self.graph = CausalGraph(directed_edges=[("Z", "A"), ("Z", "Y"), ("W", "Y"), ("A", "Y")])
        self._rng = np.random.default_rng(seed)

    def _reward_mean(self, a: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
        """Vectorized ``E[Y | A=a, Z=z, W=w]``. Safe arm 0 is flat 0.5; arm 1 adds the Z/W terms."""
        arm1 = 0.5 + _DC * np.where(z == 1, 1.0, -1.0) + np.where(w == 1, -_DT, 0.0)
        return np.where(a == 1, arm1, 0.5)

    def sample(
        self, n: int, *, domain: str = "source", seed: int | None = None
    ) -> dict[str, np.ndarray]:
        """Draw ``n`` confounded observations ``{"Z", "W", "A", "Y"}`` from ``domain``.

        ``domain`` is ``"source"`` (log-collection) or ``"target"`` (deployment); they differ only
        in ``P(W=1)``.
        """
        if domain not in ("source", "target"):
            raise ValueError("domain must be 'source' or 'target'")
        rng = np.random.default_rng(seed) if seed is not None else self._rng
        p_w = self.w_source if domain == "source" else self.w_target
        z = (rng.random(n) < _P_Z1).astype(int)
        w = (rng.random(n) < p_w).astype(int)
        a = (rng.random(n) < 0.5 + _C * self.gamma * (2 * z - 1)).astype(int)
        y = (rng.random(n) < self._reward_mean(a, z, w)).astype(float)
        return {"Z": z, "W": w, "A": a, "Y": y}

    def true_action_value(self, action: int, *, domain: str = "target") -> float:
        """Exact ``E[Y | do(A=action)] = Σ_{z,w} P(z) P(w) E[Y | action, z, w]`` in ``domain``."""
        p_w = self.w_target if domain == "target" else self.w_source
        z = np.array([0, 0, 1, 1])
        w = np.array([0, 1, 0, 1])
        means = self._reward_mean(np.full(4, action), z, w)
        p_z = np.where(z == 1, _P_Z1, 1.0 - _P_Z1)
        p_wv = np.where(w == 1, p_w, 1.0 - p_w)
        return float((p_z * p_wv * means).sum())

    def optimal_action(self, *, domain: str = "target") -> int:
        return max(range(self.n_actions), key=lambda a: self.true_action_value(a, domain=domain))

    def optimal_value(self, *, domain: str = "target") -> float:
        return max(self.true_action_value(a, domain=domain) for a in range(self.n_actions))
