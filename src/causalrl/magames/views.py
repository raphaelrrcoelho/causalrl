"""A synthetic per-agent environment for one ego inside a fixed population (plan §8.1).

:class:`LinearGaussianPopulationEnv` is a hand-written data-generating process — *not* a learner —
exposed as a :class:`~causalrl.protocols.CausalEnvProtocol` (``sample`` / ``do`` producing a
:class:`~causalrl.data.trajectory.TrajectoryLog`), so the Phase-1 estimation machinery applies to it
directly. The co-players are (context-dependent) mechanisms; an observed context confounds the ego's
logging action, so this is the *single-learner-in-a-fixed-population* topology in which the ego's
action effect is back-door identified — and a Phase-1 DR estimate matches the Monte-Carlo ground
truth (``do``). numpy only, no torch: Gaussian context and reward noise, Bernoulli actions.

For the learning side of a population — agents that actually update from payoffs — see
:func:`~causalrl.magames.learning.run_no_regret`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from causalrl.data.trajectory import TrajectoryLog
from causalrl.protocols import NoiseLedger
from causalrl.regime import Regime

__all__ = ["LinearGaussianPopulationEnv", "linear_gaussian_population_env"]

FloatArray = NDArray[np.float64]


def _sigmoid(x: FloatArray) -> FloatArray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class LinearGaussianPopulationEnv:
    """The environment one ego faces inside a fixed population: a synthetic DGP, not a learner.

    An observed context ``Z`` confounds the ego's logging action and drives a co-player's action;
    the reward is linear in the ego action, the co-player action and ``Z``, with Gaussian noise. The
    ego's causal action effect ``E[Y | do(ego=1)] - E[Y | do(ego=0)] = ego_effect`` (the co-player
    and context are unaffected by the ego's action) is back-door identified by adjusting for ``Z``.
    Nothing here updates from experience: the fields below are fixed structural coefficients.
    """

    ego: str = "ego"
    coplayer: str = "co"
    ego_effect: float = 1.5
    coplayer_effect: float = 0.8
    context_effect: float = 1.0
    confound: float = 1.0  # Z -> ego logging-action confounding strength
    coplayer_bias: float = 0.7  # Z -> co-player action
    noise: float = 0.5

    def _draw(self, n: int, seed: int | None, ego_action: float | None) -> dict[str, FloatArray]:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal(n)
        if ego_action is None:
            a = rng.binomial(1, _sigmoid(self.confound * z)).astype(np.float64)
        else:
            a = np.full(n, float(ego_action))
        b = rng.binomial(1, _sigmoid(self.coplayer_bias * z)).astype(np.float64)
        y = (
            self.ego_effect * a
            + self.coplayer_effect * b
            + self.context_effect * z
            + self.noise * rng.standard_normal(n)
        )
        return {"Z": z, self.ego: a, self.coplayer: b, "Y": y}

    def _log(self, cols: dict[str, FloatArray], regime_label: str) -> TrajectoryLog:
        kinds = {"Z": "obs", self.ego: "action", self.coplayer: "action", "Y": "reward"}
        rows: list[dict[str, Any]] = []
        for name, vals in cols.items():
            for i, v in enumerate(vals.tolist()):
                rows.append(
                    {
                        "entity_id": i,
                        "episode_id": 0,
                        "t": 0,
                        "kind": kinds[name],
                        "name": name,
                        "value": float(v),
                        "regime": regime_label,
                        "observed": True,
                    }
                )
        return TrajectoryLog.from_rows(rows)

    def sample(
        self, n: int, *, seed: int | None = None, regime: Regime | None = None
    ) -> TrajectoryLog:
        return self._log(self._draw(n, seed, None), "observed")

    def do(
        self,
        interventions: Mapping[str, Any],
        n: int,
        *,
        seed: int | None = None,
        regime: Regime | None = None,
    ) -> TrajectoryLog:
        ego_action = interventions.get(self.ego)
        return self._log(
            self._draw(n, seed, None if ego_action is None else float(ego_action)), "do"
        )

    def noise_ledger(self) -> NoiseLedger | None:
        return None


def linear_gaussian_population_env(
    ego: str = "ego", coplayer: str = "co", **kwargs: float
) -> LinearGaussianPopulationEnv:
    """Construct a :class:`LinearGaussianPopulationEnv` (see it for the DGP coefficients)."""
    return LinearGaussianPopulationEnv(ego=ego, coplayer=coplayer, **kwargs)
