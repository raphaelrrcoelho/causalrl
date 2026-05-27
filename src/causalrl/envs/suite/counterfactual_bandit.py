"""Three-arm confounded bandit for counterfactual decision-making (taxonomy Task 3).

A hidden confounder ``U in {0,1,2}`` drives both the agent's intent ``I = U`` and the lucky arm
(reward ``0.8`` iff the played arm equals ``U``, else ``0.15``). The behavior policy plays
``X = I``, so it is implicitly optimal — the observational mean is ``0.8`` — yet every fixed
intervention ``do(X = a)`` averages only ``~0.367`` because ``do(X)`` severs ``X``'s dependence on
``U``. A counterfactual-optimal policy that conditions on intent recovers ``0.8``: the MABUC lesson
(Bareinboim, Forney & Pearl, NeurIPS 2015) carried to ``K > 2`` arms.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import torch
from torch.distributions import Categorical, Distribution, Uniform

from causalrl.envs.base import CausalEnv
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_N_ARMS = 3
_P_WIN = 0.8
_P_LOSE = 0.15


def build_counterfactual_scm() -> StructuralCausalModel:
    """SCM ``U->I, U->Y, I->X, X->Y``: ``U ~ Uniform{0,1,2}``; ``I = U``; behavior ``X = I``;
    ``Y ~ Bernoulli(0.8 if X == U else 0.15)``."""
    graph = CausalGraph(directed_edges=[("U", "I"), ("I", "X"), ("U", "Y"), ("X", "Y")])

    def reward(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        win = (pa["X"] == pa["U"]).float()
        p = _P_WIN * win + _P_LOSE * (1.0 - win)
        return (u < p).float()

    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "I": FunctionalMechanism(["U"], lambda pa, u: pa["U"]),
        "X": FunctionalMechanism(["I"], lambda pa, u: pa["I"]),
        "Y": FunctionalMechanism(["X", "U"], reward),
    }
    arm_probs = torch.tensor([1 / 3, 1 / 3, 1 / 3])  # type: ignore[reportPrivateImportUsage]
    exogenous: dict[str, Distribution] = {
        "U": Categorical(probs=arm_probs),
        "I": Uniform(0.0, 1.0),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


class CounterfactualBanditEnv(CausalEnv):
    """One-step three-arm bandit. Observe intent ``intuition = I in {0,1,2}``, play arm
    ``X in {0,1,2}``, receive reward ``Y`` (``0.8`` if ``X == U`` else ``0.15``). The
    counterfactual-optimal policy plays ``X = I``."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(build_counterfactual_scm())
        self.action_space = gym.spaces.Discrete(_N_ARMS)
        self.observation_space = gym.spaces.Dict({"intuition": gym.spaces.Discrete(_N_ARMS)})
        self._rng: torch.Generator = torch.Generator()  # type: ignore[reportPrivateImportUsage]
        if seed is not None:
            self._rng.manual_seed(seed)
        self._u = 0

    def _draw_confounder(self) -> None:
        self._u = int(torch.randint(0, _N_ARMS, (1,), generator=self._rng).item())  # type: ignore[reportPrivateImportUsage]

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.manual_seed(seed)
        self._draw_confounder()
        return {"intuition": self._u}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        p = _P_WIN if action == self._u else _P_LOSE
        reward = float(torch.rand(1, generator=self._rng).item() < p)  # type: ignore[reportPrivateImportUsage]
        return {"intuition": self._u}, reward, True, False, {"u": self._u}


def make_counterfactual_bandit_env(seed: int | None = None) -> CounterfactualBanditEnv:
    """Factory mirroring the other suite environments."""
    return CounterfactualBanditEnv(seed=seed)
