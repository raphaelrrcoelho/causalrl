from typing import Any, ClassVar

import gymnasium as gym
import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.envs.base import CausalEnv
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel

_P_WIN = 0.75
_P_LOSE = 0.25


def build_mabuc_scm() -> StructuralCausalModel:
    """Two-arm MABUC. D,B unobserved confounders; I = D xor B; lucky arm = D xor B;
    Y ~ Bernoulli(0.75 if X == lucky else 0.25); behavior policy plays X = I."""
    graph = CausalGraph(
        directed_edges=[
            ("D", "I"),
            ("B", "I"),
            ("I", "X"),
            ("D", "Y"),
            ("B", "Y"),
            ("X", "Y"),
        ]
    )

    def reward(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        lucky = (pa["D"] != pa["B"]).float()
        win = (pa["X"] == lucky).float()
        p = _P_WIN * win + _P_LOSE * (1.0 - win)
        return (u < p).float()

    mechanisms: dict[str, Mechanism] = {
        "D": FunctionalMechanism([], lambda pa, u: u),
        "B": FunctionalMechanism([], lambda pa, u: u),
        "I": FunctionalMechanism(["D", "B"], lambda pa, u: (pa["D"] != pa["B"]).float()),
        "X": FunctionalMechanism(["I"], lambda pa, u: pa["I"]),
        "Y": FunctionalMechanism(["X", "D", "B"], reward),
    }
    exogenous: dict[str, Distribution] = {
        "D": Bernoulli(0.5),
        "B": Bernoulli(0.5),
        "I": Uniform(0.0, 1.0),
        "X": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    return StructuralCausalModel(graph, mechanisms, exogenous)


class MABUCEnv(CausalEnv):
    """One-step bandit. Each episode: observe intuition I, choose arm X in {0,1},
    receive reward Y. The optimal policy plays X = I (the lucky arm)."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]  # gymnasium Env.metadata is an instance var in the base

    def __init__(self, seed: int | None = None) -> None:
        super().__init__(build_mabuc_scm())
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Dict({"intuition": gym.spaces.Discrete(2)})
        self._rng: torch.Generator = torch.Generator()  # type: ignore[reportPrivateImportUsage]
        if seed is not None:
            self._rng.manual_seed(seed)
        self._d = 0
        self._b = 0

    def _draw_confounders(self) -> None:
        self._d = int(torch.randint(0, 2, (1,), generator=self._rng).item())  # type: ignore[reportPrivateImportUsage]
        self._b = int(torch.randint(0, 2, (1,), generator=self._rng).item())  # type: ignore[reportPrivateImportUsage]

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, int], dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng.manual_seed(seed)
        self._draw_confounders()
        intuition = self._d ^ self._b
        return {"intuition": intuition}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, int], float, bool, bool, dict[str, Any]]:
        lucky = self._d ^ self._b
        p = _P_WIN if action == lucky else _P_LOSE
        reward = float(torch.rand(1, generator=self._rng).item() < p)  # type: ignore[reportPrivateImportUsage]
        intuition = self._d ^ self._b
        return {"intuition": intuition}, reward, True, False, {"lucky": lucky}
