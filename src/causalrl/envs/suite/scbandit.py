import itertools
from typing import Any, ClassVar

import gymnasium as gym
import torch
from torch.distributions import Bernoulli, Distribution, Uniform

from causalrl.envs.base import CausalEnv
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism, Mechanism
from causalrl.scm.scm import StructuralCausalModel


def enumerate_arms(manipulable: list[str], domains: dict[str, list[int]]) -> list[dict[str, int]]:
    """All interventions over `manipulable`: the empty (observational) arm, then every
    (subset, assignment) over the variables' domains. Returns dicts var -> value."""
    arms: list[dict[str, int]] = [{}]
    for size in range(1, len(manipulable) + 1):
        for subset in itertools.combinations(manipulable, size):
            for combo in itertools.product(*(domains[v] for v in subset)):
                arms.append(dict(zip(subset, combo, strict=True)))
    return arms


def build_confounded_chain() -> tuple[
    StructuralCausalModel, CausalGraph, str, list[str], dict[str, list[int]]
]:
    """Demo problem: chain X1->X2->X3->Y with X1 and Y sharing a hidden confounder U.

    Naturally X1=X2=X3=U and Y=[X3==U]=1, so observing (the empty arm) scores ~1.0; any
    intervention that fixes the chain makes Y=[c==U] score ~0.5 (the MABUC effect on a
    chain). POMIS of the abstracted ADMG is {empty, {X3}}.
    """
    scm_graph = CausalGraph(
        directed_edges=[("U", "X1"), ("X1", "X2"), ("X2", "X3"), ("X3", "Y"), ("U", "Y")]
    )
    mechanisms: dict[str, Mechanism] = {
        "U": FunctionalMechanism([], lambda pa, u: u),
        "X1": FunctionalMechanism(["U"], lambda pa, u: pa["U"]),
        "X2": FunctionalMechanism(["X1"], lambda pa, u: pa["X1"]),
        "X3": FunctionalMechanism(["X2"], lambda pa, u: pa["X2"]),
        "Y": FunctionalMechanism(["X3", "U"], lambda pa, u: (pa["X3"] == pa["U"]).float()),
    }
    exogenous: dict[str, Distribution] = {
        "U": Bernoulli(0.5),
        "X1": Uniform(0.0, 1.0),
        "X2": Uniform(0.0, 1.0),
        "X3": Uniform(0.0, 1.0),
        "Y": Uniform(0.0, 1.0),
    }
    scm = StructuralCausalModel(scm_graph, mechanisms, exogenous)
    # Abstracted ADMG (POMIS view): drop the latent U, represent it as X1 <-> Y.
    admg = CausalGraph(
        directed_edges=[("X1", "X2"), ("X2", "X3"), ("X3", "Y")],
        bidirected_edges=[("X1", "Y")],
    )
    return scm, admg, "Y", ["X1", "X2", "X3"], {"X1": [0, 1], "X2": [0, 1], "X3": [0, 1]}


class StructuralCausalBanditEnv(CausalEnv):
    """One-step bandit whose arms are interventions on an SCM. `step` returns a single
    stochastic reward draw under do(arm) (not its expected value); `arm_values` and
    `optimal_value` — Monte-Carlo estimates computed once at construction — are the ground
    truth for regret. `graph` is the abstracted ADMG given to agents; `arms` and
    `manipulable` support POMIS arm pruning."""

    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}  # type: ignore[misc]

    def __init__(
        self,
        scm: StructuralCausalModel,
        admg: CausalGraph,
        reward: str,
        manipulable: list[str],
        domains: dict[str, list[int]],
        *,
        n_mc: int = 2000,
        seed: int | None = None,
    ) -> None:
        super().__init__(scm)
        self.graph = admg
        self.reward = reward
        self.manipulable = manipulable
        self.domains = domains
        self.arms = enumerate_arms(manipulable, domains)
        self.action_space = gym.spaces.Discrete(len(self.arms))
        self.observation_space = gym.spaces.Dict({})
        self._mc_seed = 12345 if seed is None else seed + 12345
        self.arm_values = self._estimate_arm_values(n_mc)
        self.optimal_value = max(self.arm_values)
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]

    def _arm_reward_mean(self, arm: dict[str, int], n: int) -> float:
        model = self.scm.do({k: float(v) for k, v in arm.items()}) if arm else self.scm
        samples = model.see(n)[self.reward]
        return float(samples.mean().item())

    def _estimate_arm_values(self, n_mc: int) -> list[float]:
        torch.manual_seed(self._mc_seed)  # type: ignore[reportUnknownMemberType]
        return [self._arm_reward_mean(arm, n_mc) for arm in self.arms]

    def reset(  # type: ignore[override]
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[reportUnknownMemberType]
        return {}, {}

    def step(  # type: ignore[override]
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        arm = self.arms[action]
        reward = self._arm_reward_mean(arm, 1)
        return {}, reward, True, False, {"optimal_value": self.optimal_value, "arm": arm}


def make_confounded_chain_env(
    seed: int | None = None, *, n_mc: int = 2000
) -> StructuralCausalBanditEnv:
    """The demo `StructuralCausalBanditEnv` over X1->X2->X3->Y with X1<->Y."""
    scm, admg, reward, manipulable, domains = build_confounded_chain()
    return StructuralCausalBanditEnv(scm, admg, reward, manipulable, domains, n_mc=n_mc, seed=seed)


def build_frontdoor() -> tuple[
    StructuralCausalModel, CausalGraph, str, list[str], dict[str, list[int]]
]:
    """The front-door / cholesterol example from Lee & Bareinboim 2019 (R-40, appendix):
    X->Z->Y with a latent confounder between X and Y, where Z (e.g. cholesterol) is
    NON-manipulable and X is the manipulable lever. Under non-manipulability the POMIS is
    {empty, {X}} (do(X) acts through the front-door). Means: observe ~0.50, do(X=0) ~0.44,
    do(X=1) ~0.56 (optimal), so steering through X beats observing."""

    def y_mech(pa: dict[str, torch.Tensor], u: torch.Tensor) -> torch.Tensor:
        return ((u * pa["U_XY"]) != pa["Z"]).float()  # (u_Y AND U_XY) XOR Z

    scm_graph = CausalGraph(directed_edges=[("U_XY", "X"), ("U_XY", "Y"), ("X", "Z"), ("Z", "Y")])
    mechanisms: dict[str, Mechanism] = {
        "U_XY": FunctionalMechanism([], lambda pa, u: u),
        "X": FunctionalMechanism(["U_XY"], lambda pa, u: (u != pa["U_XY"]).float()),
        "Z": FunctionalMechanism(["X"], lambda pa, u: (u != pa["X"]).float()),
        "Y": FunctionalMechanism(["Z", "U_XY"], y_mech),
    }
    exogenous: dict[str, Distribution] = {
        "U_XY": Bernoulli(0.5),
        "X": Bernoulli(0.5),
        "Z": Bernoulli(0.4),
        "Y": Bernoulli(0.4),
    }
    scm = StructuralCausalModel(scm_graph, mechanisms, exogenous)
    # Abstracted ADMG: project out the latent U_XY -> X <-> Y. Z is observed but non-manipulable.
    admg = CausalGraph(directed_edges=[("X", "Z"), ("Z", "Y")], bidirected_edges=[("X", "Y")])
    return scm, admg, "Y", ["X"], {"X": [0, 1]}


def make_frontdoor_env(seed: int | None = None, *, n_mc: int = 20000) -> StructuralCausalBanditEnv:
    """The front-door demo env (R-40): X->Z->Y, X<->Y, with Z non-manipulable (only X is a
    lever). A higher default `n_mc` sharpens the ground-truth means around the ~0.06 gap."""
    scm, admg, reward, manipulable, domains = build_frontdoor()
    return StructuralCausalBanditEnv(scm, admg, reward, manipulable, domains, n_mc=n_mc, seed=seed)
