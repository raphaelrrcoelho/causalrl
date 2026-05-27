from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from causalrl.scm.scm import StructuralCausalModel


class CausalEnv(gym.Env):  # type: ignore[type-arg]
    """Base class for environments backed by a StructuralCausalModel.

    Subclasses define how SCM samples map to (observation, reward) and which node is the
    action. The SCM is available as `self.scm` for see/do/counterfactual queries.
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        super().__init__()
        self.scm = scm


class ConfoundedMDP(CausalEnv):
    """Finite-horizon tabular MDP whose logging policy depends on an unobserved confounder.

    Subclasses set `n_states`, `n_actions`, `horizon`, draw a fresh UC per episode in
    `reset`, and implement step dynamics. `behavior_policy` is the UC-dependent logging
    policy used to generate offline data; it may read the per-episode UC via `self`.

    `scm` is set to None here because confounded MDPs are defined by their dynamics and
    behavior policy rather than an explicit SCM; the attribute is kept for interface
    compatibility with CausalEnv.
    """

    n_states: int
    n_actions: int
    horizon: int

    # Satisfies CausalEnv.scm without requiring a real SCM; subclasses may override.
    scm: StructuralCausalModel | None = None  # type: ignore[assignment]

    def __init__(self) -> None:
        # Do not call super().__init__(scm) — no SCM is required for confounded MDPs.
        gym.Env.__init__(self)  # type: ignore[reportUnknownMemberType]

    @abstractmethod
    def behavior_policy(self, observation: dict[str, int]) -> int:
        """The (confounded) logging policy. May depend on the hidden per-episode UC."""
