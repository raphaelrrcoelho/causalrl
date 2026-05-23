from __future__ import annotations

import gymnasium as gym

from causalrl.scm.scm import StructuralCausalModel


class CausalEnv(gym.Env):  # type: ignore[type-arg]
    """Base class for environments backed by a StructuralCausalModel.

    Subclasses define how SCM samples map to (observation, reward) and which node is the
    action. The SCM is available as `self.scm` for see/do/counterfactual queries.
    """

    def __init__(self, scm: StructuralCausalModel) -> None:
        super().__init__()
        self.scm = scm
