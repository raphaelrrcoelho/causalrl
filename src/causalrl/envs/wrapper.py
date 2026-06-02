"""CausalEnvWrapper: a Gymnasium wrapper that exposes the environment's causal structure.

The wrapper is a thin ``gymnasium.Wrapper`` subclass that adds two causal handles on top of
any :class:`~causalrl.envs.base.CausalEnv` (or any Gymnasium env whose underlying env
carries a ``.scm`` attribute):

* **``reward_parents``** — the SCM variables that are direct parents of the reward/return node
  in the underlying causal graph.
* **``intervene(node, value)``** / **``do(interventions)``** — apply a do-intervention on the
  wrapped env's SCM, returning a *new* mutilated copy of the SCM (without modifying the
  environment's live SCM).

These are pure causal-graph / SCM queries; they do not mutate the running episode. The
wrapper is framework-agnostic — no RL training library is imported.

Usage::

    from causalrl import CausalEnvWrapper, StructuralCausalBanditEnv
    env = CausalEnvWrapper(make_some_causal_env(), reward_node="Y")
    parents = env.reward_parents          # list[str] of SCM parent node names
    mutilated = env.do({"X": 1.0})        # StructuralCausalModel under do(X=1)
    mutilated2 = env.intervene("X", 1.0)  # same, single-variable convenience
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import gymnasium as gym

if TYPE_CHECKING:
    from causalrl.scm.scm import StructuralCausalModel, Value


class CausalEnvWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """A Gymnasium wrapper that exposes the wrapped environment's causal structure.

    Parameters
    ----------
    env:
        Any ``gymnasium.Env`` that carries a ``.scm`` attribute of type
        :class:`~causalrl.scm.scm.StructuralCausalModel`.  All
        :class:`~causalrl.envs.base.CausalEnv` subclasses satisfy this.
    reward_node:
        The SCM variable name that represents the reward / return signal.  Its
        graph parents are exposed via :attr:`reward_parents`.

    Attributes
    ----------
    reward_node:
        The SCM node treated as the reward/return.
    reward_parents:
        Direct SCM parents of ``reward_node`` in topological order.
    scm:
        The underlying :class:`~causalrl.scm.scm.StructuralCausalModel` (read-only proxy
        to the wrapped env's ``.scm``).
    """

    def __init__(self, env: gym.Env[Any, Any], *, reward_node: str) -> None:
        super().__init__(env)
        if not hasattr(env, "scm") or env.scm is None:  # type: ignore[union-attr]
            raise ValueError(
                "CausalEnvWrapper requires the wrapped env to expose a non-None .scm "
                f"attribute; got {type(env).__name__!r} with "
                f"scm={getattr(env, 'scm', '<missing>')!r}"
            )
        self.reward_node: str = reward_node
        scm = self.scm  # typed via the .scm property's return annotation
        if reward_node not in scm.graph.nodes:
            raise ValueError(
                f"reward_node {reward_node!r} is not a node in the SCM graph "
                f"(nodes: {sorted(scm.graph.nodes)})"
            )

    # ------------------------------------------------------------------
    # Causal interface
    # ------------------------------------------------------------------

    @property
    def scm(self) -> StructuralCausalModel:
        """The underlying SCM (live reference to the wrapped env's ``.scm``)."""
        return self.env.scm  # type: ignore[union-attr, return-value]

    @property
    def reward_parents(self) -> list[str]:
        """Direct SCM parents of ``reward_node`` in graph-topological order.

        These are the variables that causally determine the immediate reward signal.  Pass
        their names as the ``factor_nodes`` argument of
        :func:`~causalrl.agents.factored_advantage.factored_advantage`.
        """
        return self.scm.graph.parents(self.reward_node)

    def do(self, interventions: Mapping[str, Value]) -> StructuralCausalModel:
        """Return a *new* SCM mutilated by ``do(interventions)``.

        The running environment's SCM is NOT modified.  This is a pure causal-graph query
        suitable for off-policy reasoning or shaping.

        Parameters
        ----------
        interventions:
            Mapping ``{node_name: value}`` passed to
            :meth:`~causalrl.scm.scm.StructuralCausalModel.do`.

        Returns
        -------
        StructuralCausalModel
            The mutilated SCM under the specified do-intervention.
        """
        return self.scm.do(interventions)

    def intervene(self, node: str, value: Value) -> StructuralCausalModel:
        """Convenience wrapper: ``do({node: value})``.

        Parameters
        ----------
        node:
            The SCM variable to intervene on.
        value:
            The value to pin it to (scalar, sequence, or Tensor).

        Returns
        -------
        StructuralCausalModel
            The mutilated SCM under ``do({node: value})``.
        """
        return self.do({node: value})

    # ------------------------------------------------------------------
    # Gymnasium pass-through (reset / step / render inherit from Wrapper)
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the wrapped environment, forwarding ``seed`` and ``options``."""
        return self.env.reset(seed=seed, options=options)  # type: ignore[return-value]

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the wrapped environment."""
        return self.env.step(action)  # type: ignore[return-value]
