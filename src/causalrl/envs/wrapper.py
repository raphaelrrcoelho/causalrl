"""CausalEnvWrapper: a Gymnasium wrapper that exposes the environment's causal structure.

The wrapper is a thin ``gymnasium.Wrapper`` subclass that adds two groups of causal handles
on top of any :class:`~causalrl.envs.base.CausalEnv` (or any Gymnasium env whose underlying
env carries a ``.scm`` attribute):

**Pure causal-graph queries** (never mutate the running episode):

* **``reward_parents``** — the SCM variables that are direct parents of the reward/return
  node in the underlying causal graph.
* **``intervene(node, value)``** / **``do(interventions)``** — apply a do-intervention on
  the wrapped env's SCM, returning a *new* mutilated copy of the SCM.

**Persistent interventional rollouts** (affect ``step`` / ``reset`` sampling):

* **``set_intervention(interventions)``** — store a persistent intervention; subsequent
  ``reset`` and ``step`` calls run the underlying environment with its SCM temporarily
  swapped to the pre-computed mutilated SCM.  The swap is performed in a ``try/finally``
  that always restores the original SCM.  *Note:* precomputed baselines stored on the env
  (e.g. ``arm_values``) are NOT recomputed — they reflect the unintervened SCM.
* **``clear_intervention()``** — remove the persistent intervention; subsequent calls
  revert to the unintervened SCM.
* **``active_interventions``** — read the currently stored intervention mapping, or
  ``None`` if none is set.

**Graceful ``scm=None`` handling**: construction **succeeds** even when ``env.scm is None``
or ``reward_node`` is not supplied.  In that case the wrapper acts as a plain Gymnasium
pass-through; the causal interface is disabled and ``has_causal_interface`` returns
``False``.  Methods that require the causal interface raise
:class:`~causalrl.exceptions.CausalInterfaceUnavailableError` with an informative message.

Usage::

    from causalrl import CausalEnvWrapper
    from causalrl.envs.suite.scbandit import make_confounded_chain_env

    # Full causal interface:
    env = CausalEnvWrapper(make_confounded_chain_env(), reward_node="Y")
    parents = env.reward_parents          # list[str] of SCM parent node names
    mutilated = env.do({"X": 1.0})        # StructuralCausalModel under do(X=1)
    mutilated2 = env.intervene("X", 1.0)  # same, single-variable convenience

    # Persistent interventional rollout:
    env.set_intervention({"X3": 1.0})
    obs, info = env.reset(seed=0)         # runs under do(X3=1)
    obs, r, *_ = env.step(0)             # reward sampled from mutilated SCM
    env.clear_intervention()              # back to unintervened SCM

    # Pass-through mode (scm=None env, no reward_node):
    from causalrl.envs.suite.gridworld import ConfoundedGridworld
    env2 = CausalEnvWrapper(ConfoundedGridworld(size=3))
    assert not env2.has_causal_interface  # causal methods are disabled
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym

from causalrl.exceptions import CausalInterfaceUnavailableError

if TYPE_CHECKING:
    from causalrl.scm.scm import StructuralCausalModel, Value


class CausalEnvWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """A Gymnasium wrapper that exposes the wrapped environment's causal structure.

    Parameters
    ----------
    env:
        Any ``gymnasium.Env``.  When the env carries a non-``None`` ``.scm`` attribute and
        ``reward_node`` is provided (and valid), the full causal interface is enabled.
        When ``env.scm is None`` **or** ``reward_node`` is ``None``, construction still
        succeeds but the causal interface is disabled (pass-through mode).
    reward_node:
        The SCM variable name that represents the reward / return signal.  Optional —
        ``None`` disables the causal interface.  If the env has a live SCM and
        ``reward_node`` is supplied but not present in the graph, a ``ValueError`` is raised.

    Attributes
    ----------
    reward_node:
        The SCM node treated as the reward/return, or ``None`` when not set.
    has_causal_interface:
        ``True`` iff the wrapped env exposes a non-``None`` SCM *and* ``reward_node`` is set
        and present in the graph.  All causal methods require this to be ``True``.
    reward_parents:
        Direct SCM parents of ``reward_node`` in topological order.  Requires
        ``has_causal_interface``.
    scm:
        The underlying :class:`~causalrl.scm.scm.StructuralCausalModel` (live reference to
        the wrapped env's ``.scm``), or ``None`` when not available.
    active_interventions:
        The currently stored intervention mapping, or ``None`` if no persistent
        intervention is active.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        *,
        reward_node: str | None = None,
    ) -> None:
        super().__init__(env)
        # Validate reward_node only when both the SCM and reward_node are provided.
        raw_scm = getattr(env, "scm", None)
        if reward_node is not None and raw_scm is not None:
            scm = cast("StructuralCausalModel", raw_scm)
            if reward_node not in scm.graph.nodes:
                raise ValueError(
                    f"reward_node {reward_node!r} is not a node in the SCM graph "
                    f"(nodes: {sorted(scm.graph.nodes)})"
                )
        self.reward_node: str | None = reward_node
        # Persistent-intervention state.
        self._active_interventions: Mapping[str, Value] | None = None
        self._mutilated_scm: StructuralCausalModel | None = None

    # ------------------------------------------------------------------
    # Causal interface availability
    # ------------------------------------------------------------------

    @property
    def has_causal_interface(self) -> bool:
        """``True`` iff the causal interface is fully operational.

        Requires the wrapped env to expose a non-``None`` ``.scm`` **and** ``reward_node``
        to be set and present in the graph.
        """
        raw_scm = getattr(self.env, "scm", None)
        return raw_scm is not None and self.reward_node is not None

    def _require_causal_interface(self, method: str) -> None:
        """Raise :class:`~causalrl.exceptions.CausalInterfaceUnavailableError` if disabled."""
        if not self.has_causal_interface:
            raw_scm = getattr(self.env, "scm", None)
            if raw_scm is None:
                reason = (
                    f"the wrapped env {type(self.env).__name__!r} has scm=None "
                    "(confounded MDPs and envs without an explicit SCM do not support "
                    "the causal interface)"
                )
            else:
                reason = "reward_node was not provided at construction time"
            raise CausalInterfaceUnavailableError(
                f"{method} requires has_causal_interface=True, but {reason}. "
                "Construct CausalEnvWrapper with a SCM-backed env and a valid reward_node "
                "to enable the causal interface."
            )

    # ------------------------------------------------------------------
    # Causal interface — pure graph queries
    # ------------------------------------------------------------------

    @property
    def scm(self) -> StructuralCausalModel | None:
        """The underlying SCM (live reference to the wrapped env's ``.scm``), or ``None``."""
        return cast("StructuralCausalModel | None", getattr(self.env, "scm", None))

    @property
    def reward_parents(self) -> list[str]:
        """Direct SCM parents of ``reward_node`` in graph-topological order.

        These are the variables that causally determine the immediate reward signal.  Pass
        their names as the ``factor_nodes`` argument of
        :func:`~causalrl.agents.factored_advantage.factored_advantage`.

        Raises
        ------
        CausalInterfaceUnavailableError
            When ``has_causal_interface`` is ``False``.
        """
        self._require_causal_interface("reward_parents")
        scm = cast("StructuralCausalModel", self.scm)
        reward_node = cast(str, self.reward_node)
        return scm.graph.parents(reward_node)

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

        Raises
        ------
        CausalInterfaceUnavailableError
            When ``has_causal_interface`` is ``False``.
        """
        self._require_causal_interface("do")
        return cast("StructuralCausalModel", self.scm).do(interventions)

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

        Raises
        ------
        CausalInterfaceUnavailableError
            When ``has_causal_interface`` is ``False``.
        """
        return self.do({node: value})

    # ------------------------------------------------------------------
    # Persistent interventional rollouts
    # ------------------------------------------------------------------

    @property
    def active_interventions(self) -> Mapping[str, Value] | None:
        """The currently stored intervention mapping, or ``None`` if none is active."""
        return self._active_interventions

    def set_intervention(self, interventions: Mapping[str, Value]) -> None:
        """Store a persistent intervention that affects subsequent ``reset`` and ``step``.

        After this call, every ``reset()`` and ``step()`` swaps the wrapped env's ``.scm``
        to the pre-computed mutilated SCM for the duration of the call, then restores the
        original SCM in a ``finally`` block.  Precomputed baselines stored on the env
        (e.g. ``arm_values``) are **not** recomputed.

        Parameters
        ----------
        interventions:
            Mapping ``{node_name: value}`` to pin persistently.

        Raises
        ------
        CausalInterfaceUnavailableError
            When ``has_causal_interface`` is ``False``.
        """
        self._require_causal_interface("set_intervention")
        self._active_interventions = dict(interventions)
        # Pre-compute the mutilated SCM once so set_intervention is idempotent-fast.
        self._mutilated_scm = cast("StructuralCausalModel", self.scm).do(interventions)

    def clear_intervention(self) -> None:
        """Remove the persistent intervention; subsequent calls use the unintervened SCM."""
        self._active_interventions = None
        self._mutilated_scm = None

    # ------------------------------------------------------------------
    # Gymnasium pass-through with optional SCM swap
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the wrapped environment, forwarding ``seed`` and ``options``.

        When a persistent intervention is active (``set_intervention`` was called), the
        wrapped env's ``.scm`` is temporarily replaced with the mutilated SCM for the
        duration of this call, then restored unconditionally.
        """
        if self._mutilated_scm is not None:
            orig_scm = cast("StructuralCausalModel", self.scm)
            self.env.scm = self._mutilated_scm  # type: ignore[union-attr]
            try:
                return self.env.reset(seed=seed, options=options)  # type: ignore[return-value]
            finally:
                self.env.scm = orig_scm  # type: ignore[union-attr]
        return self.env.reset(seed=seed, options=options)  # type: ignore[return-value]

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the wrapped environment.

        When a persistent intervention is active (``set_intervention`` was called), the
        wrapped env's ``.scm`` is temporarily replaced with the mutilated SCM for the
        duration of this call, then restored unconditionally.
        """
        if self._mutilated_scm is not None:
            orig_scm = cast("StructuralCausalModel", self.scm)
            self.env.scm = self._mutilated_scm  # type: ignore[union-attr]
            try:
                return self.env.step(action)  # type: ignore[return-value]
            finally:
                self.env.scm = orig_scm  # type: ignore[union-attr]
        return self.env.step(action)  # type: ignore[return-value]
