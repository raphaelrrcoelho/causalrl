"""Time-unrolled (sequential) Structural Causal Model.

A controlled dynamical system ``state_{t+1} = transition(state_t, action_t, latents, noise_t)``
with one or more *shared* latent nodes (a hidden parameter that acts at every step) is, over a
fixed horizon ``T``, an ordinary DAG: unroll it into nodes ``state_0 .. state_T`` plus the shared
latent node(s), wire each step's mechanism, and you can run the full Pearl ladder on it — in
particular the sequential counterfactual

    abduct(known={state_0, latent, ...}).predict(do={latent: flipped})

i.e. "pin the (deterministic) start + latent, surgically flip the latent, re-roll the trajectory
under the SAME actions". The transition is supplied by the caller as a plain callable, so this
helper stays completely model-agnostic: no domain dynamics live in the library.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

import torch
from torch.distributions import Distribution

from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import FunctionalMechanism
from causalrl.scm.scm import StructuralCausalModel

Tensor = torch.Tensor

# transition(state_t, action_t, latents, noise_t) -> state_{t+1}
#   state_t  : (n, state_dim) tensor          latents : {name: (n, ...) tensor}
#   action_t : the per-step constant (any type) or None
#   noise_t  : (n, ...) per-step exogenous tensor (ignore it for deterministic dynamics)
Transition = Callable[[Tensor, object, Mapping[str, Tensor], Tensor], Tensor]


class _Degenerate(Distribution):
    """Point-mass-at-zero placeholder for the per-step exogenous of a deterministic step.

    Lets every state node satisfy the SCM's "one exogenous distribution per node" contract
    without injecting spurious randomness when the transition is deterministic.
    """

    arg_constraints: ClassVar[dict[str, object]] = {}  # type: ignore
    has_rsample = False

    def sample(self, sample_shape: Sequence[int] = ()) -> Tensor:
        return torch.zeros(tuple(sample_shape))  # type: ignore[reportPrivateImportUsage]


def build_unrolled_scm(
    transition: Transition,
    horizon: int,
    *,
    state0_dist: Distribution,
    latents: Mapping[str, Distribution],
    actions: Sequence[object] | None = None,
    process_noise_dist: Distribution | None = None,
    state_prefix: str = "state",
) -> StructuralCausalModel:
    """Build a time-unrolled :class:`StructuralCausalModel` over ``horizon`` steps.

    Parameters
    ----------
    transition:
        Callable ``(state_t, action_t, latents, noise_t) -> state_{t+1}``. Receives the current
        state tensor ``(n, state_dim)``, the step's action constant, a ``{name: tensor}`` mapping
        of the shared latent values, and the step's per-step exogenous tensor. Deterministic
        transitions simply ignore ``noise_t``.
    horizon:
        Number of transition steps ``T``. The model has states ``{prefix}_0 .. {prefix}_T``.
    state0_dist:
        Exogenous distribution for the initial state ``{prefix}_0``. (For the deterministic
        abduction pattern the actual start is pinned via ``abduct(known=...)``, but the node
        still needs a declared exogenous distribution.)
    latents:
        Shared latent node(s), e.g. ``{"F": Bernoulli(0.5)}``. Each is a parent of *every*
        transition step and is therefore interveneable via ``do`` / pinnable via ``abduct``.
    actions:
        Optional per-step action constants; ``actions[t]`` is passed to the transition that
        produces ``{prefix}_{t+1}``. Must have length ``horizon`` when given. ``None`` passes
        ``None`` as every action.
    process_noise_dist:
        Optional per-step exogenous distribution attached to each produced state node. Defaults
        to a point mass at zero (fully deterministic dynamics).
    state_prefix:
        Node-name prefix for the state chain (default ``"state"``).

    Returns
    -------
    StructuralCausalModel
        With nodes ``{prefix}_0 .. {prefix}_T`` and one node per latent.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if actions is not None and len(actions) != horizon:
        raise ValueError(f"actions must have length horizon={horizon}, got {len(actions)}")
    latent_names = list(latents)
    if not latent_names:
        raise ValueError("at least one shared latent node is required")
    reserved = {f"{state_prefix}_{t}" for t in range(horizon + 1)}
    clash = reserved & set(latent_names)
    if clash:
        raise ValueError(f"latent name(s) collide with state nodes: {sorted(clash)}")

    noise_dist: Distribution = process_noise_dist or _Degenerate()

    def state_name(t: int) -> str:
        return f"{state_prefix}_{t}"

    nodes: list[str] = latent_names + [state_name(t) for t in range(horizon + 1)]
    directed_edges: list[tuple[str, str]] = []
    mechanisms: dict[str, object] = {}
    exogenous: dict[str, Distribution] = {}

    # Shared latents: root nodes whose value IS their exogenous draw (identity mechanism).
    for name in latent_names:
        exogenous[name] = latents[name]
        mechanisms[name] = FunctionalMechanism([], lambda pa, u: u)

    # Initial state: root node, value is its exogenous draw.
    exogenous[state_name(0)] = state0_dist
    mechanisms[state_name(0)] = FunctionalMechanism([], lambda pa, u: u)

    # Transition steps: state_{t+1} = transition(state_t, action_t, latents, noise_t).
    for t in range(horizon):
        cur, nxt = state_name(t), state_name(t + 1)
        parents = [cur, *latent_names]
        directed_edges.append((cur, nxt))
        for name in latent_names:
            directed_edges.append((name, nxt))
        action_t = None if actions is None else actions[t]

        def make_step_fn(
            _cur: str = cur,
            _action: object = action_t,
        ) -> Callable[[dict[str, Tensor], Tensor], Tensor]:
            def step_fn(pa: dict[str, Tensor], u: Tensor) -> Tensor:
                state = pa[_cur]
                latent_vals = {name: pa[name] for name in latent_names}
                return transition(state, _action, latent_vals, u)

            return step_fn

        mechanisms[nxt] = FunctionalMechanism(parents, make_step_fn())
        exogenous[nxt] = noise_dist

    # Every node (latents, state_0..state_T) appears in ``directed_edges`` for horizon >= 1,
    # so edges alone define the node set. Passing ``nodes=`` *and* edges can double-list nodes
    # in some CausalGraph builds, which then fails the SCM's exact-coverage check — so rely on
    # the edges. ``nodes`` is retained only for the assertion below.
    graph = CausalGraph(directed_edges=directed_edges)
    assert set(graph.nodes) == set(nodes), (set(graph.nodes), set(nodes))
    return StructuralCausalModel(graph, mechanisms, exogenous)  # type: ignore[arg-type]
