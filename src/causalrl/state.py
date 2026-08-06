"""State features: the observation-side counterpart to set-valued interventions.

:mod:`causalrl.intervention` gave the *action* side a type richer than an arm index. This gives
the *state* side the same treatment. An agent's tabular interface asks for ``n_states`` and indexes
by ``int``, which forces every problem through a discretisation before it reaches the causal
machinery — even though the estimation core (cross-fitted DML, ANM/neural mechanisms, continuous
bounds) has never needed one.

A :class:`StateEncoder` maps an observation to a feature vector, and everything downstream works in
feature space. The tabular case is not lost, it is *contained*: :class:`OneHotEncoder` encodes a
discrete state index as an indicator vector, and a least-squares learner over those features
reproduces tabular backups exactly — which is the property that makes the generalisation trustworthy
rather than merely more general.

What this module does NOT do is make a *causal bound* continuous. Bounds like the Manski ceiling
used by :class:`causalrl.DOVI` are stated per ``(state, action)`` cell; in feature space there are
no cells, and a bound that holds uniformly over a function class is a different (and weaker) object.
:class:`causalrl.agents.fitted.FittedQIteration` is explicit about that rather than quietly
inheriting the tabular guarantee.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "FeatureTransition",
    "FloatArray",
    "IdentityEncoder",
    "OneHotEncoder",
    "RBFEncoder",
    "StateEncoder",
    "encode_batch",
]


@runtime_checkable
class StateEncoder(Protocol):
    """Maps an observation to a fixed-length feature vector.

    ``dim`` is the length of every vector :meth:`encode` returns; downstream learners size their
    design matrices from it, so it must not vary between calls.
    """

    dim: int

    def encode(self, observation: Mapping[str, Any]) -> FloatArray: ...


class OneHotEncoder:
    """Encode a discrete state index as an indicator vector — the tabular case, contained.

    This is the bridge between the two worlds. Feature-space learning with these features and an
    exactly-solving linear learner reproduces the tabular backup, because the indicator basis spans
    every function on a finite state set. That makes tabular behaviour a *special case* of the
    fitted path rather than a separate code path, and it is what the faithfulness tests check.

    ``key`` names the observation entry holding the state index, matching the ``"state"`` key the
    shipped tabular agents already read.
    """

    def __init__(self, n_states: int, *, key: str = "state") -> None:
        if n_states < 1:
            raise ValueError(f"n_states={n_states} must be at least 1")
        self.dim = n_states
        self.n_states = n_states
        self.key = key

    def encode(self, observation: Mapping[str, Any]) -> FloatArray:
        index = int(observation[self.key])
        if not 0 <= index < self.n_states:
            raise IndexError(
                f"state index {index} is outside [0, {self.n_states}) -- OneHotEncoder was built "
                f"for {self.n_states} states, so this observation comes from a different space."
            )
        vector = np.zeros(self.n_states, dtype=np.float64)
        vector[index] = 1.0
        return vector


class IdentityEncoder:
    """Read named continuous entries out of the observation, in the order given.

    The minimal continuous encoder: no basis expansion, so a linear learner over these features
    fits a linear value function. ``keys`` fixes the column order, which must stay stable for the
    features to mean the same thing across calls.
    """

    def __init__(self, keys: Sequence[str]) -> None:
        if not keys:
            raise ValueError("keys must name at least one observation entry")
        self.keys = tuple(keys)
        self.dim = len(self.keys)

    def encode(self, observation: Mapping[str, Any]) -> FloatArray:
        return np.array([float(observation[key]) for key in self.keys], dtype=np.float64)


class RBFEncoder:
    """Gaussian radial basis features over an inner encoder's output.

    The nonlinear counterpart to :class:`IdentityEncoder`, and the same basis
    :class:`causalrl.FunctionApproxBackdoorAgent` already uses for a continuous confounder — the
    point being that a continuous *state* and a continuous *confounder* want the same machinery,
    which is precisely why the tabular state interface was the odd one out.

    ``centers`` has one row per basis function, each of the inner encoder's dimension. A constant
    column is prepended so a learner can represent an offset without a separate intercept term.
    """

    def __init__(self, inner: StateEncoder, centers: FloatArray, *, bandwidth: float = 1.0) -> None:
        grid = np.asarray(centers, dtype=np.float64)
        if grid.ndim != 2 or grid.shape[1] != inner.dim:
            raise ValueError(
                f"centers must have shape (n_centers, {inner.dim}) to match the inner encoder's "
                f"dimension; got {grid.shape}"
            )
        if bandwidth <= 0.0:
            raise ValueError(f"bandwidth={bandwidth} must be positive")
        self._inner = inner
        self._centers = grid
        self._bandwidth = bandwidth
        self.dim = grid.shape[0] + 1

    def encode(self, observation: Mapping[str, Any]) -> FloatArray:
        raw = self._inner.encode(observation)
        squared = ((raw[None, :] - self._centers) ** 2).sum(axis=1)
        return np.concatenate(
            [np.ones(1, dtype=np.float64), np.exp(-0.5 * squared / self._bandwidth**2)]
        )


def encode_batch(encoder: StateEncoder, observations: Sequence[Mapping[str, Any]]) -> FloatArray:
    """Stack ``encoder.encode`` over ``observations`` into an ``(n, dim)`` design matrix.

    Returns a correctly-shaped empty array for an empty sequence, so a caller assembling a design
    matrix from a possibly-empty slice does not have to special-case it.
    """
    if not observations:
        return np.zeros((0, encoder.dim), dtype=np.float64)
    return np.stack([np.asarray(encoder.encode(obs), dtype=np.float64) for obs in observations])


@dataclass(frozen=True)
class FeatureTransition:
    """A logged transition whose endpoints are feature vectors rather than state indices.

    The feature-space counterpart of :class:`causalrl.data.dataset.Transition`, and the argument
    type of the fitted backup. ``done`` marks an absorbing terminal, whose successor carries no
    future value — the same convention the tabular agents use, kept so that one-hot features
    reproduce their behaviour exactly.
    """

    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool

    def __post_init__(self) -> None:
        state = np.asarray(self.state, dtype=np.float64).reshape(-1)
        next_state = np.asarray(self.next_state, dtype=np.float64).reshape(-1)
        if state.shape != next_state.shape:
            raise ValueError(
                f"state and next_state must share a feature dimension; got {state.shape} and "
                f"{next_state.shape}. Both endpoints must come from the same encoder."
            )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "next_state", next_state)
