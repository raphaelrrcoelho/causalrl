"""Causal Graph-Factored Advantage (CGFA) primitive.

This module implements the *causal core* of the CGFA-PPO algorithm introduced in:

    Cristiano da Costa Cunha, Ajmal Mian, Tim French, and Wei Liu (2026).
    "Causal Reinforcement Learning for Complex Card Games: A Magic: The Gathering
    Benchmark." arXiv:2605.06066.

The key insight is that, when an SCM defines a causal graph over the state-action-reward
variables, the advantage can be *decomposed* along the causal parents of the return.  Each
parent node ``i`` contributes a **factor advantage** ``A_i = V_i - baseline_i``, and the
full advantage is an aggregation (default: sum) of these per-factor advantages.  This is
analogous to the Generalised Advantage Estimation (GAE) trick applied along the causal
graph rather than along the time axis.

Public API
----------
:func:`factored_advantage`
    Pure-NumPy function — **no RL framework dependency**.  Given per-factor value estimates
    and a common baseline, returns a vector of causal graph-factored advantages.

:class:`FactoredAdvantageConfig`
    Lightweight dataclass that bundles the factor names, aggregation mode, and optional
    per-factor weights for reuse across many calls.

Why this belongs in the library
--------------------------------
The library builds the *novel causal core* and delegates RL training to mature libraries
(stable-baselines3, etc.).  ``factored_advantage`` is precisely the novel causal piece: it
is a deterministic, framework-agnostic numerical primitive that an outer PPO loop calls on
every rollout.  The integration glue (custom rollout buffer, callback) lives in the
``examples/`` directory and depends on stable-baselines3 as an optional extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray


@dataclass
class FactoredAdvantageConfig:
    """Configuration bundle for :func:`factored_advantage`.

    Parameters
    ----------
    factor_nodes:
        Ordered list of SCM parent-node names whose per-factor value estimates are passed to
        :func:`factored_advantage`.  The order must match the columns of the
        ``factor_values`` array.
    aggregation:
        How to combine per-factor advantages into the final scalar advantage.

        * ``"sum"`` (default): ``A = Σ_i A_i`` — the formulation in arXiv:2605.06066.
        * ``"mean"``: ``A = mean_i A_i`` — normalises when the number of factors varies.
    weights:
        Optional per-factor weights ``w_i`` (length must match ``factor_nodes`` when
        provided).  The combined advantage becomes ``A = Σ_i w_i * A_i`` for
        ``aggregation="sum"`` or the weighted mean for ``aggregation="mean"``.  ``None``
        means uniform unit weights.
    """

    factor_nodes: list[str]
    aggregation: Literal["sum", "mean"] = "sum"
    weights: list[float] | None = None
    # derived / validated at post-init
    _weights_arr: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = len(self.factor_nodes)
        if self.weights is not None:
            if len(self.weights) != n:
                raise ValueError(
                    f"weights length ({len(self.weights)}) must match factor_nodes length ({n})"
                )
            self._weights_arr = np.asarray(self.weights, dtype=np.float64)
        else:
            self._weights_arr = np.ones(n, dtype=np.float64)

    @property
    def weights_array(self) -> NDArray[np.float64]:
        """The validated per-factor weight vector (uniform unit weights when unset)."""
        return self._weights_arr


def factored_advantage(
    factor_values: NDArray[np.float64],
    baselines: NDArray[np.float64],
    *,
    config: FactoredAdvantageConfig | None = None,
    aggregation: Literal["sum", "mean"] = "sum",
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute causal graph-factored advantages from per-factor value estimates.

    Implements the SCM-aligned critic target from CGFA-PPO (arXiv:2605.06066, §3.2).
    Given ``K`` causal parent factors of the return, and for each rollout step a vector of
    per-factor value estimates ``V_1, …, V_K`` and a scalar baseline ``b``, the per-factor
    advantage is ``A_i = V_i - b`` and the combined advantage is their (weighted) sum or
    mean.

    When ``K = 1`` (single factor) the output reduces exactly to the standard advantage
    ``A = V - b``, so this is a strict generalisation of the scalar advantage.

    Parameters
    ----------
    factor_values:
        Array of shape ``(T, K)`` where ``T`` is the number of rollout steps and ``K`` is
        the number of causal factors (SCM parents of the return).  Each column ``[:,i]``
        is the critic's value estimate for factor ``i``.
    baselines:
        Array of shape ``(T,)`` — the shared scalar baseline for each step (typically the
        current value-function estimate ``V(s_t)``).
    config:
        A :class:`FactoredAdvantageConfig` that carries ``factor_nodes``, ``aggregation``,
        and optional ``weights``.  When provided, ``aggregation`` and ``weights`` keyword
        arguments are ignored (config takes precedence).
    aggregation:
        Used when ``config`` is ``None``.  ``"sum"`` (default) or ``"mean"``.
    weights:
        Used when ``config`` is ``None``.  Per-factor weights, shape ``(K,)``.  ``None``
        means uniform unit weights.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(T,)`` — the combined causal-graph-factored advantage for each step.

    Raises
    ------
    ValueError
        If ``factor_values`` is not 2-D, if ``baselines`` length does not match ``T``, or
        if ``weights`` shape does not match ``K``.

    Examples
    --------
    Single-factor (reduces to standard advantage):

    >>> import numpy as np
    >>> V = np.array([[2.0], [3.0], [1.0]])   # (T=3, K=1)
    >>> b = np.array([1.5, 2.5, 0.5])
    >>> factored_advantage(V, b)
    array([0.5, 0.5, 0.5])

    Two-factor sum (CGFA-PPO with two SCM parents of the return):

    >>> V2 = np.array([[2.0, 1.0], [3.0, 0.5]])  # (T=2, K=2)
    >>> b2 = np.array([1.5, 2.0])
    >>> factored_advantage(V2, b2)   # A_i = V_i - b; sum over i
    array([ 0. , -0.5])

    References
    ----------
    * Cristiano da Costa Cunha, Ajmal Mian, Tim French, and Wei Liu (2026). "Causal
      Reinforcement Learning for Complex Card Games: A Magic: The Gathering Benchmark."
      arXiv:2605.06066.
    """
    fv = np.asarray(factor_values, dtype=np.float64)
    bl = np.asarray(baselines, dtype=np.float64)

    if fv.ndim != 2:
        raise ValueError(f"factor_values must be 2-D (T, K); got shape {fv.shape}")
    t, k = fv.shape
    if bl.shape != (t,):
        raise ValueError(
            f"baselines must have shape ({t},) to match factor_values rows; got {bl.shape}"
        )

    # Resolve aggregation and weights.
    if config is not None:
        agg = config.aggregation
        w = config.weights_array
        if len(w) != k:
            raise ValueError(
                f"FactoredAdvantageConfig has {len(w)} factors but factor_values has K={k} columns"
            )
    else:
        agg = aggregation
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64)
            if w.shape != (k,):
                raise ValueError(f"weights must have shape ({k},) to match K; got {w.shape}")
        else:
            w = np.ones(k, dtype=np.float64)

    # Per-factor advantages: A_i(t) = V_i(t) - b(t).  Shape (T, K).
    per_factor: NDArray[np.float64] = fv - bl[:, np.newaxis]  # broadcast baseline

    # Combine: weighted sum or weighted mean over the K factors.
    combined = per_factor @ w  # (T,)
    if agg == "mean":
        combined = combined / float(w.sum()) if w.sum() != 0.0 else combined

    return combined
