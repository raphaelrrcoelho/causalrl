"""Causal Graph-Factored Advantage (CGFA) rollout arithmetic.

This module holds the **pure-NumPy, framework-agnostic** half of the CGFA-PPO algorithm of:

    Cristiano da Costa Cunha, Ajmal Mian, Tim French, and Wei Liu (2026).
    "Causal Reinforcement Learning for Complex Card Games: A Magic: The Gathering
    Benchmark." arXiv:2605.06066.

The key insight is that, when an SCM defines a causal graph over the state-action-reward
variables, the advantage can be *decomposed* along the causal parents of the return.  Each
parent node ``k`` contributes a **factor advantage** ``A_k``, and the advantage actually fed
to the policy-gradient surrogate is a gated mixture of the scalar advantage and the weighted
sum of the per-factor advantages.

The learnable half — the ``K``-head critic ``V_k(s)``, the learnable mixture logits ``beta``,
the state-conditional gate ``g(s)``, and the per-factor / intervention-calibration losses —
needs parameters and an optimiser, so it lives next door in
:mod:`causalrl.agents.cgfa_critic` behind the ``causalrl[torch]`` extra.  **Nothing in this
module imports torch**, and that is deliberate: the arithmetic below is callable from any RL
framework (or none).

Public API
----------
:func:`factor_rewards`
    ``r^factor_{k,t} = phi_k(s_{t+1}) - phi_k(s_t)`` — the per-factor reward published by the
    CGFA environment wrapper (arXiv:2605.06066 §E.1).

:func:`factor_gae`
    Per-factor returns ``G_{k,t}`` and advantages ``A_{k,t} = G_{k,t} - V_k(s_t)``
    (arXiv:2605.06066 Eq. 8 and Eq. 10), computed by the standard GAE recursion so that a
    truncated rollout bootstraps correctly.

:func:`blend_advantages`
    The state-conditional residual blend
    ``A_used = (1 - g) A_scalar + g * sum_k w_k A_k`` (arXiv:2605.06066 Eq. 11).

:func:`factored_advantage`
    The original decomposition primitive: given per-factor value estimates and a **common
    scalar baseline**, return their (weighted) sum or mean.  Kept unchanged.  Note this is
    *not* Eq. 10 — Eq. 10 subtracts the **per-factor** value ``V_k(s_t)`` from the per-factor
    return ``G_{k,t}``, which is what :func:`factor_gae` computes.

:class:`FactoredAdvantageConfig`
    Lightweight dataclass that bundles the factor names, aggregation mode, and optional
    per-factor weights for reuse across many calls.

Why this belongs in the library
--------------------------------
The library builds the *novel causal core* and delegates RL training to mature libraries
(stable-baselines3, etc.).  The functions here are deterministic numerical primitives an
outer PPO loop calls on every rollout; :class:`~causalrl.agents.cgfa_critic.FactoredCritic`
supplies the per-factor value estimates they consume.  The integration glue (custom rollout
buffer, callback) lives in the ``examples/`` directory and depends on stable-baselines3 as an
optional extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FactoredAdvantageConfig",
    "blend_advantages",
    "factor_gae",
    "factor_rewards",
    "factored_advantage",
]


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
    """Combine per-factor value estimates against a common baseline into one advantage.

    Given ``K`` causal parent factors of the return, and for each rollout step a vector of
    per-factor value estimates ``V_1, …, V_K`` and a **shared scalar** baseline ``b``, the
    per-factor advantage is ``A_i = V_i - b`` and the combined advantage is their (weighted)
    sum or mean.

    When ``K = 1`` (single factor) the output reduces exactly to the standard advantage
    ``A = V - b``, so this is a strict generalisation of the scalar advantage.

    Relation to CGFA-PPO
    --------------------
    This is a decomposition primitive, **not** Eq. 10 of arXiv:2605.06066.  Eq. 10 is
    ``A_{k,t} = G_{k,t} - V_k(s_t)``: the per-factor *return* minus the *per-factor* value,
    i.e. a per-factor baseline of shape ``(T, K)``, which this signature cannot express (its
    ``baselines`` is ``(T,)``).  Use :func:`factor_gae` for Eq. 10 and
    :func:`blend_advantages` for the Eq. 11 residual blend; both are consumed by
    :class:`~causalrl.agents.cgfa_critic.FactoredCritic`.

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


def factor_rewards(factor_trace: NDArray[np.float64]) -> NDArray[np.float64]:
    """Per-factor rewards ``r^factor_{k,t} = phi_k(s_{t+1}) - phi_k(s_t)``.

    This is the per-step quantity the CGFA environment wrapper publishes in
    arXiv:2605.06066 §E.1: the *change* in each SCM factor across a transition.  The factors
    ``phi(s)`` are the SCM parents of the return node — for a
    :class:`~causalrl.envs.wrapper.CausalEnvWrapper` those are ``wrapper.reward_parents``.

    Parameters
    ----------
    factor_trace:
        Array of shape ``(T + 1, K)``: the factor values ``phi(s_0), …, phi(s_T)`` observed
        along a rollout of ``T`` transitions.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(T, K)`` — the first difference along the time axis.

    Raises
    ------
    ValueError
        If ``factor_trace`` is not 2-D or has fewer than two rows.

    Examples
    --------
    >>> import numpy as np
    >>> phi = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 4.0]])  # (T+1=3, K=2)
    >>> factor_rewards(phi)
    array([[1., 0.],
           [0., 3.]])

    References
    ----------
    * Cunha, Mian, French and Liu (2026), arXiv:2605.06066, §E.1.
    """
    trace = np.asarray(factor_trace, dtype=np.float64)
    if trace.ndim != 2:
        raise ValueError(f"factor_trace must be 2-D (T+1, K); got shape {trace.shape}")
    if trace.shape[0] < 2:
        raise ValueError(
            f"factor_trace needs at least 2 rows (phi(s_0) and phi(s_1)); got {trace.shape[0]}"
        )
    return np.diff(trace, axis=0)


def factor_gae(
    rewards: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    gamma: float,
    lam: float = 1.0,
    bootstrap_values: NDArray[np.float64] | None = None,
    dones: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-factor advantages and returns for the CGFA-PPO factor critic.

    Runs the standard generalised-advantage recursion **independently per factor**::

        delta_{k,t} = r^factor_{k,t} + gamma * V_k(s_{t+1}) * (1 - done_t) - V_k(s_t)
        A_{k,t}     = delta_{k,t} + gamma * lam * (1 - done_t) * A_{k,t+1}
        G_{k,t}     = A_{k,t} + V_k(s_t)

    At ``lam=1.0`` with no bootstrap this collapses exactly to the paper's written
    equations: ``G_{k,t} = sum_i gamma^i r^factor_{k,t+i}`` (Eq. 8) and
    ``A_{k,t} = G_{k,t} - V_k(s_t)`` (Eq. 10).

    .. note:: **Paper ambiguity.** arXiv:2605.06066 writes Eq. 8 / Eq. 10 as a
       Monte-Carlo return and its residual, but the surrounding prose (§5, §E.2) says the
       per-factor advantage is computed "using the same generalised-advantage truncation as
       the scalar critic", and Table 6 lists a single GAE ``lambda = 0.95`` for the shared
       backbone.  Those two readings coincide only at ``lambda = 1``.  ``lam`` therefore
       defaults to ``1.0``, which reproduces the paper's *explicit equations*; pass
       ``lam=0.95`` for the prose/Table-6 reading.

    Parameters
    ----------
    rewards:
        Per-factor rewards, shape ``(T, K)`` — typically :func:`factor_rewards` output.
    values:
        Per-factor critic estimates ``V_k(s_t)``, shape ``(T, K)`` — typically
        :meth:`~causalrl.agents.cgfa_critic.FactoredCritic.values`.
    gamma:
        Discount factor, shared with the scalar return (paper default ``0.995``).
    lam:
        GAE ``lambda``.  ``1.0`` (default) reproduces Eq. 8 / Eq. 10 exactly.
    bootstrap_values:
        ``V_k(s_T)`` for the state after the last stored transition, shape ``(K,)``.
        ``None`` (default) means zero — an episode that ended at ``T``.
    dones:
        Boolean episode-termination flags, shape ``(T,)``.  ``dones[t]`` true cuts the
        bootstrap and the recursion at step ``t``.  ``None`` means no interior terminations.

    Returns
    -------
    tuple[NDArray[np.float64], NDArray[np.float64]]
        ``(advantages, returns)``, both of shape ``(T, K)``.

    Raises
    ------
    ValueError
        On any shape mismatch between ``rewards``, ``values``, ``bootstrap_values`` and
        ``dones``, or if ``rewards`` is not 2-D.

    Examples
    --------
    Two factors that accumulate at different rates, with a zero critic so the advantage is
    the raw discounted return (Eq. 8):

    >>> import numpy as np
    >>> r = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> V = np.zeros((2, 2))
    >>> adv, ret = factor_gae(r, V, gamma=0.5)
    >>> ret
    array([[1. , 0.5],
           [0. , 1. ]])

    References
    ----------
    * Cunha, Mian, French and Liu (2026), arXiv:2605.06066, Eq. 8, Eq. 10, §E.2.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"rewards must be 2-D (T, K); got shape {r.shape}")
    if v.shape != r.shape:
        raise ValueError(f"values must have the same shape as rewards {r.shape}; got {v.shape}")
    t_steps, k = r.shape

    if bootstrap_values is None:
        boot = np.zeros(k, dtype=np.float64)
    else:
        boot = np.asarray(bootstrap_values, dtype=np.float64)
        if boot.shape != (k,):
            raise ValueError(f"bootstrap_values must have shape ({k},); got {boot.shape}")

    if dones is None:
        not_done = np.ones(t_steps, dtype=np.float64)
    else:
        d = np.asarray(dones)
        if d.shape != (t_steps,):
            raise ValueError(f"dones must have shape ({t_steps},); got {d.shape}")
        not_done = 1.0 - d.astype(np.float64)

    advantages = np.zeros_like(r)
    carry = np.zeros(k, dtype=np.float64)
    next_values = boot
    for t in range(t_steps - 1, -1, -1):
        mask = not_done[t]
        delta = r[t] + gamma * next_values * mask - v[t]
        carry = delta + gamma * lam * mask * carry
        advantages[t] = carry
        next_values = v[t]
    return advantages, advantages + v


def blend_advantages(
    scalar_advantages: NDArray[np.float64],
    factor_advantages: NDArray[np.float64],
    *,
    gate: NDArray[np.float64] | float,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """The CGFA-PPO state-conditional residual blend (arXiv:2605.06066, Eq. 11).

    ``A_used(s_t, a_t) = (1 - g(s_t)) A^scalar(s_t, a_t) + g(s_t) * sum_k w_k A_k(s_t, a_t)``

    ``g -> 0`` recovers vanilla PPO exactly; ``g -> 1`` hands the policy update entirely to
    the factor-aligned signal.  In the paper ``g`` is a learned state-conditional MLP and
    ``w = softmax(beta)`` are learned mixture logits initialised from the SCM's
    logistic-regression coefficients on the return node; both are supplied by
    :class:`~causalrl.agents.cgfa_critic.FactoredCritic`.  Passing constants here is a valid
    ablation (the paper's "CGFA without the gate" row).

    Parameters
    ----------
    scalar_advantages:
        The ordinary (scalar-critic) advantage, shape ``(T,)``.
    factor_advantages:
        Per-factor advantages ``A_{k,t}``, shape ``(T, K)`` — :func:`factor_gae` output.
    gate:
        ``g(s_t)`` in ``(0, 1)``: shape ``(T,)`` or a scalar broadcast over the rollout.
    weights:
        Mixture weights ``w_k``, shape ``(K,)``.  ``None`` means uniform ``1/K``.  The paper
        uses ``softmax(beta)``, which sums to 1; this function does not renormalise, so an
        unnormalised vector is passed through as given.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(T,)`` — the advantage to feed to the policy-gradient surrogate.

    Raises
    ------
    ValueError
        On any shape mismatch, or if ``gate`` falls outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> a_scalar = np.array([1.0, 1.0])
    >>> a_factor = np.array([[4.0, 0.0], [4.0, 0.0]])
    >>> blend_advantages(a_scalar, a_factor, gate=0.0)          # pure PPO
    array([1., 1.])
    >>> blend_advantages(a_scalar, a_factor, gate=1.0)          # pure factored, uniform w
    array([2., 2.])
    >>> blend_advantages(a_scalar, a_factor, gate=0.5)          # half and half
    array([1.5, 1.5])

    References
    ----------
    * Cunha, Mian, French and Liu (2026), arXiv:2605.06066, Eq. 11, §E.3.
    """
    a_scalar = np.asarray(scalar_advantages, dtype=np.float64)
    a_factor = np.asarray(factor_advantages, dtype=np.float64)
    if a_factor.ndim != 2:
        raise ValueError(f"factor_advantages must be 2-D (T, K); got shape {a_factor.shape}")
    t_steps, k = a_factor.shape
    if a_scalar.shape != (t_steps,):
        raise ValueError(
            f"scalar_advantages must have shape ({t_steps},) to match factor_advantages rows; "
            f"got {a_scalar.shape}"
        )

    g = np.asarray(gate, dtype=np.float64)
    if g.ndim == 0:
        g = np.full(t_steps, float(g))
    elif g.shape != (t_steps,):
        raise ValueError(f"gate must be a scalar or have shape ({t_steps},); got {g.shape}")
    if bool(np.any(g < 0.0)) or bool(np.any(g > 1.0)):
        raise ValueError("gate must lie in [0, 1] — it is a residual mixing coefficient")

    if weights is None:
        w = np.full(k, 1.0 / k, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (k,):
            raise ValueError(f"weights must have shape ({k},) to match K; got {w.shape}")

    return (1.0 - g) * a_scalar + g * (a_factor @ w)
