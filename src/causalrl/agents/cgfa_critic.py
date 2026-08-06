"""The CGFA-PPO ``K``-head factored critic; torch, behind the ``causalrl[torch]`` extra.

Implements the *learnable* half of Causal Graph-Factored Advantage PPO:

    Cristiano da Costa Cunha, Ajmal Mian, Tim French, and Wei Liu (2026).
    "Causal Reinforcement Learning for Complex Card Games: A Magic: The Gathering
    Benchmark." arXiv:2605.06066 (§5, Appendix E, Algorithm 1).

Implemented from the paper only — no reference code was consulted or ported.

Architecture (arXiv:2605.06066 Figure 6 / §E.1)
-----------------------------------------------
A shared critic trunk ``h(s)`` feeding three heads plus one free parameter vector:

* a **scalar critic** ``V(s)``, so CGFA never removes the ordinary PPO value function;
* a **``K``-head factor critic** ``V_k(s) in R^K``, ``k = 1 … K``, emitted from the *same*
  critic features — one head per SCM parent of the return node;
* a **state-conditional residual gate** ``g(s) in (0, 1)`` from a small MLP over ``h(s)``;
* **learnable mixture logits** ``beta in R^K`` with ``w = softmax(beta)``.

Training targets and losses (§E.2, §E.4, §E.5)
-----------------------------------------------
Head ``k`` is regressed on the per-factor return
``G_{k,t} = sum_i gamma^i r^factor_{k,t+i}`` built from the per-factor reward
``r^factor_{k,t} = phi_k(s_{t+1}) - phi_k(s_t)`` (Eq. 8), giving

    L_factor = (1/K) sum_k E_t[(G_{k,t} - V_k(s_t))^2]                        (Eq. 9)

and the per-factor advantage ``A_{k,t} = G_{k,t} - V_k(s_t)`` (Eq. 10), which enters the
policy update only through the residual blend

    A_used = (1 - g(s_t)) A^scalar_t + g(s_t) sum_k w_k A_{k,t}               (Eq. 11)

The SCM enters training through the **intervention-calibration loss**, a negative Pearson
correlation between each factor advantage and the SCM-predicted per-factor change
``eps_{k,t} = phi^SCM_k(s_{t+1}) - phi^SCM_k(s_t)``:

    L_cal = -(1/K) sum_k Cov_t(A_{k,t}, eps_{k,t}) / (sigma_{A_k} sigma_{eps_k} + delta)
                                                                              (Eq. 12)

and the full objective is

    L_CGFA = L_ppo + c_v L_value - c_H L_ent + c_f L_factor + c_c L_cal - c_e L_gate
                                                                              (Eq. 13)

Division of labour
------------------
``L_ppo`` (the masked clipped surrogate) and ``L_ent`` belong to the RL framework, which owns
the actor.  :class:`FactoredCritic` owns everything else and accepts the caller's surrogate
as ``policy_loss`` so the whole of Eq. 13 can be stepped jointly, exactly as Algorithm 1
line 22 does.  The pure-NumPy rollout arithmetic (Eq. 8, 10, 11) lives in
:mod:`causalrl.agents.factored_advantage` and imports no torch.

Deviations and paper ambiguities
--------------------------------
* **Gradient path of ``L_cal``.** §E.2 says the per-factor advantages are "stored in the
  rollout buffer so that the intervention-calibration loss has access to them"; stored
  values carry no gradient, which would make Eq. 12 a pure diagnostic and its ablation a
  no-op.  Since the paper calls it a training signal and ablates it, ``A_{k,t}`` is
  **recomputed** here as ``G_{k,t} - V_k(s_t)`` under the current parameters, so Eq. 12
  back-propagates into the factor heads.
* **Factor GAE ``lambda``.** See :func:`~causalrl.agents.factored_advantage.factor_gae`;
  Eq. 8 / Eq. 10 are Monte-Carlo, the prose says GAE.  Default ``lam=1.0`` reconciles them.
* **Standard-deviation clamp.** §E.4 requires "a clamp on the per-factor standard deviation
  to avoid amplification when a factor has near-zero variance in a minibatch" but gives no
  value; :attr:`CGFACriticConfig.min_std` exposes it (default ``1e-3``).
* **Trunk width.** Table 6's ``MLP [512, 256]`` is sized for a 3,077-dimensional
  observation.  :attr:`CGFACriticConfig.hidden` defaults to ``(64, 64)`` so the critic is
  usable on small problems; the paper's widths are one keyword away.
* **``beta`` initialisation.** The paper initialises ``beta`` from the SCM's
  logistic-regression coefficients on the return node.  Nothing in this library fixes such a
  parameterisation, so ``mixture_init`` is a caller-supplied vector (default: zeros, i.e. a
  uniform mixture).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from causalrl.agents.factored_advantage import blend_advantages, factor_gae

if TYPE_CHECKING:  # pragma: no cover - typing only
    from types import ModuleType

    import torch
    from torch import Tensor

__all__ = [
    "CGFAAdvantages",
    "CGFACriticConfig",
    "CGFALosses",
    "CGFAUpdateStats",
    "FactoredCritic",
]


def _require_torch() -> ModuleType:
    """Import PyTorch, or raise a clear :class:`ImportError` naming the extra.

    Deferring the import to call time is what keeps this module importable — and
    ``causalrl.FactoredCritic`` resolvable — in a torch-free install.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - torch is a dev dependency
        raise ImportError(
            "FactoredCritic requires PyTorch support; install the 'causalrl[torch]' extra "
            "(pip install 'causalrl[torch]'). The pure-NumPy CGFA rollout arithmetic in "
            "causalrl.agents.factored_advantage needs no torch."
        ) from exc
    return torch


@dataclass(frozen=True)
class CGFACriticConfig:
    """Architecture and loss coefficients for :class:`FactoredCritic` (arXiv:2605.06066).

    Defaults follow the paper's Table 6 where it states a value, and are flagged in the
    module docstring where it does not.

    Parameters
    ----------
    hidden:
        Widths of the shared critic trunk.  Table 6 uses ``(512, 256)`` for a
        3,077-dimensional observation; the default here is small enough for toy problems.
    gate_hidden:
        Hidden width of the state-conditional gate MLP (Table 6: 32).
    gate_init:
        Value of ``g(s)`` at initialisation, uniform over states (Table 6: 0.5).  The gate
        head's output layer is zero-initialised with the matching bias, so the gate starts
        state-independent and differentiates as it trains.
    learning_rate:
        Adam learning rate for the critic parameters (Table 6 backbone: ``3e-4``).
    value_coef:
        ``c_v`` — weight on the scalar value loss.  Only active when ``scalar_returns`` are
        supplied to :meth:`FactoredCritic.update`; a caller whose RL framework owns the
        scalar critic leaves them out and this term is exactly zero.
    factor_coef:
        ``c_f`` — weight on the per-factor MSE, Eq. 9 (Table 6: 0.5).
    calibration_coef:
        ``c_c`` — weight on the intervention-calibration loss, Eq. 12 (Table 6: 0.1).
    gate_entropy_coef:
        ``c_e`` — weight on the Bernoulli gate-entropy bonus (Table 6 default: 0.0, swept in
        the ablation).  Eq. 13 *subtracts* it, so a positive value rewards an undecided gate.
    calibration_eps:
        ``delta`` — the stability constant in Eq. 12's denominator.
    min_std:
        Lower clamp on the per-factor standard deviations in Eq. 12 (§E.4 requires a clamp
        but gives no value).
    max_grad_norm:
        Global gradient-norm clip, Algorithm 1 line 22 (Table 6: 0.5).
    """

    hidden: tuple[int, ...] = (64, 64)
    gate_hidden: int = 32
    gate_init: float = 0.5
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    factor_coef: float = 0.5
    calibration_coef: float = 0.1
    gate_entropy_coef: float = 0.0
    calibration_eps: float = 1e-8
    min_std: float = 1e-3
    max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        if not self.hidden:
            raise ValueError("hidden must contain at least one layer width")
        if not 0.0 < self.gate_init < 1.0:
            raise ValueError(f"gate_init must lie strictly in (0, 1); got {self.gate_init}")


@dataclass(frozen=True)
class CGFAAdvantages:
    """One rollout's worth of CGFA advantage arithmetic (Algorithm 1 lines 11-13).

    Attributes
    ----------
    used:
        ``A_used``, shape ``(T,)`` — the advantage to hand to the policy-gradient surrogate.
    scalar:
        The scalar-critic advantage that was blended in, shape ``(T,)``.
    factor:
        Per-factor advantages ``A_{k,t}``, shape ``(T, K)`` (Eq. 10).
    returns:
        Per-factor returns ``G_{k,t}``, shape ``(T, K)`` (Eq. 8) — the regression targets
        :meth:`FactoredCritic.update` expects.
    gate:
        ``g(s_t)``, shape ``(T,)``.
    weights:
        ``w = softmax(beta)``, shape ``(K,)``.
    """

    used: NDArray[np.float64]
    scalar: NDArray[np.float64]
    factor: NDArray[np.float64]
    returns: NDArray[np.float64]
    gate: NDArray[np.float64]
    weights: NDArray[np.float64]


@dataclass(frozen=True)
class CGFALosses:
    """The differentiable Eq. 13 terms :class:`FactoredCritic` owns, as torch scalars.

    Attributes
    ----------
    value:
        ``L_value`` — the scalar critic's MSE (Algorithm 1 line 17).  Exactly zero when the
        caller's RL framework owns the scalar critic and supplies no ``scalar_returns``.
    factor:
        ``L_factor`` — the per-factor MSE of Eq. 9, averaged over the ``K`` heads.
    calibration:
        ``L_cal`` — the negative mean Pearson correlation of Eq. 12.  Exactly zero when no
        SCM-predicted effects are supplied (the paper's "without calibration" ablation).
    gate_entropy:
        ``L_gate`` — the mean Bernoulli entropy of ``g(s_t)`` (Algorithm 1 line 20).
    """

    value: Tensor
    factor: Tensor
    calibration: Tensor
    gate_entropy: Tensor


@dataclass(frozen=True)
class CGFAUpdateStats:
    """Loss values and the per-factor diagnostics of Algorithm 1 line 25.

    Attributes
    ----------
    total:
        Mean of the optimised objective over the update's minibatch steps.
    value, factor, calibration, gate_entropy:
        Mean of each Eq. 13 component, *before* its coefficient.
    mixture_weights:
        ``softmax(beta)`` after the update, shape ``(K,)``.
    factor_correlation:
        Per-factor Pearson correlation ``corr(A_{k,t}, eps_{k,t})`` after the update, shape
        ``(K,)``; all-``nan`` when no SCM effects were supplied.  This is the paper's
        headline calibration diagnostic: it says whether each learned factor advantage moves
        in the same direction as the SCM's predicted intervention effect.
    credit_share:
        ``mean_t |w_k A_{k,t}| / sum_j |w_j A_{j,t}|``, shape ``(K,)`` — which causal factors
        dominate the policy update.
    gate_mean:
        Mean ``g(s_t)`` after the update.  Near 0 means the gate has collapsed to vanilla
        PPO; near 1 means the update rides entirely on the factor-aligned advantage.
    """

    total: float
    value: float
    factor: float
    calibration: float
    gate_entropy: float
    mixture_weights: NDArray[np.float64]
    factor_correlation: NDArray[np.float64]
    credit_share: NDArray[np.float64]
    gate_mean: float


class FactoredCritic:
    """The ``K``-head causal-factor critic of CGFA-PPO (arXiv:2605.06066, §E.1).

    One value head per SCM parent of the return, on a trunk shared with the scalar critic,
    plus the learnable mixture logits and the state-conditional residual gate that turn those
    ``K`` heads into a single advantage for the policy-gradient surrogate.

    The heads only differentiate because they are trained against *different* targets: head
    ``k`` regresses the per-factor return ``G_{k,t}`` accumulated from
    ``r^factor_{k,t} = phi_k(s_{t+1}) - phi_k(s_t)``.  Give every head the same target and
    they collapse to the same function — which is exactly the degenerate "share one value
    head" configuration this class exists to replace.

    Parameters
    ----------
    obs_dim:
        Dimensionality of the (flattened) observation fed to the trunk.
    factor_nodes:
        Ordered SCM node names of the return's causal parents — for a
        :class:`~causalrl.envs.wrapper.CausalEnvWrapper`, ``wrapper.reward_parents``.  Their
        order fixes the column order of every ``(T, K)`` array this class consumes or emits.
    config:
        A :class:`CGFACriticConfig`; ``None`` uses the paper's defaults.
    mixture_init:
        Initial ``beta``, shape ``(K,)``.  ``None`` gives zeros (a uniform mixture).  The
        paper seeds ``beta`` with the SCM's logistic-regression coefficients on the return
        node, so that training starts from the structural prior.
    seed:
        Seed for torch's global RNG before parameter initialisation, for reproducibility.

    Raises
    ------
    ImportError
        At construction, when PyTorch is not installed — naming the ``causalrl[torch]``
        extra.
    ValueError
        On a non-positive ``obs_dim``, an empty ``factor_nodes``, duplicate factor names, or
        a ``mixture_init`` whose length does not match ``factor_nodes``.

    Examples
    --------
    >>> import numpy as np
    >>> critic = FactoredCritic(obs_dim=3, factor_nodes=["X3", "U"], seed=0)
    >>> obs = np.zeros((5, 3))
    >>> scalar, factors = critic.values(obs)
    >>> factors.shape
    (5, 2)
    >>> float(critic.mixture_weights().sum())
    1.0

    References
    ----------
    * Cunha, Mian, French and Liu (2026), arXiv:2605.06066 — §5, Appendix E, Algorithm 1.
    """

    def __init__(
        self,
        obs_dim: int,
        factor_nodes: Sequence[str],
        *,
        config: CGFACriticConfig | None = None,
        mixture_init: NDArray[np.float64] | Sequence[float] | None = None,
        seed: int | None = None,
    ) -> None:
        _require_torch()
        import torch

        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive; got {obs_dim}")
        nodes = list(factor_nodes)
        if not nodes:
            raise ValueError("factor_nodes must name at least one SCM parent of the return")
        if len(set(nodes)) != len(nodes):
            raise ValueError(f"factor_nodes must be unique; got {nodes}")

        self._factor_nodes = nodes
        self._config = config if config is not None else CGFACriticConfig()
        cfg = self._config
        k = len(nodes)

        if seed is not None:
            torch.manual_seed(seed)  # type: ignore[reportPrivateImportUsage]

        if mixture_init is None:
            beta0 = torch.zeros(k)  # type: ignore[reportPrivateImportUsage]
        else:
            beta_arr = np.asarray(mixture_init, dtype=np.float64)
            if beta_arr.shape != (k,):
                raise ValueError(
                    f"mixture_init must have shape ({k},) to match factor_nodes; "
                    f"got {beta_arr.shape}"
                )
            beta0 = torch.as_tensor(beta_arr, dtype=torch.float32)  # type: ignore[reportPrivateImportUsage]

        trunk_layers: list[torch.nn.Module] = []
        width = obs_dim
        for out in cfg.hidden:
            trunk_layers.append(torch.nn.Linear(width, out))  # type: ignore[reportPrivateImportUsage]
            trunk_layers.append(torch.nn.ReLU())  # type: ignore[reportPrivateImportUsage]
            width = out
        self._trunk = torch.nn.Sequential(*trunk_layers)  # type: ignore[reportUnknownMemberType]
        self._scalar_head = torch.nn.Linear(width, 1)  # type: ignore[reportPrivateImportUsage]
        self._factor_head = torch.nn.Linear(width, k)  # type: ignore[reportPrivateImportUsage]

        gate_out = torch.nn.Linear(cfg.gate_hidden, 1)  # type: ignore[reportPrivateImportUsage]
        # Zero weights + logit(gate_init) bias => g(s) == gate_init everywhere at step 0.
        with torch.no_grad():  # type: ignore[reportPrivateImportUsage]
            gate_out.weight.zero_()
            gate_out.bias.fill_(float(np.log(cfg.gate_init / (1.0 - cfg.gate_init))))
        self._gate_head = torch.nn.Sequential(  # type: ignore[reportUnknownMemberType]
            torch.nn.Linear(width, cfg.gate_hidden),  # type: ignore[reportPrivateImportUsage]
            torch.nn.Tanh(),  # type: ignore[reportPrivateImportUsage]
            gate_out,
        )
        self._mixture_logits = torch.nn.Parameter(beta0)  # type: ignore[reportPrivateImportUsage]

        self._module = torch.nn.ModuleList(  # type: ignore[reportUnknownMemberType]
            [
                self._trunk,
                self._scalar_head,
                self._factor_head,
                self._gate_head,
                torch.nn.ParameterList([self._mixture_logits]),  # type: ignore[reportPrivateImportUsage]
            ]
        )
        self._optimizer = torch.optim.Adam(  # type: ignore[reportPrivateImportUsage]
            self._module.parameters(), lr=cfg.learning_rate
        )

    # ------------------------------------------------------------------ properties

    @property
    def factor_nodes(self) -> list[str]:
        """The ordered SCM parent names, one per value head."""
        return list(self._factor_nodes)

    @property
    def n_factors(self) -> int:
        """``K`` — the number of causal factors, hence of value heads."""
        return len(self._factor_nodes)

    @property
    def config(self) -> CGFACriticConfig:
        """The architecture / loss-coefficient bundle this critic was built with."""
        return self._config

    @property
    def module(self) -> torch.nn.Module:
        """The underlying ``torch.nn.Module`` — for ``state_dict``, ``.to(device)``, saving."""
        return self._module

    def parameters(self) -> Iterator[Tensor]:
        """Every learnable tensor: trunk, three heads, and the mixture logits ``beta``.

        Hand these to a joint optimiser alongside the actor's parameters when following
        Algorithm 1 line 22 literally (one Adam step on the whole of Eq. 13).
        """
        return self._module.parameters()  # type: ignore[reportUnknownMemberType,no-any-return]

    # ------------------------------------------------------------------ inference

    def _as_input(self, observations: NDArray[np.float64]) -> Tensor:
        import torch

        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2:
            raise ValueError(f"observations must be 2-D (T, obs_dim); got shape {obs.shape}")
        return torch.as_tensor(obs)  # type: ignore[reportPrivateImportUsage]

    def _heads(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """``(V(s), V_k(s), g(s))`` from the shared trunk (§E.1)."""
        import torch

        features = self._trunk(obs)
        scalar = self._scalar_head(features).squeeze(-1)
        factors = self._factor_head(features)
        gate = torch.sigmoid(self._gate_head(features).squeeze(-1))  # type: ignore[reportPrivateImportUsage]
        return scalar, factors, gate

    def values(
        self, observations: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(V(s_t), V_k(s_t))`` for a batch of observations, shapes ``(T,)`` and ``(T, K)``."""
        import torch

        with torch.no_grad():  # type: ignore[reportPrivateImportUsage]
            scalar, factors, _ = self._heads(self._as_input(observations))
        return (
            scalar.numpy().astype(np.float64),
            factors.numpy().astype(np.float64),
        )

    def gate(self, observations: NDArray[np.float64]) -> NDArray[np.float64]:
        """The state-conditional residual gate ``g(s_t) in (0, 1)``, shape ``(T,)`` (§E.3)."""
        import torch

        with torch.no_grad():  # type: ignore[reportPrivateImportUsage]
            _, _, gate = self._heads(self._as_input(observations))
        return gate.numpy().astype(np.float64)

    def mixture_weights(self) -> NDArray[np.float64]:
        """``w = softmax(beta)``, shape ``(K,)`` — the learned per-factor mixture (§E.1)."""
        import torch

        with torch.no_grad():  # type: ignore[reportPrivateImportUsage]
            w = torch.softmax(self._mixture_logits, dim=0)  # type: ignore[reportPrivateImportUsage]
        return w.numpy().astype(np.float64)

    # ------------------------------------------------------------------ rollout arithmetic

    def advantages(
        self,
        observations: NDArray[np.float64],
        factor_rewards: NDArray[np.float64],
        scalar_advantages: NDArray[np.float64],
        *,
        gamma: float = 0.995,
        lam: float = 1.0,
        bootstrap_values: NDArray[np.float64] | None = None,
        dones: NDArray[np.bool_] | None = None,
        normalize: bool = True,
    ) -> CGFAAdvantages:
        """Algorithm 1 lines 11-13: per-factor GAE, then the gated residual blend.

        Runs the critic over the rollout, computes ``A_{k,t}`` and ``G_{k,t}`` against its own
        factor heads (Eq. 8, Eq. 10), blends them into the caller's scalar advantage with the
        learned gate and mixture weights (Eq. 11), and normalises the result.

        Parameters
        ----------
        observations:
            Rollout observations ``s_0 … s_{T-1}``, shape ``(T, obs_dim)``.
        factor_rewards:
            ``r^factor_{k,t}``, shape ``(T, K)`` —
            :func:`~causalrl.agents.factored_advantage.factor_rewards` of the factor trace.
        scalar_advantages:
            The RL framework's own advantage ``A^scalar_t``, shape ``(T,)``.  Passing the
            framework's GAE (rather than this class's scalar head) is the intended path: the
            paper keeps the host PPO's critic and *adds* to it.
        gamma:
            Discount, shared with the scalar return (Table 6: 0.995).
        lam:
            Per-factor GAE ``lambda``; see
            :func:`~causalrl.agents.factored_advantage.factor_gae` for the paper ambiguity.
        bootstrap_values:
            ``V_k(s_T)`` for a truncated rollout, shape ``(K,)``; ``None`` means the episode
            ended.
        dones:
            Episode-termination flags, shape ``(T,)``.
        normalize:
            Standardise ``A_used`` to zero mean / unit variance across the rollout, as
            Algorithm 1 line 13 does.  Set ``False`` to inspect the raw blend.

        Returns
        -------
        CGFAAdvantages
            ``used`` feeds the surrogate; ``returns`` feeds :meth:`update`.
        """
        v_factor = self.values(observations)[1]
        adv_factor, ret_factor = factor_gae(
            factor_rewards,
            v_factor,
            gamma=gamma,
            lam=lam,
            bootstrap_values=bootstrap_values,
            dones=dones,
        )
        gate = self.gate(observations)
        weights = self.mixture_weights()
        used = blend_advantages(scalar_advantages, adv_factor, gate=gate, weights=weights)
        if normalize:
            std = float(used.std())
            used = (used - float(used.mean())) / (std + 1e-8)
        return CGFAAdvantages(
            used=used,
            scalar=np.asarray(scalar_advantages, dtype=np.float64),
            factor=adv_factor,
            returns=ret_factor,
            gate=gate,
            weights=weights,
        )

    # ------------------------------------------------------------------ losses

    def losses(
        self,
        observations: NDArray[np.float64],
        factor_returns: NDArray[np.float64],
        *,
        scalar_returns: NDArray[np.float64] | None = None,
        scm_effects: NDArray[np.float64] | None = None,
    ) -> CGFALosses:
        """The differentiable Eq. 13 components this class owns, as torch scalars.

        ``L_ppo`` and ``L_ent`` are the RL framework's; pass its surrogate to
        :meth:`objective` to assemble the whole of Eq. 13.

        Parameters
        ----------
        observations:
            Shape ``(T, obs_dim)``.
        factor_returns:
            ``G_{k,t}``, shape ``(T, K)`` — :attr:`CGFAAdvantages.returns`.
        scalar_returns:
            Bootstrapped scalar returns ``R_t``, shape ``(T,)``.  ``None`` (default) zeroes
            ``L_value``, which is correct when the RL framework owns the scalar critic.
        scm_effects:
            ``eps_{k,t}``, the SCM-predicted per-factor change, shape ``(T, K)``.  ``None``
            zeroes ``L_cal`` — the paper's "CGFA without calibration" ablation.

        Raises
        ------
        ValueError
            On any shape mismatch, or if ``factor_returns`` has a column count other than
            :attr:`n_factors`.
        """
        import torch

        obs = self._as_input(observations)
        t_steps = int(obs.shape[0])
        g_target = self._as_matrix(factor_returns, t_steps, "factor_returns")

        scalar, factors, gate = self._heads(obs)

        # Eq. 9 — per-factor MSE, averaged over the K heads.
        factor_loss = ((g_target - factors) ** 2).mean(dim=0).mean()

        # Alg. 1 line 17 — the scalar critic, only when the caller wants us to own it.
        if scalar_returns is None:
            value_loss = torch.zeros((), dtype=scalar.dtype)  # type: ignore[reportPrivateImportUsage]
        else:
            r_target = np.asarray(scalar_returns, dtype=np.float32)
            if r_target.shape != (t_steps,):
                raise ValueError(
                    f"scalar_returns must have shape ({t_steps},); got {r_target.shape}"
                )
            value_loss = ((torch.as_tensor(r_target) - scalar) ** 2).mean()  # type: ignore[reportPrivateImportUsage]

        # Eq. 12 — negative mean Pearson correlation between A_k and the SCM's effect.
        if scm_effects is None:
            calibration = torch.zeros((), dtype=scalar.dtype)  # type: ignore[reportPrivateImportUsage]
        else:
            eps = self._as_matrix(scm_effects, t_steps, "scm_effects")
            # A_k recomputed under the current parameters so Eq. 12 has a gradient path.
            adv = g_target - factors
            calibration = -self._pearson(adv, eps).mean()

        # Alg. 1 line 20 — Bernoulli entropy of the gate, in nats.
        eps_g = 1e-8
        gate_entropy = -(
            gate * torch.log(gate + eps_g)  # type: ignore[reportPrivateImportUsage]
            + (1.0 - gate) * torch.log(1.0 - gate + eps_g)  # type: ignore[reportPrivateImportUsage]
        ).mean()

        return CGFALosses(
            value=value_loss,
            factor=factor_loss,
            calibration=calibration,
            gate_entropy=gate_entropy,
        )

    def blend(
        self,
        observations: NDArray[np.float64],
        scalar_advantages: NDArray[np.float64],
        factor_advantages: NDArray[np.float64],
    ) -> Tensor:
        """Eq. 11 as a **differentiable** torch tensor, shape ``(T,)``.

        ``A_used = (1 - g(s_t)) A^scalar_t + g(s_t) sum_k w_k A_{k,t}``, evaluated under the
        current gate and mixture parameters so gradients reach ``g`` and ``beta``.

        .. note:: **Paper ambiguity, and why this method exists.** Algorithm 1 computes
           ``A_used`` once at line 12 — outside the epoch loop — and stores it; a surrogate
           built on that stored value gives ``g(s)`` and ``beta`` **no gradient at all**, so
           with the Table 6 default ``c_e = 0`` the only two parameters the paper calls
           "learnable" would never move.  Yet §E.3 speaks of the gate collapsing and logs its
           evolution, and Algorithm 1 line 22 updates ``beta``.  The consistent reading is
           that ``A_used`` is recomputed inside the minibatch loop; use this method to build
           the surrogate, and both parameters train.  Feeding
           :attr:`CGFAAdvantages.used` (a detached NumPy array) to the surrogate instead
           reproduces the literal, frozen-gate reading.

        Parameters
        ----------
        observations:
            Shape ``(T, obs_dim)``.
        scalar_advantages:
            ``A^scalar_t``, shape ``(T,)`` — treated as a constant, as PPO always does.
        factor_advantages:
            ``A_{k,t}``, shape ``(T, K)`` — likewise constant; the gradient path here is
            through ``g`` and ``w``, not through the factor critic.
        """
        import torch

        obs = self._as_input(observations)
        t_steps = int(obs.shape[0])
        a_scalar = np.asarray(scalar_advantages, dtype=np.float32)
        if a_scalar.shape != (t_steps,):
            raise ValueError(
                f"scalar_advantages must have shape ({t_steps},); got {a_scalar.shape}"
            )
        a_factor = self._as_matrix(factor_advantages, t_steps, "factor_advantages")
        _, _, gate = self._heads(obs)
        weights = torch.softmax(self._mixture_logits, dim=0)  # type: ignore[reportPrivateImportUsage]
        mixed = a_factor @ weights
        return (1.0 - gate) * torch.as_tensor(a_scalar) + gate * mixed  # type: ignore[reportPrivateImportUsage]

    def objective(
        self,
        losses: CGFALosses,
        *,
        policy_loss: Tensor | None = None,
    ) -> Tensor:
        """Assemble Eq. 13 from :meth:`losses` output and the framework's surrogate.

        ``L_CGFA = policy_loss + c_v L_value + c_f L_factor + c_c L_cal - c_e L_gate``.

        ``policy_loss`` is the caller's ``L_ppo - c_H L_ent`` (the clipped surrogate net of
        its entropy bonus), ideally built on :meth:`blend` so the gate and mixture logits
        receive gradient.  Omit it to optimise only the critic-side terms — a deviation from
        Algorithm 1 line 22, which steps actor and critic together; see the module docstring.
        """
        cfg = self._config
        total = (
            cfg.value_coef * losses.value
            + cfg.factor_coef * losses.factor
            + cfg.calibration_coef * losses.calibration
            - cfg.gate_entropy_coef * losses.gate_entropy
        )
        if policy_loss is not None:
            total = total + policy_loss
        return total

    # ------------------------------------------------------------------ training

    def update(
        self,
        observations: NDArray[np.float64],
        factor_returns: NDArray[np.float64],
        *,
        scalar_returns: NDArray[np.float64] | None = None,
        scm_effects: NDArray[np.float64] | None = None,
        epochs: int = 1,
        batch_size: int | None = None,
        seed: int | None = None,
    ) -> CGFAUpdateStats:
        """Algorithm 1 lines 14-22, restricted to the terms this class owns.

        Each minibatch step evaluates :meth:`losses`, assembles :meth:`objective`, clips the
        global gradient norm to :attr:`CGFACriticConfig.max_grad_norm`, and takes one Adam
        step over the trunk, the three heads, and ``beta``.

        Parameters
        ----------
        observations, factor_returns, scalar_returns, scm_effects:
            As in :meth:`losses`.
        epochs:
            Passes over the rollout (Table 6: 10).
        batch_size:
            Minibatch size (Table 6: 256).  ``None`` uses the whole rollout.
        seed:
            Seed for the minibatch shuffle.

        Returns
        -------
        CGFAUpdateStats
            Mean losses over the steps taken, plus the Algorithm 1 line 25 diagnostics
            (per-factor calibration correlation, credit share, gate distribution) recomputed
            on the full batch after the update.

        Raises
        ------
        ValueError
            If ``epochs`` is not positive, ``batch_size`` is not positive, or any array shape
            disagrees.
        """
        import torch

        if epochs <= 0:
            raise ValueError(f"epochs must be positive; got {epochs}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")

        obs = np.asarray(observations, dtype=np.float64)
        if obs.ndim != 2:
            raise ValueError(f"observations must be 2-D (T, obs_dim); got shape {obs.shape}")
        t_steps = obs.shape[0]
        if t_steps == 0:
            raise ValueError("observations must contain at least one rollout step")
        size = t_steps if batch_size is None else min(batch_size, t_steps)
        rng = np.random.default_rng(seed)
        targets = np.asarray(factor_returns, dtype=np.float64)

        totals = {"total": 0.0, "value": 0.0, "factor": 0.0, "calibration": 0.0, "gate": 0.0}
        n_steps = 0
        for _ in range(epochs):
            order = rng.permutation(t_steps)
            for start in range(0, t_steps, size):
                idx = order[start : start + size]
                parts = self.losses(
                    obs[idx],
                    targets[idx],
                    scalar_returns=(None if scalar_returns is None else scalar_returns[idx]),
                    scm_effects=(None if scm_effects is None else scm_effects[idx]),
                )
                loss = self.objective(parts)
                self._optimizer.zero_grad()
                loss.backward()  # type: ignore[reportUnknownMemberType]
                torch.nn.utils.clip_grad_norm_(  # type: ignore[reportPrivateImportUsage]
                    self._module.parameters(), self._config.max_grad_norm
                )
                self._optimizer.step()  # type: ignore[reportUnknownMemberType]
                totals["total"] += float(loss.item())
                totals["value"] += float(parts.value.item())
                totals["factor"] += float(parts.factor.item())
                totals["calibration"] += float(parts.calibration.item())
                totals["gate"] += float(parts.gate_entropy.item())
                n_steps += 1

        return self._diagnostics(obs, factor_returns, scm_effects, totals, n_steps)

    # ------------------------------------------------------------------ internals

    def _as_matrix(self, array: NDArray[np.float64], t_steps: int, name: str) -> Tensor:
        import torch

        arr = np.asarray(array, dtype=np.float32)
        if arr.shape != (t_steps, self.n_factors):
            raise ValueError(
                f"{name} must have shape ({t_steps}, {self.n_factors}); got {arr.shape}"
            )
        return torch.as_tensor(arr)  # type: ignore[reportPrivateImportUsage]

    def _pearson(self, left: Tensor, right: Tensor) -> Tensor:
        """Column-wise Pearson correlation with the §E.4 standard-deviation clamp."""
        cfg = self._config
        lc = left - left.mean(dim=0, keepdim=True)
        rc = right - right.mean(dim=0, keepdim=True)
        cov = (lc * rc).mean(dim=0)
        std_l = lc.pow(2).mean(dim=0).sqrt().clamp(min=cfg.min_std)
        std_r = rc.pow(2).mean(dim=0).sqrt().clamp(min=cfg.min_std)
        return cov / (std_l * std_r + cfg.calibration_eps)

    def _diagnostics(
        self,
        observations: NDArray[np.float64],
        factor_returns: NDArray[np.float64],
        scm_effects: NDArray[np.float64] | None,
        totals: dict[str, float],
        n_steps: int,
    ) -> CGFAUpdateStats:
        """Algorithm 1 line 25 — calibration correlation, credit share, gate distribution."""
        weights = self.mixture_weights()
        gate = self.gate(observations)
        v_factor = self.values(observations)[1]
        adv = np.asarray(factor_returns, dtype=np.float64) - v_factor

        if scm_effects is None:
            corr = np.full(self.n_factors, np.nan)
        else:
            eps = np.asarray(scm_effects, dtype=np.float64)
            ac = adv - adv.mean(axis=0, keepdims=True)
            ec = eps - eps.mean(axis=0, keepdims=True)
            denom = np.sqrt((ac**2).mean(axis=0)) * np.sqrt((ec**2).mean(axis=0))
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.where(denom > 0.0, (ac * ec).mean(axis=0) / denom, np.nan)

        contrib = np.abs(weights[np.newaxis, :] * adv)
        row_total = contrib.sum(axis=1)
        live = row_total > 0.0
        if bool(live.any()):
            credit_share = (contrib[live] / row_total[live, np.newaxis]).mean(axis=0)
        else:
            credit_share = np.full(self.n_factors, np.nan)

        scale = float(max(n_steps, 1))
        return CGFAUpdateStats(
            total=totals["total"] / scale,
            value=totals["value"] / scale,
            factor=totals["factor"] / scale,
            calibration=totals["calibration"] / scale,
            gate_entropy=totals["gate"] / scale,
            mixture_weights=weights,
            factor_correlation=np.asarray(corr, dtype=np.float64),
            credit_share=np.asarray(credit_share, dtype=np.float64),
            gate_mean=float(gate.mean()),
        )
