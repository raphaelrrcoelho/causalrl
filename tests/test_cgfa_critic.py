"""Tests for the CGFA-PPO K-head factored critic (arXiv:2605.06066, Appendix E).

The load-bearing assertions are the ones that pin **head differentiation**.  A critic whose
``K`` heads all learn the same function is exactly the degenerate configuration this module
replaces, and it passes any test that merely checks the critic runs and returns finite
numbers.  Every test here is written so that tying the heads together makes it fail.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from causalrl.agents.cgfa_critic import CGFACriticConfig, FactoredCritic

# ---------------------------------------------------------------------------
# Fixtures: a two-factor problem where the factors depend on different coordinates
# ---------------------------------------------------------------------------

FACTORS = ["X3", "U"]  # two SCM parents of the return, as CausalEnvWrapper reports them


def _split_dataset(n: int = 256, seed: int = 0) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Observations plus per-factor returns driven by *different* observation coordinates.

    ``G_0 = 3 * x_0`` and ``G_1 = -2 * x_1``: uncorrelated targets of opposite sign, so a
    single shared value head cannot fit both.
    """
    rng = np.random.default_rng(seed)
    obs = rng.uniform(-1.0, 1.0, size=(n, 2))
    returns = np.stack([3.0 * obs[:, 0], -2.0 * obs[:, 1]], axis=1)
    return obs, returns


def _column_pearson(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    lc = left - left.mean(axis=0, keepdims=True)
    rc = right - right.mean(axis=0, keepdims=True)
    denom = np.sqrt((lc**2).mean(axis=0)) * np.sqrt((rc**2).mean(axis=0))
    return (lc * rc).mean(axis=0) / denom


@pytest.fixture(scope="module")
def trained() -> tuple[FactoredCritic, NDArray[np.float64], NDArray[np.float64]]:
    """A critic trained on the split targets with the Eq. 9 per-factor MSE alone."""
    obs, returns = _split_dataset()
    critic = FactoredCritic(
        obs_dim=2,
        factor_nodes=FACTORS,
        config=CGFACriticConfig(
            hidden=(32, 32), learning_rate=1e-2, factor_coef=1.0, calibration_coef=0.0
        ),
        seed=0,
    )
    critic.update(obs, returns, epochs=400)
    return critic, obs, returns


# ---------------------------------------------------------------------------
# THE assertion: the K heads converge to DIFFERENT functions
# ---------------------------------------------------------------------------


def test_each_head_fits_its_own_factor_return_far_better_than_its_neighbour_s(
    trained: tuple[FactoredCritic, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Head k tracks G_k and not G_j — the property that makes 'K-head critic' true.

    Tying the heads (one shared output broadcast to K columns) makes own- and cross-MSE
    equal by construction and this ratio collapses to 1.
    """
    critic, obs, returns = trained
    _, values = critic.values(obs)
    own = np.array([float(((values[:, k] - returns[:, k]) ** 2).mean()) for k in range(2)])
    cross = np.array([float(((values[:, k] - returns[:, 1 - k]) ** 2).mean()) for k in range(2)])
    assert np.all(own < 0.05), f"heads did not fit their own targets: {own}"
    assert np.all(cross > 20.0 * own), f"heads are interchangeable: own={own} cross={cross}"


def test_heads_respond_to_different_observation_coordinates(
    trained: tuple[FactoredCritic, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """Sweeping x_0 moves head 0 and leaves head 1 flat, and vice versa."""
    critic, _, _ = trained
    sweep = np.linspace(-1.0, 1.0, 21)
    probe_first = np.stack([sweep, np.zeros_like(sweep)], axis=1)
    probe_second = np.stack([np.zeros_like(sweep), sweep], axis=1)
    _, v_first = critic.values(probe_first)
    _, v_second = critic.values(probe_second)

    def span(col: NDArray[np.float64]) -> float:
        return float(col.max() - col.min())

    assert span(v_first[:, 0]) > 5.0 * span(v_first[:, 1])
    assert span(v_second[:, 1]) > 5.0 * span(v_second[:, 0])


def test_head_outputs_are_not_the_same_function(
    trained: tuple[FactoredCritic, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """The two heads disagree by more than the residual they each carry."""
    critic, obs, _ = trained
    _, values = critic.values(obs)
    assert float(np.abs(values[:, 0] - values[:, 1]).max()) > 2.0
    assert abs(float(_column_pearson(values[:, :1], values[:, 1:])[0])) < 0.5


def test_per_factor_advantages_are_credited_to_the_right_head(
    trained: tuple[FactoredCritic, NDArray[np.float64], NDArray[np.float64]],
) -> None:
    """A_k = G_k - V_k vanishes for a fitted head, and explodes when the columns are swapped.

    With identical heads the swapped advantages are a permutation of the originals, so their
    mean absolute value is *unchanged* and the ratio below is exactly 1.
    """
    critic, obs, returns = trained
    zero_scalar = np.zeros(obs.shape[0])
    matched = critic.advantages(obs, returns, zero_scalar, gamma=0.0, normalize=False)
    swapped = critic.advantages(obs, returns[:, ::-1], zero_scalar, gamma=0.0, normalize=False)
    matched_residual = float(np.abs(matched.factor).mean())
    swapped_residual = float(np.abs(swapped.factor).mean())
    assert matched_residual < 0.2
    assert swapped_residual > 10.0 * matched_residual


def test_credit_share_diagnostic_tracks_the_dominant_factor() -> None:
    """Algorithm 1 line 25: the factor with the larger residual takes the larger share."""
    obs, returns = _split_dataset(n=64, seed=9)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    # Give factor 0 a wildly wrong return so its advantage dominates the update.
    skewed = returns.copy()
    skewed[:, 0] += 50.0
    stats = critic.update(obs, skewed, epochs=1)
    assert stats.credit_share[0] > 0.9
    assert stats.credit_share.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# The intervention-calibration loss (Eq. 12) actually trains
# ---------------------------------------------------------------------------


def _calibration_setup(
    coef: float,
) -> tuple[FactoredCritic, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(3)
    obs = rng.uniform(-1.0, 1.0, size=(256, 2))
    returns = np.zeros((256, 2))
    effects = np.stack([obs[:, 0], obs[:, 1]], axis=1)
    critic = FactoredCritic(
        obs_dim=2,
        factor_nodes=FACTORS,
        config=CGFACriticConfig(
            hidden=(32, 32), learning_rate=1e-2, factor_coef=0.0, calibration_coef=coef
        ),
        seed=1,
    )
    return critic, obs, returns, effects


def test_calibration_loss_aligns_factor_advantages_with_the_scm_effect() -> None:
    """Eq. 12 raises corr(A_k, eps_k) per factor — the paper's structural training signal."""
    critic, obs, returns, effects = _calibration_setup(coef=1.0)
    before = _column_pearson(returns - critic.values(obs)[1], effects)
    stats = critic.update(obs, returns, scm_effects=effects, epochs=300)
    # Every factor ends aligned, none regresses, and the worst-aligned one moves a long way.
    # (At this seed factor 1 starts anti-aligned at about -0.75 and is flipped to +1.)
    assert np.all(stats.factor_correlation > 0.9), stats.factor_correlation
    assert np.all(stats.factor_correlation >= before - 1e-6)
    assert float(stats.factor_correlation.min() - before.min()) > 0.5


def test_zero_calibration_coefficient_is_the_ablation_control() -> None:
    """c_c = 0 ('CGFA without calibration') must leave the alignment exactly untouched."""
    critic, obs, returns, effects = _calibration_setup(coef=0.0)
    before = _column_pearson(returns - critic.values(obs)[1], effects)
    stats = critic.update(obs, returns, scm_effects=effects, epochs=300)
    np.testing.assert_allclose(stats.factor_correlation, before, atol=1e-6)


def test_calibration_is_exactly_zero_without_scm_effects() -> None:
    """No eps supplied => L_cal contributes nothing, not a silently wrong number."""
    obs, returns = _split_dataset(n=32)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    losses = critic.losses(obs, returns)
    assert float(losses.calibration.item()) == 0.0
    assert np.all(np.isnan(critic.update(obs, returns).factor_correlation))


def test_calibration_correlation_is_reported_per_factor() -> None:
    """The diagnostic separates factors: aligned eps for one, anti-aligned for the other."""
    rng = np.random.default_rng(7)
    obs = rng.uniform(-1.0, 1.0, size=(128, 2))
    returns = np.zeros((128, 2))
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=2)
    advantage = returns - critic.values(obs)[1]
    effects = np.stack([advantage[:, 0], -advantage[:, 1]], axis=1)
    stats = critic.update(obs, returns, scm_effects=effects, epochs=1, batch_size=None, seed=0)
    assert stats.factor_correlation[0] > 0.5
    assert stats.factor_correlation[1] < -0.5


# ---------------------------------------------------------------------------
# The gate and the mixture logits are learnable — through the differentiable blend
# ---------------------------------------------------------------------------


def test_gate_starts_state_independent_at_its_configured_value() -> None:
    """g(s) == gate_init for every state at step 0 (zero-weight output layer + logit bias)."""
    critic = FactoredCritic(
        obs_dim=3,
        factor_nodes=FACTORS,
        config=CGFACriticConfig(gate_init=0.25),
        seed=0,
    )
    gate = critic.gate(np.random.default_rng(0).standard_normal((16, 3)))
    np.testing.assert_allclose(gate, 0.25, atol=1e-6)


def test_gate_and_mixture_train_through_the_differentiable_blend() -> None:
    """A surrogate built on blend() moves both g(s) and beta; Eq. 11's parameters are real.

    Factor 0 carries a positive advantage and factor 1 a negative one, so a surrogate that
    prefers a larger A_used must shift w toward factor 0 and open the gate.
    """
    rng = np.random.default_rng(11)
    obs = rng.uniform(-1.0, 1.0, size=(64, 2))
    scalar = np.zeros(64)
    factor = np.stack([np.ones(64), -np.ones(64)], axis=1)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    opt = torch.optim.Adam(critic.parameters(), lr=0.05)
    for _ in range(200):
        loss = -critic.blend(obs, scalar, factor).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert critic.mixture_weights()[0] > 0.9
    assert float(critic.gate(obs).mean()) > 0.9


def test_critic_only_update_leaves_the_gate_and_mixture_frozen() -> None:
    """Algorithm 1 read literally: with c_e = 0 and no surrogate, g and beta get no gradient.

    This is the documented consequence of the paper's Alg. 1 line 12 ordering, and the reason
    :meth:`FactoredCritic.blend` exists.
    """
    obs, returns = _split_dataset(n=64)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    stats = critic.update(obs, returns, epochs=20)
    assert stats.gate_mean == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(stats.mixture_weights, [0.5, 0.5], atol=1e-7)


def test_gate_entropy_bonus_pushes_the_gate_off_a_collapsed_value() -> None:
    """A positive c_e rewards an undecided gate, so a near-0 gate drifts back up."""
    obs, returns = _split_dataset(n=64)
    critic = FactoredCritic(
        obs_dim=2,
        factor_nodes=FACTORS,
        config=CGFACriticConfig(
            gate_init=0.02, learning_rate=5e-2, factor_coef=0.0, gate_entropy_coef=1.0
        ),
        seed=0,
    )
    stats = critic.update(obs, returns, epochs=200)
    assert stats.gate_mean > 0.3


def test_mixture_init_seeds_the_structural_prior() -> None:
    """beta is caller-seeded (the paper uses the SCM's logistic-regression coefficients)."""
    critic = FactoredCritic(
        obs_dim=2, factor_nodes=FACTORS, mixture_init=np.array([2.0, 0.0]), seed=0
    )
    weights = critic.mixture_weights()
    np.testing.assert_allclose(weights, [0.880797, 0.119203], atol=1e-5)
    assert float(weights.sum()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rollout wiring: advantages() is Algorithm 1 lines 11-13
# ---------------------------------------------------------------------------


def test_advantages_reproduce_the_eq_11_blend_from_their_own_reported_parts() -> None:
    """used == (1-g) A_scalar + g * sum_k w_k A_k, recomputed from the returned fields."""
    from causalrl.agents.factored_advantage import blend_advantages

    rng = np.random.default_rng(5)
    obs = rng.uniform(-1.0, 1.0, size=(12, 2))
    rewards = rng.standard_normal((12, 2))
    scalar = rng.standard_normal(12)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    bundle = critic.advantages(obs, rewards, scalar, gamma=0.9, normalize=False)
    expected = blend_advantages(scalar, bundle.factor, gate=bundle.gate, weights=bundle.weights)
    np.testing.assert_allclose(bundle.used, expected, atol=1e-10)
    np.testing.assert_allclose(bundle.returns - bundle.factor, critic.values(obs)[1], atol=1e-6)


def test_advantages_normalise_the_rollout_by_default() -> None:
    """Algorithm 1 line 13 standardises A_used before the PPO update."""
    rng = np.random.default_rng(6)
    obs = rng.uniform(-1.0, 1.0, size=(64, 2))
    rewards = rng.standard_normal((64, 2))
    scalar = rng.standard_normal(64) * 5.0 + 3.0
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    used = critic.advantages(obs, rewards, scalar, gamma=0.9).used
    assert float(used.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(used.std()) == pytest.approx(1.0, abs=1e-3)


def test_bootstrap_values_flow_through_to_the_factor_returns() -> None:
    """A truncated rollout picks up gamma * V_k(s_T) per factor."""
    obs = np.zeros((1, 2))
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    plain = critic.advantages(obs, np.zeros((1, 2)), np.zeros(1), gamma=0.5, normalize=False)
    booted = critic.advantages(
        obs,
        np.zeros((1, 2)),
        np.zeros(1),
        gamma=0.5,
        bootstrap_values=np.array([4.0, 8.0]),
        normalize=False,
    )
    np.testing.assert_allclose(booted.returns - plain.returns, [[2.0, 4.0]], atol=1e-5)


# ---------------------------------------------------------------------------
# Losses and the Eq. 13 assembly
# ---------------------------------------------------------------------------


def test_scalar_value_loss_is_zero_unless_the_caller_hands_over_the_scalar_critic() -> None:
    """The RL framework usually owns V(s); omitting scalar_returns must cost exactly nothing."""
    obs, returns = _split_dataset(n=32)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    assert float(critic.losses(obs, returns).value.item()) == 0.0
    with_scalar = critic.losses(obs, returns, scalar_returns=np.full(32, 10.0))
    assert float(with_scalar.value.item()) > 1.0


def test_objective_is_the_weighted_sum_of_equation_13() -> None:
    """L_CGFA = policy + c_v L_value + c_f L_factor + c_c L_cal - c_e L_gate."""
    obs, returns = _split_dataset(n=32)
    effects = np.stack([obs[:, 0], obs[:, 1]], axis=1)
    cfg = CGFACriticConfig(
        value_coef=0.3, factor_coef=0.7, calibration_coef=0.2, gate_entropy_coef=0.4
    )
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, config=cfg, seed=0)
    losses = critic.losses(obs, returns, scalar_returns=np.zeros(32), scm_effects=effects)
    total = critic.objective(losses)
    expected = (
        0.3 * float(losses.value.item())
        + 0.7 * float(losses.factor.item())
        + 0.2 * float(losses.calibration.item())
        - 0.4 * float(losses.gate_entropy.item())
    )
    assert float(total.item()) == pytest.approx(expected, rel=1e-5)


def test_objective_adds_the_frameworks_surrogate() -> None:
    """policy_loss is the caller's L_ppo - c_H L_ent; Eq. 13 adds it verbatim."""
    obs, returns = _split_dataset(n=16)
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    losses = critic.losses(obs, returns)
    without = float(critic.objective(losses).item())
    with_policy = float(critic.objective(losses, policy_loss=torch.tensor(2.5)).item())
    assert with_policy - without == pytest.approx(2.5, rel=1e-5)


def test_update_reports_a_falling_factor_loss() -> None:
    """Consecutive updates must reduce the Eq. 9 residual — the heads are actually fitting."""
    obs, returns = _split_dataset(n=128, seed=4)
    critic = FactoredCritic(
        obs_dim=2,
        factor_nodes=FACTORS,
        config=CGFACriticConfig(hidden=(32, 32), learning_rate=1e-2),
        seed=0,
    )
    first = critic.update(obs, returns, epochs=5)
    later = critic.update(obs, returns, epochs=200)
    assert later.factor < 0.25 * first.factor


def test_minibatching_splits_the_rollout() -> None:
    """batch_size splits the rollout into the minibatches of Algorithm 1 line 15."""
    obs, returns = _split_dataset(n=64, seed=8)
    critic = FactoredCritic(
        obs_dim=2,
        factor_nodes=FACTORS,
        config=CGFACriticConfig(hidden=(16,), learning_rate=1e-2),
        seed=0,
    )
    before = critic.update(obs, returns, epochs=1, batch_size=64, seed=0).factor
    after = critic.update(obs, returns, epochs=1, batch_size=8, seed=0).factor
    # Eight gradient steps beat one on the same data.
    assert after < before
    assert np.isfinite(after)


# ---------------------------------------------------------------------------
# Surface, validation and the torch boundary
# ---------------------------------------------------------------------------


def test_factor_nodes_and_arity_are_exposed() -> None:
    critic = FactoredCritic(obs_dim=4, factor_nodes=FACTORS, seed=0)
    assert critic.factor_nodes == FACTORS
    assert critic.n_factors == 2
    assert critic.config.gate_init == 0.5
    assert isinstance(critic.module, torch.nn.Module)
    assert len(list(critic.parameters())) > 0


def test_every_reported_parameter_is_trainable() -> None:
    """parameters() is what a joint optimiser gets — Algorithm 1 line 22 updates (theta, beta)."""
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    params = list(critic.parameters())
    assert params
    assert all(p.requires_grad for p in params)


def test_factor_nodes_wire_to_the_scm_reward_parents() -> None:
    """The head order is the SCM parent order the environment wrapper publishes."""
    from causalrl.envs.suite.scbandit import make_confounded_chain_env
    from causalrl.envs.wrapper import CausalEnvWrapper

    env = CausalEnvWrapper(make_confounded_chain_env(n_mc=10, seed=0), reward_node="Y")
    critic = FactoredCritic(obs_dim=3, factor_nodes=env.reward_parents, seed=0)
    assert critic.factor_nodes == env.reward_parents
    assert critic.n_factors == len(env.reward_parents)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"obs_dim": 0, "factor_nodes": ["A"]}, "obs_dim must be positive"),
        ({"obs_dim": 2, "factor_nodes": []}, "at least one SCM parent"),
        ({"obs_dim": 2, "factor_nodes": ["A", "A"]}, "must be unique"),
    ],
)
def test_construction_validates_its_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        FactoredCritic(**kwargs)  # type: ignore[arg-type]


def test_mixture_init_length_must_match_the_factor_count() -> None:
    with pytest.raises(ValueError, match="mixture_init must have shape"):
        FactoredCritic(obs_dim=2, factor_nodes=FACTORS, mixture_init=np.zeros(3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hidden": ()}, "at least one layer width"),
        ({"gate_init": 0.0}, "gate_init must lie strictly"),
        ({"gate_init": 1.0}, "gate_init must lie strictly"),
    ],
)
def test_config_validates_its_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CGFACriticConfig(**kwargs)  # type: ignore[arg-type]


def test_rejects_a_non_2d_observation_batch() -> None:
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    with pytest.raises(ValueError, match="observations must be 2-D"):
        critic.values(np.zeros(2))
    with pytest.raises(ValueError, match="observations must be 2-D"):
        critic.update(np.zeros(2), np.zeros((1, 2)))


def test_rejects_mismatched_target_shapes() -> None:
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    obs = np.zeros((4, 2))
    with pytest.raises(ValueError, match="factor_returns must have shape"):
        critic.losses(obs, np.zeros((4, 3)))
    with pytest.raises(ValueError, match="scalar_returns must have shape"):
        critic.losses(obs, np.zeros((4, 2)), scalar_returns=np.zeros(3))
    with pytest.raises(ValueError, match="scm_effects must have shape"):
        critic.losses(obs, np.zeros((4, 2)), scm_effects=np.zeros((3, 2)))
    with pytest.raises(ValueError, match="scalar_advantages must have shape"):
        critic.blend(obs, np.zeros(3), np.zeros((4, 2)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epochs": 0}, "epochs must be positive"),
        ({"batch_size": 0}, "batch_size must be positive"),
    ],
)
def test_update_validates_its_schedule(kwargs: dict[str, object], message: str) -> None:
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    with pytest.raises(ValueError, match=message):
        critic.update(np.zeros((4, 2)), np.zeros((4, 2)), **kwargs)  # type: ignore[arg-type]


def test_update_rejects_an_empty_rollout() -> None:
    critic = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=0)
    with pytest.raises(ValueError, match="at least one rollout step"):
        critic.update(np.zeros((0, 2)), np.zeros((0, 2)))


def test_seeding_makes_initialisation_reproducible() -> None:
    obs = np.random.default_rng(0).standard_normal((8, 2))
    first = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=42).values(obs)[1]
    second = FactoredCritic(obs_dim=2, factor_nodes=FACTORS, seed=42).values(obs)[1]
    np.testing.assert_allclose(first, second)


def test_critic_is_exported_top_level_and_resolves_without_torch() -> None:
    """`causalrl.FactoredCritic` must resolve in a torch-free install; only use needs torch."""
    import causalrl

    for name in (
        "FactoredCritic",
        "CGFACriticConfig",
        "CGFALosses",
        "CGFAAdvantages",
        "CGFAUpdateStats",
        "factor_rewards",
        "factor_gae",
        "blend_advantages",
    ):
        assert name in causalrl.__all__, f"{name} missing from causalrl.__all__"
        assert getattr(causalrl, name) is not None

    source = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_torch

import causalrl
from causalrl.agents.cgfa_critic import CGFACriticConfig, FactoredCritic

# Names resolve and the pure-config dataclass works with no torch anywhere.
assert causalrl.FactoredCritic is FactoredCritic
assert CGFACriticConfig().factor_coef == 0.5

try:
    FactoredCritic(obs_dim=2, factor_nodes=["A"])
except ImportError as exc:
    assert "causalrl[torch]" in str(exc), str(exc)
else:                                      # pragma: no cover - defensive
    raise AssertionError("constructing FactoredCritic without torch must raise ImportError")
"""
    result = subprocess.run(
        [sys.executable, "-c", source], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
