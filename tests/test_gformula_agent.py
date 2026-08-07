"""GFormulaBackdoorAgent: multivariate standardization recovers the ATE where naive is biased."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.agents.mbrl import GFormulaBackdoorAgent


def _confounded_continuous(seed: int, tau: float = 2.0) -> tuple[dict[str, np.ndarray], float]:
    # Continuous confounders X drive BOTH treatment and outcome, so the naive contrast is biased.
    rng = np.random.default_rng(seed)
    n = 4000
    x = rng.normal(0.0, 1.0, (n, 2))
    prob = 1.0 / (1.0 + np.exp(-(1.2 * x[:, 0] - 0.8 * x[:, 1])))
    a = (rng.random(n) < prob).astype(int)
    y = tau * a + 3.0 * x[:, 0] - 1.5 * x[:, 1] + rng.normal(0.0, 1.0, n)
    return {"A": a, "Y": y, "X0": x[:, 0], "X1": x[:, 1]}, tau


def test_recovers_ate_where_naive_is_biased() -> None:
    data, tau = _confounded_continuous(0)
    agent = GFormulaBackdoorAgent(2, covariates=("X0", "X1"))
    agent.fit(data)
    a, y = data["A"], data["Y"]
    naive = float(y[a == 1].mean() - y[a == 0].mean())
    assert abs(agent.contrast - tau) < 0.25  # g-formula recovers the true ATE
    assert abs(naive - tau) > 1.0  # the naive contrast is badly confounded
    assert abs(agent.contrast - tau) < abs(naive - tau)  # and g-formula is far closer


def test_decision_flip_under_confounding() -> None:
    # Action 1 is truly better (tau > 0), but it is played when X is low, and low X depresses Y —
    # so the naive comparison ships action 0. G-formula adjusts for X and recovers action 1.
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(0.0, 1.0, n)
    a = (rng.random(n) < 1.0 / (1.0 + np.exp(1.5 * x))).astype(int)  # A=1 when X is low
    y = 1.0 * a + 2.0 * x + rng.normal(0.0, 1.0, n)
    data = {"A": a, "Y": y, "X": x}
    agent = GFormulaBackdoorAgent(2, covariates=("X",))
    agent.fit(data)
    assert y[a == 1].mean() < y[a == 0].mean()  # naive is fooled into action 0
    assert agent.act({}) == 1  # g-formula's MARGINAL decision recovers the truly-better action 1
    assert agent.contrast > 0.0


class _OLS:
    """A minimal sklearn-style outcome model (numpy OLS) exercising the factory hook."""

    def fit(self, x: np.ndarray, y: np.ndarray) -> _OLS:
        design = np.hstack([np.ones((len(x), 1)), x])
        self.weights = np.linalg.lstsq(design, y, rcond=None)[0]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((len(x), 1)), x]) @ self.weights


def test_outcome_model_factory_hook() -> None:
    data, tau = _confounded_continuous(0)
    agent = GFormulaBackdoorAgent(2, covariates=("X0", "X1"), outcome_model=_OLS)
    agent.fit(data)
    assert abs(agent.contrast - tau) < 0.3


def test_contrast_requires_binary_treatment() -> None:
    agent = GFormulaBackdoorAgent(3, covariates=("X",))
    with pytest.raises(ValueError):
        _ = agent.contrast


def test_cate_recovers_heterogeneous_effect() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(0.0, 1.0, (n, 2))
    a = (rng.random(n) < 1.0 / (1.0 + np.exp(-x[:, 0]))).astype(int)  # confounded
    tau = 1.0 + x[:, 1]  # the individual effect varies with x1
    y = tau * a + 2.0 * x[:, 0] + rng.normal(0.0, 0.5, n)
    data = {"A": a, "Y": y, "X0": x[:, 0], "X1": x[:, 1]}
    agent = GFormulaBackdoorAgent(2, covariates=("X0", "X1")).fit(data)
    cate = agent.cate(data)
    assert cate.shape == (n,)
    assert np.corrcoef(cate, tau)[0, 1] > 0.9  # tracks the true heterogeneity
    assert abs(float(cate.mean()) - agent.contrast) < 1e-6  # averages to the ATE


def _heterogeneous_confounded(seed: int = 0, n: int = 6000) -> dict[str, np.ndarray]:
    """Confounded logs whose per-unit effect ``tau(x) = 1 + 3*x1`` flips sign at ``x1 = -1/3``.

    The ATE is +1, so ONE action wins on average; the sign flip means a constant policy is wrong
    for every unit below the threshold.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n, 2))
    a = (rng.random(n) < 1.0 / (1.0 + np.exp(-x[:, 0]))).astype(int)  # confounded by x0
    y = (1.0 + 3.0 * x[:, 1]) * a + 2.0 * x[:, 0] + rng.normal(0.0, 0.5, n)
    return {"A": a, "Y": y, "X0": x[:, 0], "X1": x[:, 1]}


def test_act_is_the_cate_sign_not_a_constant() -> None:
    # The discriminating test for act(): two observations whose CATE signs differ must get
    # DIFFERENT actions. Mutating act() to `return int(self._best_action)` fails this.
    data = _heterogeneous_confounded()
    agent = GFormulaBackdoorAgent(2, covariates=("X0", "X1")).fit(data)
    assert agent.contrast > 0.0  # the ATE -- and so the best CONSTANT action -- is 1

    assert agent.act({"X0": 0.0, "X1": 1.5}) == 1  # tau ~ +5.5
    assert agent.act({"X0": 0.0, "X1": -1.5}) == 0  # tau ~ -3.5: the constant policy is wrong here
    assert agent.act({}) == 1  # no covariates supplied -> the marginal decision

    # act is exactly the sign of the per-unit CATE the T-learner already estimates -- the same
    # CATE-to-policy conversion the library performs for a third-party EconML estimator.
    cate = agent.cate(data)
    rows = [
        {"X0": float(x0), "X1": float(x1)}
        for x0, x1 in zip(data["X0"][:300], data["X1"][:300], strict=True)
    ]
    assert [agent.act(row) for row in rows] == (cate[:300] > 0.0).astype(int).tolist()


def test_act_rejects_a_partial_covariate_vector() -> None:
    agent = GFormulaBackdoorAgent(2, covariates=("X0", "X1")).fit(_heterogeneous_confounded())
    with pytest.raises(KeyError):
        agent.act({"X0": 0.0})  # silently marginalizing X1 would answer a different query


def test_act_before_fit_is_the_default_not_a_crash() -> None:
    agent = GFormulaBackdoorAgent(2, covariates=("X0", "X1"))
    assert agent.act({"X0": 0.0, "X1": 1.5}) == 0


def test_cate_requires_binary_and_both_arms() -> None:
    with pytest.raises(ValueError):
        GFormulaBackdoorAgent(3, covariates=("X",)).cate(
            {"A": np.array([0, 1]), "Y": np.array([0.0, 1.0]), "X": np.array([0.0, 1.0])}
        )
    one_arm = {
        "A": np.array([1, 1, 1]),
        "Y": np.array([1.0, 0.0, 1.0]),
        "X": np.array([0.0, 1.0, 2.0]),
    }
    with pytest.raises(ValueError):
        GFormulaBackdoorAgent(2, covariates=("X",)).cate(one_arm)
