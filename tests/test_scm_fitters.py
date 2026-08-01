import numpy as np
import torch

from causalrl.scm.fitters import ANMFit, LinearGaussianFit, TabularCPT


def test_tabular_cpt_root_recovers_the_marginal():
    rng = np.random.default_rng(0)
    child = (rng.random(20_000) < 0.7).astype(int)
    fitted = TabularCPT().fit({}, child)
    u = fitted.noise.sample((40_000,)).reshape(40_000).float()
    drawn = fitted.mechanism({}, u)
    assert abs(float(drawn.mean()) - 0.7) < 0.02
    assert fitted.invertible is False


def test_tabular_cpt_recovers_a_conditional():
    rng = np.random.default_rng(1)
    a = (rng.random(40_000) < 0.5).astype(int)
    # P(Y=1|A=0) = 0.2, P(Y=1|A=1) = 0.9
    y = (rng.random(40_000) < np.where(a == 1, 0.9, 0.2)).astype(int)
    fitted = TabularCPT().fit({"A": a}, y)
    n = 40_000
    u = fitted.noise.sample((n,)).reshape(n).float()
    ones = torch.ones(n)
    zeros = torch.zeros(n)
    assert abs(float(fitted.mechanism({"A": ones}, u).mean()) - 0.9) < 0.02
    assert abs(float(fitted.mechanism({"A": zeros}, u).mean()) - 0.2) < 0.02


def test_tabular_cpt_handles_multiple_parents():
    rng = np.random.default_rng(2)
    x = (rng.random(40_000) < 0.5).astype(int)
    z = (rng.random(40_000) < 0.5).astype(int)
    y = ((x + z) > 0).astype(int)
    fitted = TabularCPT().fit({"X": x, "Z": z}, y)
    n = 20_000
    u = fitted.noise.sample((n,)).reshape(n).float()
    out = fitted.mechanism({"X": torch.zeros(n), "Z": torch.zeros(n)}, u)
    assert float(out.mean()) < 0.02


def test_tabular_cpt_smooths_an_unseen_parent_configuration():
    # A=1 never co-occurs with Z=1 in training; the fitted table must still be a distribution.
    a = np.array([0, 0, 1, 1] * 100)
    z = np.array([0, 1, 0, 0] * 100)
    y = np.array([0, 1, 1, 0] * 100)
    fitted = TabularCPT().fit({"A": a, "Z": z}, y)
    n = 1000
    u = fitted.noise.sample((n,)).reshape(n).float()
    out = fitted.mechanism({"A": torch.ones(n), "Z": torch.ones(n)}, u)
    assert set(np.unique(out.numpy())) <= {0.0, 1.0}


def test_tabular_cpt_conditional_log_likelihood_is_reported():
    rng = np.random.default_rng(3)
    a = (rng.random(5000) < 0.5).astype(int)
    y = (rng.random(5000) < np.where(a == 1, 0.9, 0.1)).astype(int)
    informative = TabularCPT().fit({"A": a}, y)
    uninformative = TabularCPT().fit({}, y)
    assert informative.score > uninformative.score


def test_linear_gaussian_recovers_weights_and_noise_scale():
    rng = np.random.default_rng(4)
    x = rng.normal(size=20_000)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=20_000)
    fitted = LinearGaussianFit().fit({"X": x}, y)
    mech = fitted.mechanism
    assert abs(mech._weights["X"] - 2.0) < 0.05  # type: ignore[attr-defined]
    assert abs(mech._bias - 1.0) < 0.05  # type: ignore[attr-defined]
    assert abs(float(fitted.noise.stddev) - 0.5) < 0.05
    assert fitted.invertible is True
    assert fitted.score > 0.9  # R^2


def test_linear_gaussian_root_is_the_marginal():
    rng = np.random.default_rng(5)
    y = rng.normal(loc=3.0, scale=2.0, size=20_000)
    fitted = LinearGaussianFit().fit({}, y)
    n = 20_000
    u = fitted.noise.sample((n,)).reshape(n).float()
    drawn = fitted.mechanism({}, u)
    assert abs(float(drawn.mean()) - 3.0) < 0.1
    assert abs(float(drawn.std()) - 2.0) < 0.1


def test_anm_fits_a_nonlinear_mean_with_the_default_ridge():
    rng = np.random.default_rng(6)
    x = rng.uniform(-2.0, 2.0, size=8000)
    y = np.sin(3.0 * x) + rng.normal(scale=0.1, size=8000)
    fitted = ANMFit().fit({"X": x}, y)
    grid = torch.linspace(-1.5, 1.5, 64)
    predicted = fitted.mechanism({"X": grid}, torch.zeros(64))
    truth = torch.sin(3.0 * grid)
    assert float((predicted - truth).abs().mean()) < 0.15
    assert fitted.invertible is True


def test_anm_accepts_a_duck_typed_estimator_factory():
    class ConstantModel:
        def fit(self, x, y):
            self.value = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.value)

    rng = np.random.default_rng(7)
    y = rng.normal(loc=5.0, size=2000)
    fitted = ANMFit(estimator=ConstantModel).fit({"X": rng.normal(size=2000)}, y)
    out = fitted.mechanism({"X": torch.zeros(4)}, torch.zeros(4))
    assert abs(float(out.mean()) - 5.0) < 0.15
