import numpy as np
import pytest
import torch

from causalrl.scm.fitters import ANMFit, LinearGaussianFit, NeuralFit, PoissonGLMFit, TabularCPT


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


def test_tabular_cpt_refuses_a_table_too_large_to_be_a_conditional_distribution():
    # I5 regression: each parent is discretised by its distinct observed values, so a CONTINUOUS
    # parent contributes one level per row and the table's row count (the product of parent
    # cardinalities) explodes -- 640 x 640 = 409,600 rows here. Left unguarded this either
    # exhausts memory or, where it fits, becomes a nearest-neighbour memoriser of the training
    # rows dressed as a conditional distribution.
    rng = np.random.default_rng(0)
    n = 640
    parents = {"X1": rng.normal(size=n), "X2": rng.normal(size=n)}
    child = (rng.random(n) < 0.5).astype(int)
    with pytest.raises(ValueError, match="_MAX_CPT_ROWS"):
        TabularCPT().fit(parents, child)


def test_tabular_cpt_rejects_a_non_positive_pseudo_count():
    # I8 regression: alpha=0 is the obvious maximum-likelihood choice, and used to be accepted --
    # an unobserved parent configuration then has an all-zero count row whose 0/0 normalisation
    # makes every draw there the smallest level, silently.
    with pytest.raises(ValueError, match="must be positive"):
        TabularCPT(alpha=0.0)
    with pytest.raises(ValueError, match="must be positive"):
        TabularCPT(alpha=-1.0)


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


def test_linear_gaussian_residual_round_trips():
    # invertible=True means mechanism(parents, residual(parents, v)) == v exactly; Task 8 computes
    # counterfactuals by inverting noise through residual(), so this must hold on held-out rows,
    # not just training data the mechanism was fit on.
    rng = np.random.default_rng(8)
    x = rng.normal(size=4000)
    z = rng.normal(size=4000)
    y = 2.0 * x - 0.5 * z + 1.0 + rng.normal(scale=0.3, size=4000)
    fitted = LinearGaussianFit().fit({"X": x[:3000], "Z": z[:3000]}, y[:3000])
    held_out = {
        "X": torch.tensor(x[3000:], dtype=torch.float32),
        "Z": torch.tensor(z[3000:], dtype=torch.float32),
    }
    value = torch.tensor(y[3000:], dtype=torch.float32)
    noise = fitted.mechanism.residual(held_out, value)  # type: ignore[attr-defined]
    recovered = fitted.mechanism(held_out, noise)
    assert torch.allclose(recovered, value, rtol=0.0, atol=1e-4)


def test_linear_gaussian_root_residual_round_trips():
    # Same invariant on the root/intercept-only path (no parents) -- a different branch through
    # _design/_attach_residual than the with-parents case above, so it can break independently.
    rng = np.random.default_rng(9)
    y = rng.normal(loc=3.0, scale=2.0, size=4000)
    fitted = LinearGaussianFit().fit({}, y[:3000])
    value = torch.tensor(y[3000:], dtype=torch.float32)
    noise = fitted.mechanism.residual({}, value)  # type: ignore[attr-defined]
    recovered = fitted.mechanism({}, noise)
    assert torch.allclose(recovered, value, rtol=0.0, atol=1e-4)


def test_anm_default_ridge_residual_round_trips():
    # Same invariant through the default _RBFRidge's mean_fn route (model.predict), which
    # LinearGaussianFit's closed-form coefficients never exercise.
    rng = np.random.default_rng(10)
    x = rng.uniform(-2.0, 2.0, size=4000)
    y = np.sin(3.0 * x) + rng.normal(scale=0.1, size=4000)
    fitted = ANMFit().fit({"X": x[:3000]}, y[:3000])
    held_out = {"X": torch.tensor(x[3000:], dtype=torch.float32)}
    value = torch.tensor(y[3000:], dtype=torch.float32)
    noise = fitted.mechanism.residual(held_out, value)  # type: ignore[attr-defined]
    recovered = fitted.mechanism(held_out, noise)
    assert torch.allclose(recovered, value, rtol=0.0, atol=1e-4)


def test_anm_duck_typed_estimator_residual_round_trips():
    # Same invariant through a caller-supplied estimator's mean_fn route, not just the default
    # ridge -- ANMFit's residual/mechanism wiring must not assume _RBFRidge specifically.
    class ConstantModel:
        def fit(self, x, y):
            self.value = float(np.mean(y))
            return self

        def predict(self, x):
            return np.full(len(x), self.value)

    rng = np.random.default_rng(11)
    x = rng.normal(size=2000)
    y = rng.normal(loc=5.0, size=2000)
    fitted = ANMFit(estimator=ConstantModel).fit({"X": x[:1500]}, y[:1500])
    held_out = {"X": torch.tensor(x[1500:], dtype=torch.float32)}
    value = torch.tensor(y[1500:], dtype=torch.float32)
    noise = fitted.mechanism.residual(held_out, value)  # type: ignore[attr-defined]
    recovered = fitted.mechanism(held_out, noise)
    assert torch.allclose(recovered, value, rtol=0.0, atol=1e-4)


def test_neural_fit_learns_a_nonlinear_mean():
    rng = np.random.default_rng(8)
    x = rng.uniform(-2.0, 2.0, size=4000)
    y = np.tanh(2.0 * x) + rng.normal(scale=0.1, size=4000)
    fitted = NeuralFit(epochs=300, seed=0).fit({"X": x}, y)
    grid = torch.linspace(-1.5, 1.5, 32)
    predicted = fitted.mechanism({"X": grid}, torch.zeros(32))
    assert float((predicted - torch.tanh(2.0 * grid)).abs().mean()) < 0.2
    assert fitted.invertible is True
    assert fitted.score > 0.8

    # I3 regression: fitting used torch.manual_seed(), reseeding the caller's process-global
    # stream -- so a user's subsequent draws silently became a function of when they fitted a
    # mechanism. The fit must be deterministic for a given seed AND must neither reseed nor
    # consume the global stream (both would move `after` off `before`).
    torch.manual_seed(1234)
    before = torch.rand(3)
    torch.manual_seed(1234)
    again = NeuralFit(epochs=300, seed=0).fit({"X": x}, y)
    after = torch.rand(3)
    assert torch.equal(after, before)
    assert torch.equal(again.mechanism({"X": grid}, torch.zeros(32)), predicted)


def test_neural_fit_residual_round_trips():
    # Same invariant through NeuralFit's _AdditiveHead route -- the noise must stay exactly
    # additive (net(parents) + noise) for invertible=True to be honest; Task 8 inverts noise
    # through residual() here the same way it does for the other three invertible families.
    rng = np.random.default_rng(12)
    x = rng.uniform(-2.0, 2.0, size=4000)
    y = np.tanh(2.0 * x) + rng.normal(scale=0.1, size=4000)
    fitted = NeuralFit(epochs=300, seed=0).fit({"X": x[:3000]}, y[:3000])
    held_out = {"X": torch.tensor(x[3000:], dtype=torch.float32)}
    value = torch.tensor(y[3000:], dtype=torch.float32)
    noise = fitted.mechanism.residual(held_out, value)  # type: ignore[attr-defined]
    recovered = fitted.mechanism(held_out, noise)
    assert torch.allclose(recovered, value, rtol=0.0, atol=1e-4)


def test_poisson_glm_recovers_log_linear_coefficients():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    rate = np.exp(0.4 + 0.8 * x - 0.5 * z)
    y = rng.poisson(rate)
    fitted = PoissonGLMFit().fit({"X": x, "Z": z}, y)
    coefficients = fitted.mechanism.coefficients  # type: ignore[attr-defined]
    assert abs(coefficients["intercept"] - 0.4) < 0.05
    assert abs(coefficients["X"] - 0.8) < 0.05
    assert abs(coefficients["Z"] + 0.5) < 0.05
    assert fitted.invertible is False


def test_poisson_glm_mechanism_reproduces_the_conditional_mean():
    rng = np.random.default_rng(1)
    n = 20_000
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + 0.6 * x))
    fitted = PoissonGLMFit().fit({"X": x}, y)
    m = 40_000
    u = fitted.noise.sample((m,)).reshape(m).float()
    drawn = fitted.mechanism({"X": torch.ones(m)}, u)
    assert abs(float(drawn.mean()) - float(np.exp(0.8))) < 0.05
    # Mutation guard: a mean predictor that discards `noise` and returns `rate(...)` directly would
    # satisfy the mean assertion above exactly (constant lambda for every unit), so the inverse-CDF
    # construction needs its own check -- integer-valued draws with Poisson dispersion
    # (variance/mean ~= 1), neither of which a mean predictor can produce.
    assert torch.allclose(drawn, drawn.round())
    variance = float(drawn.var())
    mean = float(drawn.mean())
    assert abs(variance / mean - 1.0) < 0.1


def test_poisson_glm_attaches_log_prob_for_holdout_scoring():
    rng = np.random.default_rng(2)
    n = 8000
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.7 * x))
    fitted = PoissonGLMFit().fit({"X": x}, y)
    informative = float(
        fitted.mechanism.log_prob(  # type: ignore[attr-defined]
            {"X": torch.tensor(x, dtype=torch.float32)},
            torch.tensor(y, dtype=torch.float32),
        ).mean()
    )
    flat = PoissonGLMFit().fit({}, y)
    uninformative = float(
        flat.mechanism.log_prob(  # type: ignore[attr-defined]
            {}, torch.tensor(y, dtype=torch.float32)
        ).mean()
    )
    assert informative > uninformative


def test_poisson_glm_beats_tabular_cpt_on_many_lagged_parents():
    # 6 ternary parents = 729 configurations; at this n most are occupied but each by only a
    # handful of raw rows, so TabularCPT's Laplace smoothing still dominates cell-by-cell while the
    # GLM pools information across configurations through its shared linear coefficients. Fit on
    # one partition, score on a disjoint 500-row holdout neither model trained on -- a same-rows
    # in-sample comparison is fragile (favours the CPT, which can memorise rows it was fit on) and
    # does not test the claim in this test's own name.
    rng = np.random.default_rng(3)
    n = 3000
    parents = {f"lag{i}": rng.integers(0, 3, size=n).astype(float) for i in range(6)}
    eta = 0.1 + sum(0.15 * parents[f"lag{i}"] for i in range(6))
    y = rng.poisson(np.exp(eta))
    train = {k: v[500:] for k, v in parents.items()}
    held = {k: v[:500] for k, v in parents.items()}
    glm = PoissonGLMFit().fit(train, y[500:])
    cpt = TabularCPT().fit({k: v.astype(int) for k, v in train.items()}, y[500:])
    target = torch.tensor(y[:500], dtype=torch.float32)
    glm_ll = float(
        glm.mechanism.log_prob(  # type: ignore[attr-defined]
            {k: torch.tensor(v, dtype=torch.float32) for k, v in held.items()}, target
        ).mean()
    )
    cpt_ll = float(
        cpt.mechanism.log_prob(  # type: ignore[attr-defined]
            {k: torch.tensor(v, dtype=torch.float32) for k, v in held.items()}, target
        ).mean()
    )
    assert glm_ll > cpt_ll + 0.2


def test_poisson_glm_refuses_a_cell_budget_blown_by_a_runaway_lambda():
    # A runaway upstream VALUE (not a runaway fit) pushes eta to its clamp boundary; the sampler's
    # per-row support table must refuse rather than build a multi-gigabyte allocation. Small n,
    # huge lambda -- the corner a column-only guard can see.
    rng = np.random.default_rng(4)
    x = rng.normal(size=2000)
    y = rng.poisson(np.exp(0.1 + 0.1 * x))
    fitted = PoissonGLMFit().fit({"X": x}, y)
    huge = torch.full((4,), 1000.0)
    with pytest.raises(ValueError, match="_MAX_POISSON_CELLS"):
        fitted.mechanism({"X": huge}, torch.rand(4))


def test_poisson_glm_refuses_a_cell_budget_blown_by_a_large_batch():
    # Same guard, the corner a column-only guard CANNOT see: an ordinary lambda (cap stays at the
    # 20 floor -- no runaway rate anywhere) but a batch large enough that n * (cap + 1) alone
    # exceeds the cell budget. Nothing about this call is pathological except its size.
    rng = np.random.default_rng(5)
    x = rng.normal(size=2000)
    y = rng.poisson(np.exp(0.1 + 0.1 * x))
    fitted = PoissonGLMFit().fit({"X": x}, y)
    n = 2_000_000
    with pytest.raises(ValueError, match="_MAX_POISSON_CELLS"):
        fitted.mechanism({"X": torch.zeros(n)}, torch.rand(n))
