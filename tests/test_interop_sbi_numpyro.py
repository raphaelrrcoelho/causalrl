"""Phase 4 §10: posterior-over-parameters -> Regime samplers (pure numpy; adapters duck-typed)."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.identification.bounds import Interval
from causalrl.interop.sbi_numpyro import (
    PosteriorRegimeSampler,
    across_regimes,
    regimes_from_numpyro,
    regimes_from_posterior,
    regimes_from_sbi_posterior,
)
from causalrl.regime import Regime


def test_regimes_from_posterior_dict_carries_params() -> None:
    samples = {"theta": [0.1, 0.2, 0.3], "sigma": [1.0, 1.1, 1.2]}
    regimes = regimes_from_posterior(samples, selection=["M"])
    assert len(regimes) == 3
    assert all(isinstance(r, Regime) for r in regimes)
    assert regimes[0].params == {"theta": 0.1, "sigma": 1.0}
    assert regimes[2].params == {"theta": 0.3, "sigma": 1.2}
    assert regimes[0].selection == frozenset({"M"})


def test_regimes_from_posterior_matrix_form() -> None:
    matrix = np.array([[0.5, 2.0], [0.6, 2.1]])
    regimes = regimes_from_posterior((matrix, ["a", "b"]))
    assert regimes[1].params == {"a": 0.6, "b": 2.1}


def test_max_regimes_subsamples_deterministically() -> None:
    samples = {"theta": np.arange(100, dtype=float)}
    a = regimes_from_posterior(samples, max_regimes=10, seed=0)
    b = regimes_from_posterior(samples, max_regimes=10, seed=0)
    assert len(a) == 10
    assert [r.params["theta"] for r in a] == [r.params["theta"] for r in b]  # seed-deterministic


def test_mismatched_column_lengths_rejected() -> None:
    with pytest.raises(ValueError, match="samples"):
        regimes_from_posterior({"a": [1.0, 2.0], "b": [1.0]})


def test_sampler_regimes_sample_and_mean() -> None:
    samples = {"theta": [0.0, 1.0, 2.0, 3.0], "k": [10.0, 10.0, 10.0, 10.0]}
    sampler = PosteriorRegimeSampler(samples, selection=["M"])
    assert len(sampler) == 4
    assert sampler.names == ("k", "theta")  # sorted
    assert len(sampler.regimes()) == 4
    drawn = sampler.sample(7, seed=1)
    assert len(drawn) == 7
    assert [r.params["theta"] for r in drawn] == [
        r.params["theta"] for r in sampler.sample(7, seed=1)
    ]
    mean = sampler.mean_regime()
    assert mean.params["theta"] == pytest.approx(1.5)
    assert mean.params["k"] == pytest.approx(10.0)


class _FakeMCMC:
    def get_samples(self):
        return {"theta": np.array([0.2, 0.4]), "sigma": np.array([1.0, 1.5])}


def test_regimes_from_numpyro_duck_typed() -> None:
    regimes = regimes_from_numpyro(_FakeMCMC(), name_prefix="post")
    assert len(regimes) == 2
    assert regimes[0].params == {"theta": 0.2, "sigma": 1.0}
    assert regimes[1].name == "post[1]"


class _FakeSBIPosterior:
    def sample(self, shape, x):
        n = shape[0]
        # A deterministic 2-parameter "posterior" that depends on the observation.
        base = float(np.asarray(x).mean())
        return np.stack([np.full(n, base), np.arange(n, dtype=float)], axis=1)


def test_regimes_from_sbi_posterior_duck_typed() -> None:
    regimes = regimes_from_sbi_posterior(
        _FakeSBIPosterior(), observation=[2.0, 4.0], param_names=["mu", "idx"], n=3
    )
    assert len(regimes) == 3
    assert regimes[0].params["mu"] == pytest.approx(3.0)  # mean of [2, 4]
    assert [r.params["idx"] for r in regimes] == [0.0, 1.0, 2.0]


def test_across_regimes_returns_min_max_interval() -> None:
    regimes = regimes_from_posterior({"theta": [0.1, 0.5, 0.3, 0.9]})
    # A functional of the calibrated configuration (here just the parameter itself).
    interval = across_regimes(regimes, lambda r: r.params["theta"])
    assert isinstance(interval, Interval)
    assert interval.lower == pytest.approx(0.1)
    assert interval.upper == pytest.approx(0.9)


def test_across_regimes_empty_raises() -> None:
    with pytest.raises(ValueError, match="no regimes"):
        across_regimes([], lambda r: 0.0)
