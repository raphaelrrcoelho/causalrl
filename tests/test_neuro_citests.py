"""Conditional-independence tests for continuous and point-process data."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.neuro.citests import (
    CITestError,
    KnnCMITest,
    PartialCorrelationTest,
    PoissonGLMTest,
    chi2_sf,
    digamma,
    normal_sf,
)


def test_digamma_matches_known_values() -> None:
    assert float(digamma(1.0)) == pytest.approx(-0.5772156649, abs=1e-8)
    assert float(digamma(2.0)) == pytest.approx(0.4227843351, abs=1e-8)
    assert float(digamma(0.5)) == pytest.approx(-1.9635100260, abs=1e-8)
    assert float(digamma(10.0)) == pytest.approx(2.2517525891, abs=1e-9)


def test_digamma_rejects_non_positive_arguments() -> None:
    with pytest.raises(CITestError, match="strictly positive"):
        digamma(0.0)


def test_chi2_and_normal_tails_match_standard_critical_values() -> None:
    assert chi2_sf(3.8415, 1) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf(5.9915, 2) == pytest.approx(0.05, abs=1e-4)
    assert chi2_sf(0.5, 3) == pytest.approx(0.918891, abs=1e-5)
    assert chi2_sf(0.0, 2) == 1.0
    assert normal_sf(1.959964) == pytest.approx(0.025, abs=1e-6)


def _chain(n: int = 4000, seed: int = 0) -> dict[str, np.ndarray]:
    """X -> Y -> Z, so X is dependent on Z marginally but independent given Y."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 0.8 * x + 0.6 * rng.standard_normal(n)
    z = 0.8 * y + 0.6 * rng.standard_normal(n)
    return {"X": x, "Y": y, "Z": z}


def test_partial_correlation_separates_a_chain() -> None:
    test = PartialCorrelationTest(alpha=0.01)
    assert not test(_chain(), "X", "Z", [])
    assert test(_chain(), "X", "Z", ["Y"])


def test_partial_correlation_value_shrinks_when_conditioning() -> None:
    test = PartialCorrelationTest()
    data = _chain()
    assert abs(test.partial_correlation(data, "X", "Z", [])) > 0.4
    assert abs(test.partial_correlation(data, "X", "Z", ["Y"])) < 0.05


def test_knn_cmi_separates_a_chain() -> None:
    test = KnnCMITest(max_samples=1200, seed=0)
    marginal, _ = test.estimate(_chain(), "X", "Z", [])
    conditional, _ = test.estimate(_chain(), "X", "Z", ["Y"])
    assert marginal > 0.1
    assert conditional < 0.02


def test_knn_cmi_permutation_p_value_is_calibrated_under_the_null() -> None:
    test = KnnCMITest(max_samples=400, permutations=120, alpha=0.02, seed=1)
    result = test(_chain(), "X", "Z", ["Y"])
    assert result.p_value is not None
    assert result.independent


def test_knn_cmi_permutation_rejects_a_genuine_dependence() -> None:
    test = KnnCMITest(max_samples=400, permutations=120, alpha=0.02, seed=1)
    result = test(_chain(), "X", "Z", [])
    assert result.p_value is not None and result.p_value < 0.02
    assert not result.independent


def test_knn_cmi_refuses_an_unreachable_alpha() -> None:
    test = KnnCMITest(max_samples=200, permutations=10, alpha=0.001, seed=0)
    with pytest.raises(CITestError, match="unreachable"):
        test(_chain(), "X", "Z", [])


def _poisson_chain(n: int = 5000, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    cy = rng.poisson(np.exp(-1.0 + 0.7 * x)).astype(np.float64)
    cz = rng.poisson(np.exp(-1.0 + 0.5 * cy)).astype(np.float64)
    return {"X": x, "CY": cy, "CZ": cz}


def test_poisson_glm_separates_a_point_process_chain() -> None:
    test = PoissonGLMTest(alpha=0.01)
    data = _poisson_chain()
    assert not test(data, "X", "CZ", [])
    assert test(data, "X", "CZ", ["CY"])


def test_poisson_glm_is_calibrated_on_independent_counts() -> None:
    rng = np.random.default_rng(3)
    n = 4000
    data = {
        "X": rng.standard_normal(n),
        "Y": rng.poisson(0.2, size=n).astype(np.float64),
    }
    result = PoissonGLMTest(alpha=0.01)(data, "X", "Y", [])
    assert result.independent
    assert result.p_value is not None and result.p_value > 0.01


def test_poisson_glm_handles_a_silent_unit() -> None:
    n = 500
    data = {"X": np.arange(n, dtype=np.float64), "Y": np.zeros(n)}
    result = PoissonGLMTest()(data, "X", "Y", [])
    assert result.independent and result.p_value == 1.0


def test_poisson_glm_rejects_negative_counts() -> None:
    data = {"X": np.ones(10), "Y": -np.ones(10)}
    with pytest.raises(CITestError, match="non-negative counts"):
        PoissonGLMTest()(data, "X", "Y", [])


def test_ci_test_result_is_truthy_exactly_when_independent() -> None:
    test = PartialCorrelationTest(alpha=0.01)
    assert bool(test(_chain(), "X", "Z", ["Y"])) is True
    assert bool(test(_chain(), "X", "Z", [])) is False


def test_reduced_model_cache_does_not_change_the_verdict() -> None:
    """Caching the null fit is an optimisation; it must reproduce the uncached statistic exactly."""
    data = _poisson_chain()
    cached = PoissonGLMTest(alpha=0.01, cache_reduced=True)
    uncached = PoissonGLMTest(alpha=0.01, cache_reduced=False)
    for z in ([], ["CY"]):
        a = cached(data, "X", "CZ", z)
        b = uncached(data, "X", "CZ", z)
        assert a.statistic == pytest.approx(b.statistic, rel=1e-9)
        assert a.independent == b.independent


def test_reduced_model_cache_is_reused_across_candidates_sharing_a_null() -> None:
    data = _poisson_chain()
    test = PoissonGLMTest(alpha=0.01)
    test(data, "X", "CZ", [])
    test(data, "CY", "CZ", [])  # same target, same (empty) conditioning set
    assert len(test._cache) == 1


def test_reduced_model_cache_separates_different_data() -> None:
    """The key carries sample size and column sums, so another dataset cannot collide."""
    test = PoissonGLMTest(alpha=0.01)
    test(_poisson_chain(seed=0), "X", "CZ", [])
    test(_poisson_chain(seed=1), "X", "CZ", [])
    assert len(test._cache) == 2


def test_warm_started_full_model_matches_a_cold_start() -> None:
    data = _poisson_chain()
    test = PoissonGLMTest(alpha=0.01)
    ys = data["CZ"]
    design = np.column_stack([np.ones(len(ys)), data["CY"], data["X"]])
    cold, _ = test._fit(ys, design)
    warm, _ = test._fit(ys, design, np.array([0.1, 0.2, 0.3]))
    assert cold == pytest.approx(warm, rel=1e-6)
