"""Anytime-valid bands: the guarantee is coverage under repeated looking, so test exactly that."""

from __future__ import annotations

import math

import numpy as np
import pytest

from causalrl import ConfoundedTrajectoryDataset, Kind, Transition
from causalrl.ope.sequential_test import (
    confidence_sequence,
    sequential_policy_comparison,
)


def _peeking_false_positives(*, use_sequence: bool, trials: int, horizon: int, seed: int) -> float:
    """Fraction of null trials in which a procedure EVER rejects while being watched every step.

    The null is true (mean zero), so every rejection is an error. A fixed-sample interval promises
    5% at one pre-declared sample size and makes no promise at all under this protocol; a
    confidence sequence promises 5% over the whole path.
    """
    rng = np.random.default_rng(seed)
    alpha = 0.05
    z = 1.959963984540054
    errors = 0
    for _ in range(trials):
        x = rng.uniform(-1.0, 1.0, size=horizon)
        running_sum = np.cumsum(x)
        n = np.arange(1, horizon + 1)
        mean = running_sum / n
        if use_sequence:
            sigma, rho = 1.0, 1.0 / horizon
            inner = (n * rho + 1.0) / (n * n * rho)
            radius = sigma * np.sqrt(2.0 * inner * np.log(np.sqrt(n * rho + 1.0) / alpha))
        else:
            running_var = np.cumsum(x**2) / n - mean**2
            radius = z * np.sqrt(np.maximum(running_var, 0.0) / n)
        # "Stop as soon as it looks decisive", checked at every single sample size.
        if np.any((mean - radius > 0.0) | (mean + radius < 0.0)):
            errors += 1
    return errors / trials


def test_the_sequence_survives_peeking_where_a_fixed_sample_interval_does_not() -> None:
    """The whole reason this exists. Same data, same alpha, same stopping rule."""
    sequence_rate = _peeking_false_positives(use_sequence=True, trials=200, horizon=500, seed=0)
    fixed_rate = _peeking_false_positives(use_sequence=False, trials=200, horizon=500, seed=0)

    assert sequence_rate <= 0.05, f"time-uniform guarantee violated: {sequence_rate:.3f}"
    assert fixed_rate > 0.15, (
        f"fixed-sample peeking should inflate error badly, got {fixed_rate:.3f}"
    )
    assert fixed_rate > 3 * max(sequence_rate, 0.005)


def test_the_band_covers_the_truth() -> None:
    rng = np.random.default_rng(1)
    x = rng.uniform(-1.0, 1.0, size=2000) + 0.3
    cs = confidence_sequence(x, alpha=0.05, value_range=(-0.7, 1.3))
    assert cs.interval.lower <= 0.3 <= cs.interval.upper


def test_a_real_difference_is_eventually_separated_from_zero() -> None:
    """Anytime validity is worthless without power: a true effect must still be detected."""
    rng = np.random.default_rng(2)
    x = rng.uniform(-1.0, 1.0, size=20000) + 0.5
    cs = confidence_sequence(x, alpha=0.05, value_range=(-0.5, 1.5))
    assert cs.excludes_zero
    assert cs.interval.lower > 0.0


def test_the_band_shrinks_as_evidence_accumulates() -> None:
    rng = np.random.default_rng(3)
    x = rng.uniform(-1.0, 1.0, size=10000)
    narrow = confidence_sequence(x, alpha=0.05, value_range=(-1.0, 1.0))
    wide = confidence_sequence(x[:100], alpha=0.05, value_range=(-1.0, 1.0))
    assert narrow.width < wide.width


def test_a_range_that_does_not_contain_the_data_is_refused() -> None:
    """A too-narrow range voids the guarantee outright, so it must not be a silent widening."""
    with pytest.raises(ValueError, match="does not contain every observation"):
        confidence_sequence([0.0, 5.0], alpha=0.05, value_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="at least one observation"):
        confidence_sequence([], alpha=0.05, value_range=(-1.0, 1.0))
    with pytest.raises(ValueError, match="alpha"):
        confidence_sequence([0.0], alpha=1.5, value_range=(-1.0, 1.0))


def _log(better_action: int, n: int, seed: int) -> ConfoundedTrajectoryDataset:
    rng = np.random.default_rng(seed)
    transitions = []
    for _ in range(n):
        a = int(rng.integers(0, 2))
        reward = 1.0 if a == better_action and rng.random() < 0.9 else 0.0
        transitions.append(Transition(0, a, reward, 0, True))
    return ConfoundedTrajectoryDataset(transitions, n_states=1, n_actions=2)


def test_policy_comparison_reports_a_stopping_verdict() -> None:
    dataset = _log(better_action=1, n=4000, seed=4)
    verdict = sequential_policy_comparison(
        dataset, [1] * len(dataset), alpha=0.05, reward_range=(0.0, 1.0), weight_cap=4.0
    )

    assert verdict.certificate.kind is Kind.EMPIRICAL
    assert "every sample size" in verdict.certificate.claim
    assert verdict.stop is True
    assert verdict.better == "target"
    assert verdict.certificate.hedge is None


def test_an_undecided_comparison_hedges_instead_of_reporting_a_winner() -> None:
    dataset = _log(better_action=1, n=30, seed=5)
    verdict = sequential_policy_comparison(
        dataset, [1] * len(dataset), alpha=0.05, reward_range=(0.0, 1.0), weight_cap=4.0
    )

    assert verdict.stop is False
    assert verdict.better is None
    assert verdict.certificate.hedge is not None
    assert "undecided" in verdict.certificate.hedge.reason


def test_the_radius_is_finite_and_positive_at_every_size() -> None:
    for n in (1, 2, 10, 1000):
        cs = confidence_sequence([0.0] * n, alpha=0.05, value_range=(-1.0, 1.0))
        assert math.isfinite(cs.width) and cs.width > 0.0
