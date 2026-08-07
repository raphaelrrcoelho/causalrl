"""Phase 3 §9: streaming certificate kernels + the end-to-end Parquet-streamed OPE demo (numpy)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from causalrl.bounds.streaming import stream_msm_bounds
from causalrl.certify.certificate import Kind
from causalrl.data.streaming_join import KeyJoiner
from causalrl.data.trajectory import TrajectoryLog
from causalrl.ope.bounds import ipw_sensitivity_bounds
from causalrl.ope.ipw import stream_policy_value


def _weight_reward_log(weights: np.ndarray, rewards: np.ndarray) -> TrajectoryLog:
    """A name-major log (all weight rows, then all reward rows) — exercises the carry-over join."""
    rows: list[dict[str, Any]] = []
    for i, w in enumerate(weights.tolist()):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "w", "name": "weight", "value": float(w)})
    for i, r in enumerate(rewards.tolist()):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "r", "name": "reward", "value": float(r)})
    return TrajectoryLog.from_rows(rows)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _floats(cells: np.ndarray) -> np.ndarray:
    return np.array([float(v) for v in cells.tolist()], dtype=float)


# --- KeyJoiner ---------------------------------------------------------------------------------


def test_key_joiner_matches_across_batch_boundaries() -> None:
    rng = np.random.default_rng(0)
    w = rng.uniform(0.5, 2.0, size=500)
    y = rng.standard_normal(500)
    log = _weight_reward_log(w, y)  # weights and rewards are 500 rows apart
    joiner = KeyJoiner(("weight", "reward"))
    got_w: list[float] = []
    got_r: list[float] = []
    for batch in log.scan(37):  # boundaries fall inside both the weight and reward blocks
        cols = joiner.drain(batch)
        got_w.extend(cols["weight"].tolist())
        got_r.extend(cols["reward"].tolist())
    assert joiner.dropped == 0
    # Recover the by-key pairing exactly (entity order).
    assert np.allclose(sorted(got_w), sorted(w.tolist()))
    assert len(got_r) == 500


def test_key_joiner_reports_incomplete_decisions() -> None:
    rows: list[dict[str, Any]] = [
        {"entity_id": 0, "episode_id": 0, "t": 0, "kind": "w", "name": "weight", "value": 1.0},
        {"entity_id": 0, "episode_id": 0, "t": 0, "kind": "r", "name": "reward", "value": 2.0},
        {"entity_id": 1, "episode_id": 0, "t": 0, "kind": "w", "name": "weight", "value": 1.0},
    ]  # entity 1 never gets a reward
    joiner = KeyJoiner(("weight", "reward"))
    cols = joiner.drain(TrajectoryLog.from_rows(rows))
    assert cols["weight"].tolist() == [1.0] and cols["reward"].tolist() == [2.0]
    assert joiner.dropped == 1


# --- stream_policy_value -----------------------------------------------------------------------


def test_stream_policy_value_matches_hajek() -> None:
    rng = np.random.default_rng(1)
    w = rng.uniform(0.3, 3.0, size=6_000)
    y = rng.standard_normal(6_000) + 0.5
    cert = stream_policy_value(_weight_reward_log(w, y), batch_size=512)
    assert cert.kind is Kind.IDENTIFIED and cert.hedge is None
    assert isinstance(cert.value, float)
    assert math.isclose(cert.value, float(np.average(y, weights=w)), rel_tol=1e-9)
    assert cert.ci is not None and cert.ci.lower < cert.value < cert.ci.upper


def test_stream_policy_value_hedges_on_low_ess() -> None:
    y = np.zeros(2_000)
    w = np.concatenate([np.array([1e6]), np.full(1_999, 1e-4)])  # one weight dominates -> ESS ~ 1
    cert = stream_policy_value(_weight_reward_log(w, y), batch_size=256, min_ess_fraction=0.1)
    assert cert.value is None and cert.hedge is not None
    assert cert.hedge.reason == "overlap-violation"


def test_stream_policy_value_hedges_when_empty() -> None:
    cert = stream_policy_value(_weight_reward_log(np.array([]), np.array([])))
    assert cert.value is None and cert.hedge is not None and cert.hedge.reason == "no-decisions"


# --- stream_msm_bounds -------------------------------------------------------------------------


def _y_e_t_log(y: np.ndarray, e: np.ndarray, treat: np.ndarray) -> TrajectoryLog:
    rows: list[dict[str, Any]] = []
    for i in range(len(y)):
        base = {"entity_id": i, "episode_id": 0, "t": 0}
        rows.append({**base, "kind": "r", "name": "reward", "value": float(y[i])})
        rows.append({**base, "kind": "p", "name": "propensity", "value": float(e[i])})
        rows.append({**base, "kind": "t", "name": "treat", "value": float(treat[i])})
    return TrajectoryLog.from_rows(rows)


def test_stream_msm_bounds_matches_closed_form() -> None:
    rng = np.random.default_rng(3)
    y = rng.uniform(0.0, 1.0, size=3_000)
    e = rng.uniform(0.2, 0.8, size=3_000)
    treat = np.ones(3_000)
    cert = stream_msm_bounds(
        _y_e_t_log(y, e, treat),
        gamma=2.0,
        outcome="reward",
        propensity="propensity",
        batch_size=250,
    )
    assert cert.kind is Kind.BOUNDED
    ref = ipw_sensitivity_bounds(y.tolist(), e.tolist(), gamma=2.0, return_certificate=False)
    assert cert.value is not None
    assert math.isclose(cert.value.lower, ref.lower, rel_tol=1e-9)
    assert math.isclose(cert.value.upper, ref.upper, rel_tol=1e-9)


def test_stream_msm_bounds_collapses_at_gamma_one() -> None:
    rng = np.random.default_rng(4)
    y = rng.uniform(0.0, 1.0, size=1_000)
    e = rng.uniform(0.3, 0.7, size=1_000)
    cert = stream_msm_bounds(
        _y_e_t_log(y, e, np.ones(1_000)), gamma=1.0, outcome="reward", propensity="propensity"
    )
    assert cert.value is not None
    assert math.isclose(cert.value.lower, cert.value.upper, rel_tol=1e-9)


def test_stream_msm_bounds_treatment_filter() -> None:
    rng = np.random.default_rng(5)
    y = rng.uniform(0.0, 1.0, size=2_000)
    e = rng.uniform(0.2, 0.8, size=2_000)
    treat = rng.integers(0, 2, size=2_000).astype(float)
    cert = stream_msm_bounds(
        _y_e_t_log(y, e, treat),
        gamma=1.5,
        outcome="reward",
        propensity="propensity",
        treatment="treat",
        batch_size=300,
    )
    mask = treat > 0.5
    ref = ipw_sensitivity_bounds(
        y[mask].tolist(), e[mask].tolist(), gamma=1.5, return_certificate=False
    )
    assert cert.value is not None and math.isclose(cert.value.lower, ref.lower, rel_tol=1e-9)


# --- end-to-end: Phase-2 population env -> Parquet -> streamed OPE certificate (acceptance) -----


def test_end_to_end_parquet_streamed_ope(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow")
    from causalrl.magames.views import linear_gaussian_population_env

    ego_effect, coplayer_effect, context_effect = 1.5, 0.8, 1.0
    confound, coplayer_bias, target_slope = 1.0, 0.7, 0.5
    view = linear_gaussian_population_env(
        ego="ego",
        ego_effect=ego_effect,
        coplayer_effect=coplayer_effect,
        context_effect=context_effect,
        confound=confound,
        coplayer_bias=coplayer_bias,
        noise=0.5,
    )
    n = 200_000
    log = view.sample(n, seed=7)
    z = _floats(log.values_by_name("Z"))
    a = _floats(log.values_by_name("ego"))
    y = _floats(log.values_by_name("Y"))

    # Importance weight of a target policy pi_t(a=1|z)=sigmoid(target_slope*z) vs the logger.
    pb1, pt1 = _sigmoid(confound * z), _sigmoid(target_slope * z)
    pb = a * pb1 + (1.0 - a) * (1.0 - pb1)
    pt = a * pt1 + (1.0 - a) * (1.0 - pt1)
    rho = pt / pb

    # Ground truth V(π_t): only the action law changes; z and the co-player law are untouched.
    zt = np.random.default_rng(99).standard_normal(2_000_000)
    truth = float(
        (
            ego_effect * _sigmoid(target_slope * zt)
            + coplayer_effect * _sigmoid(coplayer_bias * zt)
            + context_effect * zt
        ).mean()
    )

    # Build a decision log (weight + reward per entity), key-sort it, write Parquet, stream it back.
    log2 = _weight_reward_log(rho, y).sorted_by_key()
    path = tmp_path / "ope_log.parquet"
    log2.to_parquet(path)

    cert = stream_policy_value(str(path), weight="weight", reward="reward", batch_size=50_000)
    assert cert.kind is Kind.IDENTIFIED and cert.hedge is None
    assert isinstance(cert.value, float) and cert.ci is not None
    half = (cert.ci.upper - cert.ci.lower) / 2.0
    assert abs(cert.value - truth) < max(0.1, 3.0 * half)  # streamed OPE recovers the truth

    # Streaming from Parquet == streaming the in-memory log (estimator determinism).
    cert_mem = stream_policy_value(log2, weight="weight", reward="reward", batch_size=50_000)
    assert isinstance(cert_mem.value, float)
    assert math.isclose(cert.value, cert_mem.value, rel_tol=1e-9)


def test_sorted_by_key_orders_rows() -> None:
    rows = [
        {"entity_id": 2, "episode_id": 0, "t": 0, "kind": "r", "name": "reward", "value": 1.0},
        {"entity_id": 0, "episode_id": 0, "t": 1, "kind": "r", "name": "reward", "value": 2.0},
        {"entity_id": 0, "episode_id": 0, "t": 0, "kind": "r", "name": "reward", "value": 3.0},
    ]
    srt = TrajectoryLog.from_rows(rows).sorted_by_key()
    assert srt.column("entity_id").tolist() == [0, 0, 2]
    assert srt.column("t").tolist() == [0, 1, 0]
