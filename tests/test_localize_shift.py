"""localize_mechanism_shift: find which mechanisms moved, rather than being told."""

from __future__ import annotations

import numpy as np
import pytest

from causalrl import CausalGraph, Kind, identify_transport
from causalrl.transport.localize import localize_mechanism_shift


def _chain_data(
    *, n: int, seed: int, p_x: float, y_shift: float, z_coef: float = 0.7
) -> dict[str, np.ndarray]:
    """X -> Y -> Z. ``p_x`` moves X's mechanism; ``y_shift`` moves Y's; Z's is always the same."""
    rng = np.random.default_rng(seed)
    x = (rng.random(n) < p_x).astype(np.int_)
    y = (rng.random(n) < 0.2 + 0.5 * x + y_shift).astype(np.int_)
    z = (rng.random(n) < 0.15 + z_coef * y).astype(np.int_)
    return {"X": x, "Y": y, "Z": z}


def _graph() -> CausalGraph:
    return CausalGraph(directed_edges=[("X", "Y"), ("Y", "Z")], nodes=["X", "Y", "Z"])


def test_it_finds_the_one_mechanism_that_moved() -> None:
    """Only X's marginal changes; Y's and Z's conditionals are identical."""
    source = _chain_data(n=6000, seed=0, p_x=0.3, y_shift=0.0)
    target = _chain_data(n=6000, seed=1, p_x=0.8, y_shift=0.0)

    report = localize_mechanism_shift({"source": source, "target": target}, graph=_graph())

    assert report.selection == frozenset({"X"})
    assert report.invariant == frozenset({"Y", "Z"})


def test_it_finds_a_shifted_conditional_not_just_a_shifted_marginal() -> None:
    """Y's marginal moves in the first test only because X's did. Here Y's MECHANISM moves."""
    source = _chain_data(n=6000, seed=2, p_x=0.5, y_shift=0.0)
    target = _chain_data(n=6000, seed=3, p_x=0.5, y_shift=0.25)

    report = localize_mechanism_shift({"source": source, "target": target}, graph=_graph())

    assert "Y" in report.selection
    assert "X" not in report.selection  # its mechanism is untouched


def test_no_shift_is_reported_when_the_regimes_agree() -> None:
    source = _chain_data(n=5000, seed=4, p_x=0.4, y_shift=0.0)
    target = _chain_data(n=5000, seed=5, p_x=0.4, y_shift=0.0)

    report = localize_mechanism_shift({"source": source, "target": target}, graph=_graph())

    assert report.selection == frozenset()
    assert report.certificate().kind is Kind.EMPIRICAL


def test_the_selection_set_plugs_into_identify_transport() -> None:
    """The loop closes: what this emits is exactly what the transport algorithm consumes."""
    source = _chain_data(n=4000, seed=6, p_x=0.3, y_shift=0.0)
    target = _chain_data(n=4000, seed=7, p_x=0.85, y_shift=0.0)

    report = localize_mechanism_shift({"source": source, "target": target}, graph=_graph())
    estimand = identify_transport(_graph(), ["Y"], ["Z"], report.selection)

    assert estimand.render()


def test_the_certificate_is_empirical_and_reports_every_p_value() -> None:
    """Failing to reject invariance is not proof of it, and multiplicity is the caller's call."""
    source = _chain_data(n=3000, seed=8, p_x=0.3, y_shift=0.0)
    target = _chain_data(n=3000, seed=9, p_x=0.7, y_shift=0.0)

    report = localize_mechanism_shift({"source": source, "target": target}, graph=_graph())
    cert = report.certificate()

    assert cert.kind is Kind.EMPIRICAL
    multiplicity = next(a for a in cert.assumptions if a.name == "no-multiplicity-correction")
    assert multiplicity.diagnostic is not None
    assert set(multiplicity.diagnostic) == {"X", "Y", "Z"}
    assert len(report.shifts) == 3  # every node reported, not only the rejections
    assert "SHIFTED" in report.summary() or "invariant" in report.summary()


def test_it_refuses_a_single_regime_and_missing_columns() -> None:
    data = _chain_data(n=100, seed=10, p_x=0.5, y_shift=0.0)
    with pytest.raises(ValueError, match="at least 2 regimes"):
        localize_mechanism_shift({"only": data}, graph=_graph())
    with pytest.raises(KeyError, match="missing column"):
        localize_mechanism_shift({"a": data, "b": {"X": data["X"], "Y": data["Y"]}}, graph=_graph())
