"""Interference: exposure mappings and the direct / spillover / total contrasts they license."""

import itertools

import numpy as np
import pytest

from causalrl.exceptions import NotIdentifiableError
from causalrl.interference import (
    adjacency_from_matrix,
    any_neighbour_treated,
    direct_effect,
    neighbourhood_count,
    neighbourhood_fraction,
    population_share,
    spillover_effect,
    total_effect,
)

# A three-unit line: 0 - 1 - 2.
_LINE = [[1], [0, 2], [1]]


def test_neighbourhood_count_excludes_the_unit_itself() -> None:
    mapping = neighbourhood_count(_LINE)
    # Unit 0 is treated but its only peer is not, so its own treatment must not leak in.
    assert list(mapping.column(np.array([1, 0, 1]))) == [0, 2, 0]


def test_neighbourhood_fraction_rounds_into_strata() -> None:
    mapping = neighbourhood_fraction(_LINE)
    assert list(mapping.column(np.array([1, 0, 1]))) == [0.0, 1.0, 0.0]
    assert list(mapping.column(np.array([1, 0, 0]))) == [0.0, 0.5, 0.0]


def test_neighbourhood_fraction_of_a_peerless_unit_is_zero() -> None:
    mapping = neighbourhood_fraction([[], [0]])
    assert list(mapping.column(np.array([1, 1]))) == [0.0, 1.0]


def test_any_neighbour_treated_is_the_binary_regime() -> None:
    mapping = any_neighbour_treated(_LINE)
    assert list(mapping.column(np.array([1, 0, 0]))) == [False, True, False]


def test_population_share_counts_every_other_unit() -> None:
    mapping = population_share()
    # Unit 0 treated, units 1-3 give 2 of 3 others treated -> 0.67.
    assert mapping(0, np.array([1, 1, 1, 0])) == pytest.approx(0.67)
    assert mapping(3, np.array([1, 1, 1, 0])) == pytest.approx(1.0)


def test_population_share_of_a_lone_unit_is_zero() -> None:
    assert population_share()(0, np.array([1])) == 0.0


def test_adjacency_from_matrix_reads_rows_as_peers() -> None:
    matrix = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    assert [list(row) for row in adjacency_from_matrix(matrix)] == _LINE


def test_adjacency_from_matrix_refuses_a_self_edge() -> None:
    with pytest.raises(ValueError, match="nonzero diagonal"):
        adjacency_from_matrix(np.array([[1, 0], [0, 0]]))


def test_adjacency_from_matrix_refuses_a_non_square_matrix() -> None:
    with pytest.raises(ValueError, match="must be square"):
        adjacency_from_matrix(np.array([[0, 1, 0], [1, 0, 1]]))


def test_a_self_peer_in_an_adjacency_list_is_refused() -> None:
    with pytest.raises(ValueError, match="own peer"):
        neighbourhood_count([[0], [0]])


def test_column_refuses_a_non_vector() -> None:
    with pytest.raises(ValueError, match="1-D vector"):
        population_share().column(np.zeros((2, 2)))


def _additive_population() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four fully-coupled units under every treatment vector, with ``Y = 2*A + 3*share``.

    Enumerating all 16 assignments populates every (own treatment, exposure) cell, and the
    outcome is noiseless, so the estimators must recover 2.0 and 3.0 exactly.
    """
    mapping = population_share()
    outcome: list[float] = []
    treatment: list[int] = []
    exposure: list[float] = []
    for vector in itertools.product([0, 1], repeat=4):
        treatments = np.array(vector)
        shares = mapping.column(treatments)
        for unit in range(4):
            treatment.append(int(treatments[unit]))
            exposure.append(float(shares[unit]))
            outcome.append(2.0 * treatments[unit] + 3.0 * shares[unit])
    return np.array(outcome), np.array(treatment), np.array(exposure)


def test_direct_effect_recovers_the_own_treatment_coefficient() -> None:
    y, a, e = _additive_population()
    contrast = direct_effect(y, a, e, at_exposure=0.0)
    assert contrast.estimate == pytest.approx(2.0)
    assert contrast.estimand == "direct_effect"
    assert contrast.high.n > 0 and contrast.low.n > 0


def test_direct_effect_is_the_same_at_every_exposure_under_additivity() -> None:
    y, a, e = _additive_population()
    for at in (0.0, 1.0):
        assert direct_effect(y, a, e, at_exposure=at).estimate == pytest.approx(2.0)


def test_spillover_effect_recovers_the_peer_coefficient() -> None:
    y, a, e = _additive_population()
    contrast = spillover_effect(y, a, e, at_treatment=0, exposed=1.0, unexposed=0.0)
    assert contrast.estimate == pytest.approx(3.0)
    assert contrast.estimand == "spillover_effect"


def test_total_effect_moves_both_at_once() -> None:
    y, a, e = _additive_population()
    contrast = total_effect(
        y, a, e, treated=1, treated_exposure=1.0, control=0, control_exposure=0.0
    )
    assert contrast.estimate == pytest.approx(5.0)


def _clustered_population() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The same additive outcome under a design that treats whole groups together.

    Enumerating every assignment (as :func:`_additive_population` does) makes a unit's own
    treatment independent of its exposure by symmetry, which is exactly the case where the naive
    marginal contrast happens to be unbiased. Clustered assignment — the realistic design — breaks
    that independence while still reaching the two cells the direct effect needs.
    """
    mapping = population_share()
    vectors = [(0, 0, 0, 0)] * 10 + [(1, 1, 1, 1)] * 10 + [(1, 0, 0, 0), (0, 1, 1, 1)]
    outcome: list[float] = []
    treatment: list[int] = []
    exposure: list[float] = []
    for vector in vectors:
        treatments = np.array(vector)
        shares = mapping.column(treatments)
        for unit in range(4):
            treatment.append(int(treatments[unit]))
            exposure.append(float(shares[unit]))
            outcome.append(2.0 * treatments[unit] + 3.0 * shares[unit])
    return np.array(outcome), np.array(treatment), np.array(exposure)


def test_marginal_comparison_would_confound_direct_with_spillover() -> None:
    # The reason direct_effect holds exposure fixed: under clustered assignment a unit's own
    # treatment is correlated with its peers', so the naive treated-minus-untreated mean picks up
    # the spillover as well and lands near the total effect (5.0) rather than the direct one.
    y, a, e = _clustered_population()
    naive = float(y[a == 1].mean() - y[a == 0].mean())
    assert naive == pytest.approx(5.0, abs=0.3)
    assert direct_effect(y, a, e, at_exposure=0.0).estimate == pytest.approx(2.0)


def test_an_empty_cell_raises_positivity_rather_than_extrapolating() -> None:
    y, a, e = _additive_population()
    with pytest.raises(NotIdentifiableError) as excinfo:
        direct_effect(y, a, e, at_exposure=0.42)  # a stratum the design never realises
    assert "positivity" in str(excinfo.value)
    assert excinfo.value.witness == (1, 0.42)


def test_cells_report_the_footing_the_estimate_rests_on() -> None:
    y, a, e = _additive_population()
    contrast = direct_effect(y, a, e, at_exposure=0.0)
    assert contrast.high.treatment == 1
    assert contrast.high.exposure == 0.0
    assert contrast.estimate == pytest.approx(contrast.high.mean - contrast.low.mean)
    assert "direct_effect" in contrast.summary()


def test_misaligned_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="same length"):
        direct_effect(np.zeros(4), np.zeros(3), np.zeros(4), at_exposure=0.0)


def test_non_vector_columns_are_refused() -> None:
    with pytest.raises(ValueError, match="1-D"):
        direct_effect(np.zeros((2, 2)), np.zeros(4), np.zeros(4), at_exposure=0.0)


def test_no_observations_is_refused() -> None:
    with pytest.raises(ValueError, match="no observations"):
        direct_effect(np.array([]), np.array([]), np.array([]), at_exposure=0.0)
