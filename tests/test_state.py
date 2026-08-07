"""State encoders — the observation-side vocabulary that replaces the integer state index."""

import numpy as np
import pytest

from causalrl.state import (
    FeatureTransition,
    IdentityEncoder,
    OneHotEncoder,
    RBFEncoder,
    StateEncoder,
    encode_batch,
)


def test_one_hot_encodes_the_tabular_case() -> None:
    encoder = OneHotEncoder(4)
    assert encoder.dim == 4
    assert list(encoder.encode({"state": 2})) == [0.0, 0.0, 1.0, 0.0]


def test_one_hot_reads_a_custom_key() -> None:
    assert list(OneHotEncoder(2, key="s").encode({"s": 1})) == [0.0, 1.0]


def test_one_hot_rejects_an_index_from_a_different_space() -> None:
    with pytest.raises(IndexError, match="outside"):
        OneHotEncoder(3).encode({"state": 5})


def test_one_hot_requires_at_least_one_state() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        OneHotEncoder(0)


def test_identity_reads_named_entries_in_order() -> None:
    encoder = IdentityEncoder(["b", "a"])
    assert encoder.dim == 2
    assert list(encoder.encode({"a": 1.0, "b": 2.0})) == [2.0, 1.0]


def test_identity_requires_at_least_one_key() -> None:
    with pytest.raises(ValueError, match="at least one"):
        IdentityEncoder([])


def test_rbf_prepends_a_constant_and_peaks_at_its_centre() -> None:
    inner = IdentityEncoder(["x"])
    centers = np.array([[0.0], [1.0]])
    encoder = RBFEncoder(inner, centers, bandwidth=0.25)
    assert encoder.dim == 3  # constant + one feature per centre
    features = encoder.encode({"x": 0.0})
    assert features[0] == 1.0
    assert features[1] == pytest.approx(1.0)  # exactly at the first centre
    assert features[2] < 1e-3  # far from the second


def test_rbf_checks_centre_shape_against_the_inner_encoder() -> None:
    with pytest.raises(ValueError, match="must have shape"):
        RBFEncoder(IdentityEncoder(["x"]), np.zeros((3, 2)))


def test_rbf_requires_a_positive_bandwidth() -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        RBFEncoder(IdentityEncoder(["x"]), np.zeros((2, 1)), bandwidth=0.0)


def test_encoders_satisfy_the_protocol() -> None:
    for encoder in (OneHotEncoder(2), IdentityEncoder(["x"])):
        assert isinstance(encoder, StateEncoder)


def test_encode_batch_stacks_into_a_design_matrix() -> None:
    encoder = OneHotEncoder(3)
    matrix = encode_batch(encoder, [{"state": 0}, {"state": 2}])
    assert matrix.shape == (2, 3)
    assert list(matrix[1]) == [0.0, 0.0, 1.0]


def test_encode_batch_of_nothing_keeps_the_feature_width() -> None:
    # A caller assembling a design matrix from a possibly-empty slice should not have to
    # special-case it, so the empty result still has the right number of columns.
    assert encode_batch(OneHotEncoder(4), []).shape == (0, 4)


def test_feature_transition_normalises_its_endpoints() -> None:
    transition = FeatureTransition(np.array([[1.0, 0.0]]), 1, 0.5, [0.0, 1.0], False)
    assert transition.state.shape == (2,)
    assert transition.next_state.shape == (2,)


def test_feature_transition_refuses_mismatched_endpoints() -> None:
    with pytest.raises(ValueError, match="share a feature dimension"):
        FeatureTransition(np.zeros(3), 0, 0.0, np.zeros(2), False)
