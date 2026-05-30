"""Tests for the Interval return type and migrated MSM/partial-ID bounds."""

from causalrl.identification.bounds import Interval, manski_bounds


def test_interval_is_tuple_compatible():
    iv = Interval(0.2, 0.8)
    lo, hi = iv  # unpacks like a tuple
    assert (lo, hi) == (0.2, 0.8)
    assert iv[0] == 0.2 and iv[1] == 0.8
    assert iv.lower == 0.2 and iv.upper == 0.8


def test_manski_returns_interval():
    data = {"t": [1, 1, 0, 0], "y": [1.0, 0.0, 1.0, 0.0]}
    iv = manski_bounds(data, treatment="t", outcome="y", action=1)
    assert isinstance(iv, Interval)
    assert iv.lower <= iv.upper
