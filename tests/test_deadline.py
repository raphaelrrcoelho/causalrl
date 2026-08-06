"""Deadline — the per-decision wall-clock budget an agent may be handed."""

import time

import pytest

from causalrl.deadline import Deadline


def test_after_leaves_roughly_the_requested_budget() -> None:
    deadline = Deadline.after(5.0)
    assert 4.0 < deadline.remaining() <= 5.0
    assert not deadline.expired()


def test_non_positive_duration_is_already_expired() -> None:
    # Allowed on purpose: a caller subtracting elapsed time from a budget should get "no time
    # left", not an exception, when the budget is already gone.
    assert Deadline.after(0.0).expired()
    assert Deadline.after(-1.0).expired()
    assert Deadline.after(-1.0).remaining() == 0.0


def test_remaining_is_clamped_at_zero() -> None:
    past = Deadline(time.monotonic() - 10.0)
    assert past.remaining() == 0.0
    assert past.expired()


def test_fraction_of_splits_the_remaining_budget() -> None:
    deadline = Deadline.after(10.0)
    half = deadline.fraction_of(0.5)
    assert 4.0 < half <= 5.0
    assert deadline.fraction_of(0.0) == 0.0


@pytest.mark.parametrize("share", [-0.1, 1.1, 2.0])
def test_fraction_of_refuses_a_share_outside_the_unit_interval(share: float) -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        Deadline.after(1.0).fraction_of(share)


def test_is_hashable_and_comparable_by_value() -> None:
    a, b = Deadline(100.0), Deadline(100.0)
    assert a == b
    assert len({a, b}) == 1
