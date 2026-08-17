"""The measured game: payoffs that came from finite simulation, and the error that came with them.

`Population` builds a game from a payoff *function* — exact by construction. The other way a finite
game arrives is a round-robin: play every profile of a strategy pool some number of times and
average. That table is an estimate, and the whole point of :class:`EmpiricalGame` is that it refuses
to forget so: it carries the standard error alongside the mean and hands the certificate layer a
:class:`~causalrl.magames.cce.PayoffError` derived from it.

The tests below pin three things: the table is materialised in the strategy order given (so the
integer actions the rest of the module speaks are the pool's own order), the reported error falls
like ``1/sqrt(n)``, and a single replication per cell is refused outright — one sample is a number
with no error bar, and this object exists to carry the error bar.
"""

from __future__ import annotations

import numpy as np
import pytest

from causalrl.magames.cce import PayoffError
from causalrl.magames.empirical import EmpiricalGame

STRATEGIES = ("hold", "push", "mirror")


def _samples(
    rng: np.random.Generator, replications: int, noise: float = 1.0
) -> dict[tuple[str, ...], list[tuple[float, ...]]]:
    """A full round-robin over :data:`STRATEGIES` with a known mean and Gaussian noise on top.

    The row player's true mean is ``10 * row_index - col_index``; the column player's is its mirror,
    so the game is symmetric in the roles and asymmetric in the payoffs.
    """
    out: dict[tuple[str, ...], list[tuple[float, ...]]] = {}
    for i, row in enumerate(STRATEGIES):
        for j, col in enumerate(STRATEGIES):
            truth = (10.0 * i - j, 10.0 * j - i)
            out[(row, col)] = [
                (truth[0] + noise * rng.standard_normal(), truth[1] + noise * rng.standard_normal())
                for _ in range(replications)
            ]
    return out


def _game(replications: int = 40, noise: float = 1.0, seed: int = 0) -> EmpiricalGame:
    return EmpiricalGame.from_samples(
        STRATEGIES, _samples(np.random.default_rng(seed), replications, noise)
    )


def test_from_samples_averages_the_replications_and_keeps_their_error() -> None:
    """The mean is the estimate; the standard error is what stops it being read as exact."""
    empirical = _game(replications=200, noise=1.0, seed=0)

    assert empirical.strategies == STRATEGIES
    assert empirical.means[("push", "mirror")][0] == pytest.approx(8.0, abs=0.3)  # 10*1 - 2
    assert empirical.means[("push", "mirror")][1] == pytest.approx(19.0, abs=0.3)  # 10*2 - 1
    assert empirical.replications[("push", "mirror")] == 200
    assert empirical.stderrs[("push", "mirror")][0] == pytest.approx(1.0 / np.sqrt(200), rel=0.25)


def test_to_game_speaks_integer_actions_in_the_strategy_order_given() -> None:
    """The pool's order *is* the action encoding, so a marginal can be read back by name."""
    game = _game().to_game()

    assert game.agents == ("A1", "A2")
    assert game.actions["A1"] == (0, 1, 2)
    assert game.utilities["A1"][(1, 2)] == pytest.approx(8.0, abs=0.5)
    assert game.utilities["A2"][(1, 2)] == pytest.approx(19.0, abs=0.5)


def test_payoff_error_falls_like_one_over_root_n() -> None:
    """More replications, a tighter certificate — the reason to record ``n`` at all."""
    few = _game(replications=25, seed=1).payoff_error()
    many = _game(replications=400, seed=1).payoff_error()

    assert isinstance(few, PayoffError)
    assert few.utility > 0.0
    ratio = few.utility / many.utility
    assert 3.0 < ratio < 5.5  # 4x on the nose for a 16x sample, up to sampling slop in the SEs


def test_payoff_error_carries_the_level_it_was_computed_at() -> None:
    """A looser level buys a smaller slack; the level travels with the number that used it."""
    tight = _game(seed=2).payoff_error(alpha=0.01)
    loose = _game(seed=2).payoff_error(alpha=0.20)

    assert tight.alpha == 0.01
    assert loose.alpha == 0.20
    assert tight.utility > loose.utility


def test_functional_terms_scales_the_functional_error() -> None:
    """A functional summing k estimated payoffs carries k times the error of one; 0 carries none."""
    empirical = _game(seed=3)

    assert empirical.payoff_error(functional_terms=0).functional == 0.0
    assert empirical.payoff_error(functional_terms=1).functional == pytest.approx(
        empirical.payoff_error(functional_terms=1).utility
    )
    assert empirical.payoff_error(functional_terms=2).functional == pytest.approx(
        2.0 * empirical.payoff_error(functional_terms=1).utility
    )


def test_a_single_replication_per_cell_is_refused() -> None:
    """One sample has no error bar, and an object whose job is the error bar must not invent one."""
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match="at least 2"):
        EmpiricalGame.from_samples(STRATEGIES, _samples(rng, replications=1))


def test_an_incomplete_round_robin_is_refused_by_name() -> None:
    """A normal form needs every cell; a missing pairing is a hole, not a zero."""
    samples = _samples(np.random.default_rng(5), replications=4)
    del samples[("push", "mirror")]

    with pytest.raises(ValueError, match=r"push.*mirror"):
        EmpiricalGame.from_samples(STRATEGIES, samples)


def test_ragged_payoff_vectors_are_refused() -> None:
    """Every replication reports one payoff per agent, or the table is not a game."""
    samples = _samples(np.random.default_rng(6), replications=4)
    samples[("hold", "hold")] = [(1.0, 2.0), (1.0,)]  # type: ignore[list-item]

    with pytest.raises(ValueError, match="payoff"):
        EmpiricalGame.from_samples(STRATEGIES, samples)


def test_duplicate_strategy_labels_are_refused() -> None:
    """Labels are the action encoding, so two of the same name is an ambiguous game."""
    with pytest.raises(ValueError, match="duplicate"):
        EmpiricalGame.from_samples(("hold", "hold"), {})


def test_top_level_exports() -> None:
    import causalrl

    for name in ("EmpiricalGame", "PayoffError"):
        assert name in causalrl.__all__
        assert getattr(causalrl, name) is not None
