"""The no-regret population that produces the empirical joint `cce_regret` / `certify_cce_do` eat.

The point of these tests is that the learners *learn*: the measured regret of the realized joint has
to fall as the run goes on and end far below what a population that ignores its payoffs reaches. A
test that only asserted the run terminated, or that regret was finite, would pass against a learner
that plays uniformly at random forever, so every convergence assertion here is written against that
null — spelled `explore=1.0`, which replaces each learner's distribution with the uniform one and so
runs the identical loop with the payoffs disconnected.

Two anchor games, because no single one pins both halves of the claim:

* ``_DOMINANT`` — action 1 strictly dominates, so the CCE set is the single profile ``(1, 1)`` and
  payoff-blind play sits a measured 0.75 away from it. This is where "beats the null" is sharp.
* ``_ZERO_SUM`` — an asymmetric matching-pennies with **no pure equilibrium**, so every short-run
  empirical joint has regret bounded well away from zero and the approach to the CCE set is gradual.
  This is where "falls with the horizon" is sharp.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from causalrl.agents.no_regret import MultiplicativeWeights, NoRegretLearner, RegretMatching
from causalrl.certify.certificate import Kind
from causalrl.games import CausalGame
from causalrl.magames import (
    NoRegretRun,
    cce_bounds,
    cce_polytope,
    cce_regret,
    certify_cce_do,
    run_no_regret,
)
from causalrl.magames.population import AgentType, Population

ALGORITHMS = ("regret_matching", "multiplicative_weights")

_DOMINANT = {(0, 0): 3.0, (0, 1): 0.0, (1, 0): 5.0, (1, 1): 1.0}  # action 1 strictly dominant
_ANTI = {(0, 0): 0.0, (0, 1): 7.0, (1, 0): 2.0, (1, 1): 6.0}  # anti-coordination ("chicken")
_ZERO_SUM = np.array([[2.0, -1.0], [-1.0, 1.0]])  # row's payoff; no pure equilibrium


def _symmetric_population(table: Mapping[tuple[int, int], float]) -> Population:
    def payoff(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return table[(own, others[0])]

    agent_type = AgentType(name="sym", actions=(0, 1), payoff=payoff)
    return Population(agents=("A1", "A2"), types={"A1": agent_type, "A2": agent_type})


def _zero_sum_population() -> Population:
    def row(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return float(_ZERO_SUM[own, others[0]])

    def column(own: int, others: tuple[int, ...], params: Mapping[str, float]) -> float:
        return -float(_ZERO_SUM[others[0], own])

    return Population(
        agents=("A1", "A2"),
        types={
            "A1": AgentType(name="row", actions=(0, 1), payoff=row),
            "A2": AgentType(name="col", actions=(0, 1), payoff=column),
        },
    )


def _uniform_regret(game: CausalGame, do: Mapping[str, int] | None = None) -> float:
    """The measured regret of the joint a payoff-blind population converges to."""
    profiles = cce_polytope(game, do=do).profiles
    return cce_regret(game, {p: 1.0 / len(profiles) for p in profiles}, do=do)


def _welfare(game: CausalGame):
    def functional(profile: Mapping[str, int]) -> float:
        joint = tuple(profile[a] for a in game.agents)
        return sum(game.utilities[a][joint] for a in game.agents)

    return functional


# --- the test that matters: regret falls, and beats a payoff-blind population -------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_learned_regret_is_far_below_a_payoff_blind_population(algorithm: str) -> None:
    """`explore=1.0` runs the same loop with the payoffs disconnected; learning must beat it."""
    population = _symmetric_population(_DOMINANT)
    game = population.to_game()
    assert _uniform_regret(game) == pytest.approx(0.75)  # where payoff-blind play lands

    blind = run_no_regret(population, 2_000, algorithm=algorithm, explore=1.0, seed=0)
    learned = run_no_regret(population, 4_000, algorithm=algorithm, seed=0)

    assert blind.regret > 0.7  # the null really does sit at the uniform joint
    assert learned.regret < 0.1 * blind.regret
    assert learned.regret < 0.05  # the CCE of this game is a point and the run reaches it
    assert learned.empirical_joint[(1, 1)] > 0.9  # nearly all mass on the dominant profile (0.25
    #                                               is what the payoff-blind null puts there)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_regret_falls_materially_with_the_horizon(algorithm: str) -> None:
    """On a game with no pure equilibrium the approach to the CCE set is gradual and measurable."""
    population = _zero_sum_population()
    blind = _uniform_regret(population.to_game())  # exactly where payoff-blind play converges
    short = run_no_regret(population, 20, algorithm=algorithm, seed=0)
    long = run_no_regret(population, 5_000, algorithm=algorithm, seed=0)

    assert blind == pytest.approx(0.25)
    assert short.regret > 0.1  # 20 rounds is nowhere near the equilibrium
    assert long.regret < 0.3 * short.regret
    assert long.regret < 0.25 * blind


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_regret_trace_records_the_fall(algorithm: str) -> None:
    """The trace is the falling curve, not just its endpoints."""
    run = run_no_regret(_zero_sum_population(), 5_000, algorithm=algorithm, seed=1)
    horizons = [t for t, _ in run.regret_trace]
    values = [r for _, r in run.regret_trace]
    assert horizons == sorted(horizons) and horizons[-1] == 5_000
    assert len(values) >= 5
    assert values[0] >= 2.0  # a single round is a point mass, and none of them is an equilibrium
    # 0.1 * values[0] = 0.2 sits *below* the 0.25 a payoff-blind population's time average reaches,
    # so this threshold is not met by merely averaging away the first round's point mass.
    assert values[-1] < 0.1 * values[0]
    assert values[-1] == run.regret


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_regret_falls_under_an_intervention(algorithm: str) -> None:
    """With a co-player pinned, the free learner must find its best response to that action."""
    population = _symmetric_population(_DOMINANT)
    game = population.to_game()
    do = {"A2": 0}
    blind = _uniform_regret(game, do=do)  # ignoring payoffs is measurably wrong here too
    assert blind == pytest.approx(1.0)

    run = run_no_regret(population, 3_000, do=do, algorithm=algorithm, seed=2)
    assert run.regret < 0.1 * blind
    assert run.empirical_joint[(1, 0)] > 0.95  # the best response to the pinned action pays 5
    assert all(profile[1] == 0 for profile in run.profiles)


# --- the seam: the run is exactly what the certificate layer already accepts --------------------


def test_run_feeds_cce_regret_in_both_accepted_forms() -> None:
    population = _symmetric_population(_ANTI)
    game = population.to_game()
    run = run_no_regret(population, 2_000, seed=3)
    assert cce_regret(game, run.empirical_joint) == pytest.approx(run.regret)
    assert cce_regret(game, run.weights) == pytest.approx(run.regret)  # cce_polytope profile order
    assert run.weights.sum() == pytest.approx(1.0)
    assert run.profiles == cce_polytope(game).profiles
    assert run.agents == game.agents and run.rounds == 2_000 and run.do == {}


def test_measured_regret_certificate_contains_the_realized_average() -> None:
    """The finite-time route (T2): the measured-epsilon interval brackets what actually happened."""
    population = _symmetric_population(_ANTI)
    game = population.to_game()
    run = run_no_regret(population, 4_000, seed=4)
    welfare = _welfare(game)

    realized = sum(
        weight * welfare(dict(zip(game.agents, profile, strict=True)))
        for profile, weight in run.empirical_joint.items()
    )
    interval = cce_bounds(game, welfare, epsilon=run.regret)
    assert interval.lower - 1e-9 <= realized <= interval.upper + 1e-9

    cert = certify_cce_do(game, welfare, no_regret=False, epsilon=run.regret)
    assert cert.kind is Kind.BOUNDED
    assert cert.witness is not None and cert.witness.detail["epsilon"] == pytest.approx(run.regret)


def test_run_is_reproducible_and_seed_sensitive() -> None:
    population = _zero_sum_population()
    first = run_no_regret(population, 400, seed=7)
    same = run_no_regret(population, 400, seed=7)
    other = run_no_regret(population, 400, seed=8)
    assert np.allclose(first.weights, same.weights)
    assert not np.allclose(first.weights, other.weights)


def test_accepts_a_game_directly_as_well_as_a_population() -> None:
    population = _symmetric_population(_DOMINANT)
    from_population = run_no_regret(population, 200, seed=5)
    from_game = run_no_regret(population.to_game(), 200, seed=5)
    assert np.allclose(from_population.weights, from_game.weights)
    assert isinstance(from_game, NoRegretRun)


def test_fully_pinned_population_has_no_free_learner() -> None:
    run = run_no_regret(_symmetric_population(_DOMINANT), 10, do={"A1": 0, "A2": 1})
    assert run.empirical_joint == {(0, 1): 1.0}
    assert run.regret == 0.0  # no free agent contributes a deviation constraint


def test_run_validates_its_arguments() -> None:
    population = _symmetric_population(_DOMINANT)
    with pytest.raises(ValueError, match="rounds must be at least 1"):
        run_no_regret(population, 0)
    with pytest.raises(ValueError, match="unknown algorithm"):
        run_no_regret(population, 5, algorithm="bandit")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="parameter-free"):
        run_no_regret(population, 5, algorithm="regret_matching", learning_rate=0.1)
    with pytest.raises(KeyError, match="unknown agent"):
        run_no_regret(population, 5, do={"nope": 0})


def test_trace_points_are_configurable_and_degenerate_horizons_work() -> None:
    single = run_no_regret(_symmetric_population(_DOMINANT), 1)
    assert single.regret_trace == ((1, single.regret),)
    coarse = run_no_regret(_symmetric_population(_DOMINANT), 100, trace_points=0)
    assert coarse.regret_trace == ((100, coarse.regret),)


def test_top_level_exports() -> None:
    import causalrl

    for name in ("run_no_regret", "NoRegretRun", "NoRegretLearner", "RegretMatching"):
        assert name in causalrl.__all__
        assert getattr(causalrl, name) is not None


# --- the learners themselves --------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_learner_concentrates_on_the_best_action_under_full_information(algorithm: str) -> None:
    """Against a stationary payoff vector the strategy must move onto the best action."""
    payoffs = np.array([0.0, 1.0, 0.2])
    learner: NoRegretLearner = (
        RegretMatching(3, seed=0)
        if algorithm == "regret_matching"
        else MultiplicativeWeights(3, learning_rate=0.5, seed=0)
    )
    assert learner.distribution() == pytest.approx(np.full(3, 1 / 3))  # uniform before any signal
    for _ in range(200):
        learner.observe(payoffs)
    assert learner.rounds == 200
    assert learner.distribution()[1] > 0.9


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_learner_concentrates_under_bandit_feedback(algorithm: str) -> None:
    """`Agent.update` sees only the played arm's reward; IPS weighting still finds the best one."""
    means = np.array([0.1, 0.8, 0.2])
    rng = np.random.default_rng(0)
    learner: NoRegretLearner = (
        RegretMatching(3, explore=0.1, seed=1)
        if algorithm == "regret_matching"
        else MultiplicativeWeights(3, learning_rate=0.02, explore=0.1, seed=1)
    )
    pulls = np.zeros(3)
    for _ in range(3_000):
        action = learner.act({})
        pulls[action] += 1
        learner.update({}, action, float(means[action] + 0.05 * rng.standard_normal()))
    assert learner.distribution()[1] > 0.7  # the best arm, never having seen the others' payoffs
    assert pulls[1] / pulls.sum() > 0.6


def test_explore_floors_every_action_probability() -> None:
    learner = RegretMatching(4, explore=0.4, seed=0)
    learner.observe(np.array([0.0, 0.0, 0.0, 10.0]))
    assert learner.distribution().min() >= 0.4 / 4 - 1e-12
    assert learner.distribution()[3] == pytest.approx(0.7)


def test_multiplicative_weights_default_rates() -> None:
    tuned = MultiplicativeWeights(2, horizon=100, payoff_range=2.0)
    assert tuned._rate() == pytest.approx(np.sqrt(8.0 * np.log(2) / 100) / 2.0)
    anytime = MultiplicativeWeights(2)
    first = anytime._rate()
    anytime.observe(np.array([1.0, 0.0]))
    anytime.observe(np.array([1.0, 0.0]))
    assert anytime._rate() < first  # the anytime rate decays with the rounds seen


def test_learner_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_actions"):
        RegretMatching(0)
    with pytest.raises(ValueError, match="explore"):
        RegretMatching(2, explore=1.5)
    with pytest.raises(ValueError, match="learning_rate"):
        MultiplicativeWeights(2, learning_rate=0.0)
    with pytest.raises(ValueError, match="horizon"):
        MultiplicativeWeights(2, horizon=0)
    with pytest.raises(ValueError, match="payoff_range"):
        MultiplicativeWeights(2, payoff_range=0.0)
    learner = RegretMatching(2, seed=0)
    with pytest.raises(ValueError, match="shape"):
        learner.observe(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="outside"):
        learner.update({}, 5, 1.0)
