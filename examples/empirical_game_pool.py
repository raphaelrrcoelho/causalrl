"""EmpiricalGame — solving a game whose payoff table you had to *measure*.

The equilibrium machinery in :mod:`causalrl.magames` reads a payoff table. Sometimes that table is
given (a closed-form payoff function); often it is measured — pick a pool of candidate strategies,
play every pairing some number of times, average. This example does the measured case end to end and
shows the two things that go wrong if you forget the table is an estimate.

1. **A certificate off a thin round-robin should not read like a certificate off the game.**
   :class:`~causalrl.magames.EmpiricalGame` keeps each cell's standard error and hands
   :func:`~causalrl.magames.certify_cce_do` a :class:`~causalrl.magames.PayoffError`; the reported
   ``value`` is the partial-identification region of the table you measured, while ``ci`` is what
   also survives the measurement. Ten matches per cell abstains where two hundred certifies.

2. **An equilibrium sitting on the edge of its own action grid has not been found, only capped.**
   :meth:`~causalrl.magames.NoRegretRun.boundary_mass` is the check, and the fix is a wider grid,
   not a longer run.

Run:  python examples/empirical_game_pool.py
"""

from __future__ import annotations

import numpy as np

from causalrl import AgentType, EmpiricalGame, Population, certify_cce_do, run_no_regret

# A pool of policies for one shared, depletable resource, each labelled by how hard it draws.
POOL = ("cautious", "steady", "eager", "greedy")
DRAW = {"cautious": 0.2, "steady": 0.4, "eager": 0.6, "greedy": 0.8}
NOISE = 0.05


def _round(first: float, second: float) -> tuple[float, float]:
    """One match: what each side draws is worth less the more the two of them draw together."""
    left = first * (1.0 - 0.5 * (first + second))
    right = second * (1.0 - 0.5 * (first + second))
    return left, right


def round_robin(replications: int, rng: np.random.Generator) -> dict[tuple[str, ...], list]:
    """Play every pairing ``replications`` times, keeping the individual matches, not the mean."""
    return {
        (row, col): [
            tuple(v + rng.normal(0.0, NOISE) for v in _round(DRAW[row], DRAW[col]))
            for _ in range(replications)
        ]
        for row in POOL
        for col in POOL
    }


def solve(replications: int, rng: np.random.Generator) -> None:
    """Measure the pool, learn on it, and certify — reporting what the measurement cost."""
    empirical = EmpiricalGame.from_samples(POOL, round_robin(replications, rng))
    game = empirical.to_game()
    run = run_no_regret(game, 20_000, seed=0)
    error = empirical.payoff_error()  # 95%, Bonferroni over every cell of the table

    played = run.marginal("A1")
    weights = {POOL[action]: weight for action, weight in played.items() if weight > 1e-3}
    certificate = certify_cce_do(
        game,
        lambda profile: game.utilities["A1"][(profile["A1"], profile["A2"])],
        no_regret=False,
        epsilon=run.regret,
        payoff_error=error,
    )

    print(f"\n{replications} matches per pairing")
    print(f"  equilibrium play   {weights}")
    print(f"  measured regret    {run.regret:.4f}")
    print(f"  payoff error       +/-{error.utility:.3f} per cell (alpha={error.alpha})")
    print(f"  {certificate}")


def grid_truncation() -> None:
    """The same contested draw as a quantity, solved on a grid that does and does not contain it."""

    def payoff(own: int, others: tuple[int, ...], params: object = None) -> float:
        return _round(own / 100.0, others[0] / 100.0)[0]

    for label, actions in (("0-30", (0, 10, 20, 30)), ("0-90", tuple(range(0, 100, 10)))):
        agent = AgentType(name="drawer", actions=actions, payoff=payoff)
        population = Population(agents=("A1", "A2"), types={"A1": agent, "A2": agent})
        run = run_no_regret(population, 20_000, seed=0)
        best = max(run.marginal("A1").items(), key=lambda kv: kv[1])
        print(
            f"  grid {label:5s} -> settles at {best[0]:3d} (weight {best[1]:.2f}), "
            f"boundary mass {run.boundary_mass('A1'):.2f}"
        )


def main() -> None:
    rng = np.random.default_rng(0)
    print("Solving a game whose payoffs had to be measured\n" + "-" * 47)
    solve(10, rng)
    solve(200, rng)

    print("\nIs the answer inside its own action grid?\n" + "-" * 41)
    grid_truncation()
    print(
        "\n  Mass on the edge means the grid stopped before the game did: the '0-30' answer is\n"
        "  the grid's maximum, not the equilibrium the '0-90' grid finds."
    )


if __name__ == "__main__":
    main()
