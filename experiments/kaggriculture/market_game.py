"""The contested-product game: what the single-seller economics miss.

`economics.py` answers "what is a tile worth" assuming one seller. Melon comes out ~4x anything
else per tile, with revenue peaking well past the rate the town can absorb. But the market is
*shared*: a co-player selling the same product drains the same inventory and moves the same price,
so profitable over-production may not survive contact with a co-player doing the same thing.

That is a game, and it is small enough to solve exactly. Each player picks how many tiles to devote
to the contested product; both sell as harvested into one price curve; the payoff is each player's
season revenue from that line. The action set is a handful of allocations, so the normal form is
tiny and `causalrl.magames` applies directly:

* :class:`causalrl.AgentType` / :class:`causalrl.Population` materialise the payoff tables,
* :func:`causalrl.run_no_regret` plays the repeated game to a realised empirical joint, and
* :func:`causalrl.certify_cce_do` bounds a functional over the coarse-correlated-equilibrium
  polytope, with the run's measured regret as the finite-time epsilon.

The payoffs here are *known in closed form* — they come from the simulator's own published price
curve, not from sampling — which is exactly the condition `run_no_regret` needs.
"""

from __future__ import annotations

from economics import (
    MARKET_I0,
    MARKET_PARAMS,
    TILES_PER_UNIT,
    YIELD_PER_TILE_DAY,
    price,
    town_drain_per_day,
)

CONTESTED = "MELON"
ALTERNATIVE = "MILK"
"""What a tile earns if it is *not* growing the contested product — the next-best line."""

TOTAL_TILES = 100
"""Board size at full expansion. Allocation is a split, which is what makes a tile costly."""

ALLOCATIONS = (0, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100)
"""Tile counts each player may devote to the contested product, spanning the whole board.

Wide on purpose. A first version of this study stopped at 40 and the equilibrium sat exactly on
that bound — the sign of a grid truncating the answer rather than containing it. Widening it turned
out to expose a *modelling* error rather than a resolution one (see :func:`portfolio_revenue`), but
the check is the point: an equilibrium at the edge of its own grid has not been found, only capped.
"""

FIRST_YIELD_DAY = 10  # melon's first harvest
SEASON_END = 30


def joint_revenue(
    mine: int, theirs: int, *, item: str = CONTESTED, start_day: int = FIRST_YIELD_DAY
) -> tuple[float, float]:
    """Season revenue for each seller when both sell ``item`` as harvested into one curve.

    Both players' output hits the same inventory, so each unit either player sells depresses the
    price both of them face for every later unit. The town's drain is the only thing lifting it.
    """
    per_tile = YIELD_PER_TILE_DAY[item] / TILES_PER_UNIT[item]
    inventory = float(MARKET_I0)
    earned = [0.0, 0.0]
    for day in range(start_day, SEASON_END):
        output = (mine * per_tile, theirs * per_tile)
        unit_price = price(item, inventory)
        earned[0] += output[0] * unit_price
        earned[1] += output[1] * unit_price
        inventory = max(0.0, inventory + sum(output) - town_drain_per_day(item, day))
    return earned[0], earned[1]


def alternative_rate(item: str = ALTERNATIVE) -> float:
    """Revenue per tile per day from the next-best line, at its base price.

    Conservative on purpose: the alternatives the town drains faster than the board can supply
    (milk, wool, strawberry) actually sell *above* base all season, so costing a diverted tile at
    base understates what it gives up — the direction that makes the contested product look better,
    not worse.
    """
    return YIELD_PER_TILE_DAY[item] / TILES_PER_UNIT[item] * MARKET_PARAMS[item]["base"]


def portfolio_revenue(mine: int, theirs: int) -> tuple[float, float]:
    """Whole-farm revenue for each seller: contested tiles plus what the rest of the board earns.

    **A tile has to cost something.** Scoring the contested line alone makes every extra tile weakly
    profitable — even at the $1 price floor an extra unit still earns $1 — so the game degenerates
    and both players run to whatever bound the grid happens to impose. That is exactly what the
    first version of this study did. A tile devoted here is a tile not devoted to
    :data:`ALTERNATIVE`, and pricing that forgone revenue is what makes the trade-off real.
    """
    contested = joint_revenue(mine, theirs)
    days = SEASON_END - FIRST_YIELD_DAY
    rate = alternative_rate()
    return (
        contested[0] + (TOTAL_TILES - mine) * rate * days,
        contested[1] + (TOTAL_TILES - theirs) * rate * days,
    )


def payoff(own: int, others: tuple[int, ...], params: object = None) -> float:
    """``AgentType`` payoff: this seller's whole-farm revenue in thousands."""
    return portfolio_revenue(own, others[0])[0] / 1000.0


def build_population():
    """A two-seller :class:`~causalrl.magames.population.Population` over :data:`ALLOCATIONS`."""
    from causalrl import AgentType, Population

    seller = AgentType(name="seller", actions=ALLOCATIONS, payoff=payoff)
    return Population(agents=("P0", "P1"), types={"P0": seller, "P1": seller})
