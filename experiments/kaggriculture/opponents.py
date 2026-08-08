"""A reference pool of distinct strategies, so "better" means something.

`starter` is a one-tile syntax demo, so beating it by 10x measures almost nothing. Without access
to the real ladder (kaggle.com is unreachable from here) the honest substitute is a pool of
*coherent, different* strategies to play against — and, better than a round-robin table, the
equilibrium over that pool, which says which strategy is robust to its opponent rather than which
happens to win the most cells.

**The limitation, stated up front:** every bot here shares `agent.py`'s routing and market
plumbing, and differs in *portfolio and selling policy* — which is genuinely where this game's
strategy lives, but does mean the pool cannot tell us anything about whether that shared
implementation is any good. A real competitor might route far better and beat all of these. This
measures strategy, not craft.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import agent as base


@dataclass(frozen=True)
class Strategy:
    """A named portfolio-and-selling policy, applied by swapping `agent.py`'s module knobs."""

    name: str
    melon_tiles: int
    animal_target: int
    quadrants: int
    hands: int
    animal: str = "COW"
    sell_meter: int = 0
    """Units of a premium product held back per turn, 0 to sell everything as harvested.

    The one structurally different lever in the pool. Melon's only drain is the town centre at
    1 unit/day (FINDINGS §10), so a burst harvest walks the price down its `sq` curve; a metered
    seller trades immediacy for a better average price. Whether that pays is the question.
    """

    def build(self):
        """A `kaggle_environments` agent function that applies this strategy for its turn."""
        knobs = {
            "MELON_TILES": self.melon_tiles,
            "ANIMAL_TARGET": self.animal_target,
            "MAX_QUADRANTS": self.quadrants,
            "MAX_HANDS": self.hands,
            "ANIMAL": self.animal,
            "SELL_METER": self.sell_meter,
        }

        def play(obs: Any, config: Any = None) -> dict[str, Any]:
            saved = {k: getattr(base, k, None) for k in knobs}
            for k, v in knobs.items():
                setattr(base, k, v)
            try:
                return base.agent(obs, config)
            finally:
                for k, v in saved.items():
                    if v is not None:
                        setattr(base, k, v)

        play.__name__ = self.name
        return play


POOL = (
    # The tuned agent: contested-line cap from the equilibrium, six cows, two quadrants.
    Strategy("balanced", melon_tiles=10, animal_target=6, quadrants=2, hands=7),
    # All-in on the highest $/tile/day line, ignoring that it is the one product that saturates.
    Strategy("melon_rush", melon_tiles=30, animal_target=0, quadrants=3, hands=9),
    # No melon at all: volume in the fast-turnaround crop, maximum land and labour.
    Strategy("volume", melon_tiles=0, animal_target=0, quadrants=4, hands=10),
    # Livestock-led: the lines the town under-supplies, run as hard as the feed allows.
    Strategy("rancher", melon_tiles=4, animal_target=9, quadrants=3, hands=9),
    # The balanced portfolio, but metering premium sales instead of dumping them.
    Strategy("metered", melon_tiles=10, animal_target=6, quadrants=2, hands=7, sell_meter=6),
    # A deliberately small, cheap farm: one quadrant, few hands, no livestock overhead.
    Strategy("smallholder", melon_tiles=8, animal_target=2, quadrants=1, hands=4),
)

BY_NAME = {s.name: s for s in POOL}


def payoff_matrix(seeds: range = range(3)) -> dict[tuple[str, str], float]:
    """Mean bank for the row strategy when it plays the column strategy, over ``seeds``.

    Both seats are played for every ordered pair, so a strategy is never scored only from the seat
    the schedule happened to give it.
    """
    from harness import run_episode

    built = {s.name: s.build() for s in POOL}
    out: dict[tuple[str, str], float] = {}
    for row in POOL:
        for col in POOL:
            total = 0.0
            for seed in seeds:
                total += run_episode(built[row.name], built[col.name], seed=seed)[0]
            out[(row.name, col.name)] = total / len(seeds)
    return out
