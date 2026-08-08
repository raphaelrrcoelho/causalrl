"""A submittable agent, driven by the economics rather than a fixed script.

The shipped `starter` agent farms a single tile and finishes the season with $3,510 against a
$3,000 stake -- $510 of profit in thirty days. Everything below follows from `economics.py`:

* **Land binds, saturation does not** (FINDINGS §1), so buy every quadrant as soon as it is
  affordable and fill it. Only melon can be over-supplied.
* **Rank tiles by revenue at base price** (§2). Melon is worth ~4x anything else.
* **The contested line is capped by the game, not the town** (§6): the two-seller equilibrium is
  ~10 melon tiles, half what a lone seller would plant.
* **Labour is cheap and Fibonacci-priced** (§5), so hire until the marginal hand stops paying.

The structure is a task queue, not a state machine: every turn each tile is scored for what it
needs, units are matched to the nearest work, and whatever is left over plants. That keeps the
per-turn cost linear in tiles and well inside the 1-second budget.
"""

from __future__ import annotations

from typing import Any

CROP_SPEC = {
    "WHEAT": {"seed": 10, "first": 2, "max_day": 4, "ongoing": False},
    "CARROT": {"seed": 20, "first": 2, "max_day": 3, "ongoing": False},
    "MELON": {"seed": 80, "first": 10, "max_day": 12, "ongoing": False},
    "TOMATO": {"seed": 50, "first": 8, "max_day": 8, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first": 10, "max_day": 10, "ongoing": True},
}

LAND_COST = {1: 1000, 2: 2000, 3: 4000}
MAX_QUADRANTS = 2
"""How many quadrants to own -- measured, and it contradicts the paper economics.

FINDINGS §1 concluded that land binds and should all be bought: town demand outruns a full board
several times over. Playing it says otherwise. Measured against `starter`, 5 seeds each:

    1 quadrant  (25 tiles)  $31,298
    2 quadrants (50 tiles)  $31,793   <- best
    3 quadrants (75 tiles)  $29,433
    4 quadrants (100 tiles) $24,970

Land is only worth what the labour can service, and hands are priced on a Fibonacci curve, so the
farm cannot staff a full board. Tiles bought past that point are dead capital twice over -- $7,000
spent, and the unwatered plants on them turn to weeds that then cost actions to dig out.
"""
QUADRANT_ORDER = ("NE", "SW", "SE")

MELON_TILES = 10
"""Contested-line cap, from the two-seller equilibrium in FINDINGS §6 -- not the solo optimum.

A lone seller maximises melon revenue at ~20 tiles. Against a co-player who also grows melon that
is a $4,713 mistake, and 10 is the interior equilibrium both grids agree on.
"""

CASH_FLOOR = 0
"""Working capital never spent on seed or land.

The first version had none and starved: it sank the whole $3,000 stake into melon seed and land on
day 0, hit $0 by day 8, and could then buy no seed at all -- so tiles sat empty, income never
recovered, and the farm ran at under half utilisation for the rest of the season. Melon pays 10
days after planting; a farm that cannot fund the wait does not get to collect.
"""

MELON_CAPITAL = 0
"""Bank needed before melon is worth starting, on top of :data:`CASH_FLOOR`.

Melon is worth ~5x a carrot tile per day (FINDINGS §2) but yields nothing for ten days, while carrot
turns over every three. Early money buys cashflow; melon is what that cashflow is *for*.
"""

SEED_SHARE = 1.0
"""Fraction of the surplus above :data:`CASH_FLOOR` that may go on seed in one turn.

Spending all of it pins the bank at the floor and the farm never affords a land tier -- measured:
one quadrant bought all season and $2,079 final, against $25,202 for a version with no floor at
all. The share is what lets income outrun expenditure.
"""

MAX_HANDS = 7
SELL_RESERVE = {"WHEAT": 12}
"""Wheat held back rather than sold: it is animal feed first and a cash crop a distant second."""


def _half(board: int) -> int:
    return board // 2


def _quadrant(x: int, y: int, board: int) -> str:
    h = _half(board)
    return ("NW" if x < h else "NE") if y < h else ("SW" if x < h else "SE")


def _plantable(day: int, crop: str, season: int = 30) -> bool:
    """Whether a crop planted today can still reach its first yield before the season ends."""
    return day + CROP_SPEC[crop]["first"] < season


def _crop_for(day: int, melon_planted: int, money: float) -> str | None:
    """What to plant now, given the bank.

    Melon only once the farm can afford to wait ten days for it; carrot otherwise, because a
    three-day turnaround is what funds everything else. Falls through to wheat when even carrot can
    no longer reach a yield before the season ends.
    """
    if (
        melon_planted < MELON_TILES
        and _plantable(day, "MELON")
        and money >= MELON_CAPITAL + CASH_FLOOR
    ):
        return "MELON"
    for crop in ("CARROT", "WHEAT"):
        if _plantable(day, crop):
            return crop
    return None


def _tile_tasks(
    farm: dict[str, Any], day: int, board: int
) -> list[tuple[int, tuple[int, int], str]]:
    """Every tile wanting attention, as ``(priority, (x, y), op)``; lower priority runs first."""
    tasks: list[tuple[int, tuple[int, int], str]] = []
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if tile == "LOCKED" or tile is None:
                continue
            kind = tile.get("kind")
            if kind == "WEED":
                tasks.append((3, (x, y), "DIG"))
            elif kind == "PLANT":
                spec = CROP_SPEC.get(tile["crop"])
                age = day - tile["planted_day"]
                ready = tile.get("yield_units", 0) > 0 and (
                    spec is None or spec["ongoing"] or age >= spec["max_day"]
                )
                if ready:
                    tasks.append((0, (x, y), "HARVEST"))
                elif not tile.get("watered_today"):
                    tasks.append((1, (x, y), "WATER"))
            elif kind in ("COOP", "PASTURE") and tile.get("animal"):
                if not tile.get("fed_today"):
                    tasks.append((1, (x, y), "FEED"))
                elif tile.get("yield_units", 0) > 0:
                    tasks.append((0, (x, y), "HARVEST"))
    return tasks


def units_positions(farm: dict[str, Any]) -> list[tuple[int, int]]:
    """The farmer plus every hand hired today, in the order the action dict expects."""
    return [tuple(farm["farmer"])] + [tuple(h) for h in farm.get("hands", [])]


def _empty_tiles(farm: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        (x, y) for y, row in enumerate(farm["tiles"]) for x, tile in enumerate(row) if tile is None
    ]


def _step_toward(at: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    """One move along the Manhattan path, or the empty list when already there."""
    ax, ay = at
    gx, gy = goal
    if ax != gx:
        return ["EAST"] if gx > ax else ["WEST"]
    if ay != gy:
        return ["SOUTH"] if gy > ay else ["NORTH"]
    return []


def _assign(
    units: list[tuple[int, int]],
    tasks: list[tuple[int, tuple[int, int], str]],
    plantable: list[tuple[int, int]],
    crop: str | None,
    seeds_available: int,
) -> list[list[str]]:
    """Greedy nearest-work matching: each unit takes the closest unclaimed task, then plants."""
    ordered = sorted(tasks, key=lambda t: t[0])
    claimed: set[tuple[int, int]] = set()
    actions: list[list[str]] = []
    seeds_left = seeds_available
    for unit in units:
        target: tuple[tuple[int, int], str] | None = None
        best = None
        for _priority, cell, op in ordered:
            if cell in claimed:
                continue
            d = abs(cell[0] - unit[0]) + abs(cell[1] - unit[1])
            if best is None or d < best:
                best, target = d, (cell, op)
            if d == 0:
                break
        if target is None and crop is not None and seeds_left > 0:
            free = [c for c in plantable if c not in claimed]
            if free:
                cell = min(free, key=lambda c: abs(c[0] - unit[0]) + abs(c[1] - unit[1]))
                target = (cell, "PLANT")
                seeds_left -= 1
        if target is None:
            actions.append(["PASS"])
            continue
        cell, op = target
        claimed.add(cell)
        move = _step_toward(unit, cell)
        if move:
            actions.append(move)
        elif op == "PLANT":
            actions.append(["PLANT", crop])
        else:
            actions.append([op])
    return actions


def act(obs: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
    """One turn: hire, buy, assign every unit to its nearest work, and sell what is in the shed."""
    farms = obs.get("farms") or []
    player = obs.get("player", 0)
    if player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    private = obs.get("private") or {}
    seeds = private.get("seeds", {}) or {}
    shed = private.get("shed", {}) or {}
    day, hour = obs.get("day", 0), obs.get("hour", 0)
    money = farm["money"]
    board = len(farm["tiles"])

    market: list[list[Any]] = []

    # --- land: the binding resource, bought as soon as each tier is affordable -----------------
    owned = len(farm.get("unlocked_quadrants", ["NW"]))
    cost = LAND_COST.get(owned) if owned < MAX_QUADRANTS else None
    # Land is the binding resource, so buy the moment the bank clears the price plus working
    # capital -- but never at the cost of the seed budget, which is what starves the farm.
    if cost is not None and money >= cost + CASH_FLOOR + 600:
        market.append(["BUY_LAND", QUADRANT_ORDER[owned - 1], 1])
        money -= cost

    # --- labour: cheap, Fibonacci-priced, and only useful for a day ----------------------------
    if hour == 0:
        want = min(MAX_HANDS, max(2, len(_empty_tiles(farm)) // 6 + 2))
        for _ in range(want):
            market.append(["HIRE"])

    # --- sell: land binds rather than demand, so hold nothing back except feed wheat -----------
    for item, count in sorted(shed.items()):
        keep = SELL_RESERVE.get(item, 0)
        if count > keep and item in {
            "WHEAT",
            "CARROT",
            "TOMATO",
            "STRAWBERRY",
            "MELON",
            "EGG",
            "MILK",
            "WOOL",
            "FERTILIZER",
        }:
            market.append(["SELL", item, count - keep])

    # --- seeds: keep enough to fill the free tiles this turn's units can reach ------------------
    melon_planted = sum(
        1
        for row in farm["tiles"]
        for t in row
        if isinstance(t, dict) and t.get("kind") == "PLANT" and t.get("crop") == "MELON"
    )
    crop = _crop_for(day, melon_planted, money)
    free = _empty_tiles(farm)
    if crop is not None and free:
        # Enough seed to keep every unit planting, but spend only a SHARE of the surplus: buying
        # seed with every last dollar pins the bank at the floor forever, so the farm can never
        # accumulate the $1k-$4k a land tier costs and stays locked in one quadrant all season.
        want = min(len(free), max(4, len(units_positions(farm))))
        have = seeds.get(crop, 0)
        seed_cost = CROP_SPEC[crop]["seed"]
        budget = max(0.0, money - CASH_FLOOR) * SEED_SHARE
        can_afford = int(budget // seed_cost)
        buy = max(0, min(want - have, can_afford))
        if buy:
            market.append(["BUY_SEED", crop, buy])

    units = units_positions(farm)
    tasks = _tile_tasks(farm, day, board)
    assigned = _assign(units, tasks, free, crop, seeds.get(crop, 0) if crop else 0)
    return {
        "farmer": assigned[0] if assigned else ["PASS"],
        "hands": assigned[1:],
        "market": market[:10],
    }


def agent(obs, config=None):  # kaggle-environments entry point
    return act(dict(obs), dict(config) if config else None)
