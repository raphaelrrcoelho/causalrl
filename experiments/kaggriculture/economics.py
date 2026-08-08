"""Exact economic model of the Kaggriculture simulator, read off its shipped source.

Every constant here is transcribed from `kaggle_environments/envs/kaggriculture/kaggriculture.py`
(v1.32.4), not from the prose tables, so the numbers are the ones the referee actually uses.

The question this answers: **what is a tile-day worth?** Naively it is
`production_rate * base_price`, which says melon dominates everything. That is wrong, because
selling depresses the price and the only thing that lifts it back is the town consuming inventory.
So the quantity that matters is revenue at a *sustainable* sell rate -- the rate at which the town
drains what you produce, leaving the price where it started.
"""

from __future__ import annotations

import math

MARKET_I0 = 10_000
MARKET_PARAMS = {
    "WHEAT": {
        "base": 25,
        "T": 400,
        "below_func": "sqrt",
        "below_target": 0.80,
        "above_func": "log",
        "above_target": 0.20,
    },
    "CARROT": {
        "base": 35,
        "T": 450,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sqrt",
        "above_target": 0.70,
    },
    "TOMATO": {
        "base": 60,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "sqrt",
        "above_target": 0.60,
    },
    "STRAWBERRY": {
        "base": 120,
        "T": 100,
        "below_func": "sqrt",
        "below_target": 0.70,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "MELON": {
        "base": 250,
        "T": 300,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.60,
    },
    "EGG": {
        "base": 50,
        "T": 332,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "log",
        "above_target": 0.20,
    },
    "MILK": {
        "base": 160,
        "T": 122,
        "below_func": "sqrt",
        "below_target": 0.60,
        "above_func": "linear",
        "above_target": 1.60,
    },
    "WOOL": {
        "base": 200,
        "T": 105,
        "below_func": "log",
        "below_target": 0.20,
        "above_func": "sq",
        "above_target": 3.20,
    },
    "FERTILIZER": {
        "base": 100,
        "T": 200,
        "below_func": "linear",
        "below_target": 0.40,
        "above_func": "linear",
        "above_target": 0.40,
    },
}

SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}
TOWN_CENTER_PRODUCTS = [p for p in MARKET_PARAMS if p != "FERTILIZER"]
TOWN_CENTER_DEMAND_SCHEDULE = [(20, 4), (10, 2), (0, 1)]

TURNS_PER_DAY = 24
SHOP_SELL_INTERVAL = 4  # turns between each unlocked shop's consumption tick
CENTER_SELL_INTERVAL = 12  # turns between town-centre ticks
SHOP_UNLOCK_INTERVAL = 3  # days between shop unlocks
SEASON_DAYS = 30

# Units harvested per tile per day, watering daily, no fertiliser (README "Yield / tile / day";
# for animals the steady-state 1/interval once producing).
YIELD_PER_TILE_DAY = {
    "WHEAT": 0.80,
    "CARROT": 0.75,
    "TOMATO": 0.33,
    "STRAWBERRY": 0.24,
    "MELON": 0.55,
    "EGG": 1.00,
    "MILK": 0.50,
    "WOOL": 1 / 3,
}
# Tiles a producing unit occupies, plus the wheat tile an animal's feed needs (1 wheat/day at
# 0.80 wheat/tile/day => 1.25 wheat tiles per animal).
TILES_PER_UNIT = {
    "WHEAT": 1.0,
    "CARROT": 1.0,
    "TOMATO": 1.0,
    "STRAWBERRY": 1.0,
    "MELON": 1.0,
    "EGG": 1 + 1 / 0.80,
    "MILK": 1 + 1 / 0.80,
    "WOOL": 1 + 1 / 0.80,
}


def _shape(func: str, x: float) -> float:
    x = max(0.0, x)
    return {"linear": x, "sq": x * x, "sqrt": math.sqrt(x), "log": math.log1p(x)}.get(func, x)


def price(item: str, inventory: float) -> int:
    """The referee's sell price at a given market inventory (floored at $1, rounded)."""
    p = MARKET_PARAMS[item]
    if inventory < MARKET_I0:
        func, target, sign = p["below_func"], p["below_target"], 1.0
    else:
        func, target, sign = p["above_func"], p["above_target"], -1.0
    denom = _shape(func, p["T"])
    amp = target * p["base"] / denom if denom else 0.0
    value = p["base"] + sign * amp * _shape(func, abs(inventory - MARKET_I0))
    return max(1, round(value))


def shops_unlocked(day: int) -> int:
    """Shops active on ``day`` -- one every SHOP_UNLOCK_INTERVAL days, capped by the pool."""
    return min(len(SHOPS), day // SHOP_UNLOCK_INTERVAL)


def town_drain_per_day(item: str, day: int, *, shop_names: list[str] | None = None) -> float:
    """Units of ``item`` the town removes from the market per day.

    This is the number that makes a sell rate sustainable: sell at it and inventory holds, so the
    price holds. Shops are averaged over the pool when the specific unlock order is unknown -- the
    order is random per episode, so the expected drain is what a strategy can plan against.
    """
    if item == "FERTILIZER":
        return 0.0
    multiplier = next(m for threshold, m in TOWN_CENTER_DEMAND_SCHEDULE if day >= threshold)
    centre = (TURNS_PER_DAY / CENTER_SELL_INTERVAL) * multiplier
    ticks = TURNS_PER_DAY / SHOP_SELL_INTERVAL
    active = shop_names if shop_names is not None else None
    if active is None:
        n = shops_unlocked(day)
        wanting = sum(1 for products in SHOPS.values() if item in products)
        single = sum(1 for products in SHOPS.values() if products == [item])
        expected = (wanting + single) * (n / len(SHOPS))  # single-product shops consume 2x
        return centre + ticks * expected
    demand = sum((2 if SHOPS[s] == [item] else 1) for s in active if item in SHOPS[s])
    return centre + ticks * demand
