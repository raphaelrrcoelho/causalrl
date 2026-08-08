"""Economic model of the Kaggriculture simulator, read from the referee at import time.

**Nothing here is transcribed.** An earlier version copied `MARKET_PARAMS`, `SHOPS`, the crop and
animal tables and the price function into this file from a reading of the env source. That was
wrong twice over: the price function is public (`market_price`) so reimplementing it was redundant,
and the copy went stale. Between kaggle-environments 1.32.4 and 1.32.6 the town centre changed from
consuming every 12 turns with a multiplier rising to 4x after day 20, to consuming every 24 turns
with no multiplier at all -- up to 8x less demand -- and shops began being drawn *with replacement*
so the same shop can be unlocked twice and consume twice. A transcription cannot notice that; an
import cannot miss it.

So this module imports the referee's own constants and only adds what the referee does not expose:
the per-day drain rate implied by its consumption loop, and the per-tile yield rates.
"""

from __future__ import annotations

from kaggle_environments.envs.kaggriculture.kaggriculture import (
    ANIMALS,
    CROPS,
    MARKET_I0,
    MARKET_PARAMS,
    PRODUCTS,
    SHOPS,
    TOWN_CENTER_PRODUCTS,
    market_price,
)

__all__ = [
    "ANIMALS",
    "CROPS",
    "MARKET_I0",
    "MARKET_PARAMS",
    "PRODUCTS",
    "SHOPS",
    "TILES_PER_UNIT",
    "YIELD_PER_TILE_DAY",
    "price",
    "shops_unlocked",
    "town_drain_per_day",
]

TURNS_PER_DAY = 24
SHOP_SELL_INTERVAL = 4  # `townShopSellInterval` default
CENTER_SELL_INTERVAL = 24  # `townCenterSellInterval` default -- was 12 in 1.32.4
SHOP_UNLOCK_INTERVAL = 3
SEASON_DAYS = 30

#: Units harvested per tile per day under daily watering and no fertiliser; for animals the
#: steady-state ``1 / interval`` once producing. Derived from CROPS/ANIMALS, which the referee
#: exposes, but the *rate* is ours -- the referee models growth step by step, not as a rate.
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
#: Tiles a producing unit occupies, including the wheat an animal's feed needs
#: (1 wheat/day at 0.80 wheat/tile/day => 1.25 extra tiles per animal).
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


def price(item: str, inventory: float) -> int:
    """The referee's own sell price. Thin alias for :func:`market_price`, kept for readability."""
    return market_price(item, inventory)


def shops_unlocked(day: int) -> int:
    """Shop instances active on ``day``. Drawn WITH replacement, so this can exceed len(SHOPS)."""
    return day // SHOP_UNLOCK_INTERVAL


def town_drain_per_day(item: str, day: int, *, shop_names: list[str] | None = None) -> float:
    """Units of ``item`` the town removes from the market per day.

    The rate a seller can sustain without moving the price. Shops are averaged over the pool when
    the realised unlock order is unknown, since it is drawn fresh each episode; pass ``shop_names``
    to score a specific one. Note the draw is with replacement, so a shop may appear more than once
    and each instance consumes independently.
    """
    if item == "FERTILIZER" or item not in TOWN_CENTER_PRODUCTS:
        centre = 0.0
    else:
        centre = TURNS_PER_DAY / CENTER_SELL_INTERVAL
    ticks = TURNS_PER_DAY / SHOP_SELL_INTERVAL
    if shop_names is not None:
        demand = sum((2 if len(SHOPS[s]) == 1 else 1) for s in shop_names if item in SHOPS[s])
        return centre + ticks * demand
    instances = shops_unlocked(day)
    wanting = sum(
        (2 if len(products) == 1 else 1) for products in SHOPS.values() if item in products
    )
    expected = wanting * (instances / len(SHOPS))
    return centre + ticks * expected
