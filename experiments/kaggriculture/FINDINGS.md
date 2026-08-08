# Kaggriculture: what the economics say

All constants transcribed from `kaggle_environments/envs/kaggriculture/kaggriculture.py` (v1.32.4).
`economics.py` reproduces the referee's published price table **exactly** at every checkpoint
(P(I0−T), P(I0+T), P(I0+2T) for all nine resources), which is the check that the model is the
sim's and not a paraphrase of its prose.

## 1. Land is the binding constraint, not market saturation

Town demand, summed over products and converted to the tiles needed to serve it:

| day | shops | sustainable revenue/day | tiles needed |
| --- | --- | --- | --- |
| 3 | 1 | $3,158 | 99 |
| 10 | 3 | $7,672 | 245 |
| 20 | 6 | $15,345 | 491 |
| 25+ | 8 | $18,060 | 588 |

A farm caps at **100 tiles**. From roughly day 6 onward the town can absorb several times
everything the whole board can grow, so for seven of the eight products the price never leaves its
base. **The sell-rate anxiety the rules invite is mostly misplaced** — you cannot produce enough to
crash wheat, carrot, tomato, egg, milk or wool.

## 2. Rank tiles by revenue at base price

`$/tile/day` counts the feed tiles an animal needs (1 wheat/day at 0.80 wheat/tile/day = 1.25
extra tiles per animal), which is what makes the animal lines cheaper than they first look:

| product | units/tile/day | base | **$/tile/day** |
| --- | --- | --- | --- |
| **MELON** | 0.55 | 250 | **137.50** |
| MILK | 0.22 | 160 | 35.56 |
| WOOL | 0.15 | 200 | 29.63 |
| STRAWBERRY | 0.24 | 120 | 28.80 |
| CARROT | 0.75 | 35 | 26.25 |
| EGG | 0.44 | 50 | 22.22 |
| WHEAT | 0.80 | 25 | 20.00 |
| TOMATO | 0.33 | 60 | 19.80 |

Melon is worth ~4x anything else per tile.

## 3. Melon is the one product you *can* saturate — and the optimum is well past its
   sustainable rate

Revenue over days 10–29, selling as harvested:

| melon tiles | season revenue | $/tile |
| --- | --- | --- |
| 8 | $22,823 | 2,853 |
| 15 | $39,237 | 2,616 |
| **20** | **$45,738** | 2,287 |
| 25 | $43,698 | 1,748 |
| 30 | $37,900 | 1,263 |

Town drain supports only ~7 melon tiles, but revenue peaks at **~20** — roughly **2.7x the
sustainable rate**. Selling only what the town absorbs would leave about half the melon money on
the table. The correct rule is marginal-revenue-zero, not zero-inventory-drift.

Past ~25 tiles the `sq` curve collapses it (`above_target = 3.60`) and revenue falls off a cliff.

## 4. Milk's price *rises*

At 40 milk tiles the market still ends the season ~212 units **below** I0, so milk sells at
$287–322 against a $160 base. The town drains milk faster than a third of the board can produce it.
Same shape for wool and strawberry.

**Caveat, and it is the important one:** this assumes a single seller. The market is shared, so a
co-player selling the same product splits the drain. That is the one place the opponent enters the
economics, and it is where a strategic (rather than optimisation) treatment is needed.

## 5. Labour is affordable but the marginal hand is not free

Hire cost is Fibonacci in the day's hire count. A unit spawns at the shed, walks a mean of ~5
Manhattan steps, then costs ~2 actions per tile (move + water), so it services ~9 tiles/day:

| tiles | units/day | hire $/day | $/season |
| --- | --- | --- | --- |
| 25 | 3 | 2 | 60 |
| 50 | 6 | 12 | 360 |
| 75 | 9 | 54 | 1,620 |
| 100 | 12 | 232 | 6,960 |

Buying all land costs $7,000 and servicing it ~$6,960 — together ~$14k against a revenue ceiling
well north of $45k. Both are clearly worth it; the order-of-magnitude conclusion is robust even
though the 9-tiles-per-unit figure is a routing approximation, not a simulation.

## Strategy skeleton

1. Buy all three quadrants early; land is the binding resource.
2. ~20 tiles of melon — the single highest-value line, run deliberately past its sustainable rate.
3. Fill the rest with milk, then wool/strawberry.
4. Hire ~12 hands/day once the board is full.
5. Wheat only as animal feed, not as a cash crop.

## What is *not* settled

- **The opponent.** Every number above is single-seller. Shared drain is the open question.
- **Routing.** The 9-tiles-per-unit figure needs a real scheduler to confirm.
- **Ramp-up.** Days 0–10 are capital- and growth-constrained; this analyses steady state.

## 6. The contested-product game: the solo optimum is a mistake

`market_game.py` turns §3's melon question into a two-player normal form — each seller picks how
many of 100 tiles to devote to melon; both sell as harvested into one shared curve; the payoff is
whole-farm revenue. Payoffs are known in closed form from the simulator's own price function, which
is precisely the condition `causalrl.run_no_regret` requires.

Regret matching over 20,000 rounds (final measured regret 0.0029) converges to an **interior
equilibrium of 10 melon tiles each — exactly half the single-seller optimum**, and the same point on
two independent action grids.

| profile | my revenue | their revenue |
| --- | --- | --- |
| 20 vs nobody (the solo dream) | $102,627 | — |
| **10 vs 10 (equilibrium)** | **$86,869** | $86,869 |
| 20 vs 10 (over-committing) | $82,156 | $86,869 |
| 20 vs 20 (both greedy) | $74,489 | $74,489 |

**Playing §3's answer against a competent opponent costs $4,713.** No deviation from 10 pays: the
best alternative (15 tiles) is −$206 and everything else is worse by thousands. Note this is *not* a
prisoner's dilemma — 10v10 beats 20v20 for both — so the equilibrium is somewhere both players are
content to sit, which makes it a reasonable thing to plan against.

`certify_cce_do` with the run's measured regret as its epsilon bounds the time-averaged whole-farm
revenue of *any* outcome a no-regret population can realise:

```
[BOUNDED] time-averaged functional of the learning population lies in the CCE interval
          [measured realized regret (finite-time, no asymptotic assumption)] | value=[81.26, 89.57]
```

No asymptotics, and the equilibrium point ($86,869) sits inside it.

### A modelling error the boundary check caught

The first version of this game scored the melon line alone. Its equilibrium sat exactly on the
action grid's maximum — at 40 tiles, then 100, then 200, wherever the grid stopped. That is the
signature of a truncated answer, and chasing it exposed the real fault: **tiles were free**. With no
opportunity cost, an extra tile is weakly profitable even at the $1 price floor, so the game
degenerates. Charging a diverted tile the forgone `MILK` revenue ($35.56/tile/day) makes the
trade-off real and the equilibrium interior. Worth recording because the symptom pointed at
resolution and the cause was specification.
