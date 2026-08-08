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

## 7. Playing it: where the paper economics were wrong

`agent.py` is a submittable agent built from §1–§6; `harness.py` evaluates it across seeds with
paired comparisons. Against the shipped `starter` agent over 8 seeds:

| | final bank |
| --- | --- |
| `starter` (baseline) | $3,510 |
| **this agent** | **$31,499** (median $31,603, range $28,718–$33,618, sd $1,728) |
| self-play | $17,410 |

**9.0x the baseline.** Decision time is 0.07 ms mean / 0.41 ms worst against a 1,000 ms budget, so
the turn limit is nowhere near binding.

Two things only running it could show.

### Land does *not* bind — labour does

§1 concluded that town demand outruns a full board several times over, so every quadrant should be
bought. Measured, that is simply false:

| quadrants | tiles | final bank |
| --- | --- | --- |
| 1 | 25 | $31,298 |
| **2** | **50** | **$31,793** |
| 3 | 75 | $29,433 |
| 4 | 100 | $24,970 |

Buying the whole board is *worse than buying none of it past the second quadrant*. Land is only
worth what the labour can service, and hands are Fibonacci-priced (§5), so a full board cannot be
staffed. Tiles bought past that point cost twice: $7,000 in purchase, and the unwatered plants on
them decay into weeds that then consume actions to clear. §5 priced labour correctly and still drew
the wrong conclusion, because it costed the hands without asking whether they could reach the work.

### Melon is the whole game

| melon tiles (of 50) | final bank |
| --- | --- |
| 0 | $5,948 |
| 5 | $19,501 |
| **10** | **$31,803** |
| 15 | $28,878 |
| 20 | $30,714 |

Dropping melon costs 5x. The measured optimum is 10 tiles — the same number §6's equilibrium
analysis produced, arrived at independently.

## 8. The equilibrium prediction, tested

§6 predicted that playing the *solo* optimum (20 melon tiles) against an opponent who also grows
melon costs $4,713. That is a falsifiable claim about the real simulator, so: two copies of this
agent, one capped at the equilibrium 10 and one at the solo 20, played against each other across 6
seeds in both seats.

| | mean final bank |
| --- | --- |
| **melon = 10 (equilibrium)** | **$20,179** |
| melon = 20 (solo optimum) | $12,334 |
| paired delta | **+$7,845, winning 12 of 12** |

The direction and order of magnitude hold; the real penalty is larger than the model's $4,713,
which is expected, since the closed-form game abstracts away the labour contention that makes extra
melon tiles cost more than their seed. **A game-theoretic analysis of an abstraction predicted a
result in the full simulator, and won every paired matchup.** That is the one place in this study
where the causal-RL library did something a spreadsheet could not.

## 9. What is still open

- **Animals.** Not implemented. §2 ranks milk second by $/tile/day and §4 shows milk, wool and
  strawberry selling *above* base all season, so the animal lines are the obvious next gain.
- **Routing.** The `~9 tiles per unit` estimate in §5 was never confirmed; the agent uses greedy
  nearest-work matching, which thrashes when two units want the same tile.
- **Opening.** Days 0–10 are cash- and growth-constrained and were tuned by sweep, not solved.
- **Self-play is ~55% of the score against `starter`**, which is the shared market biting exactly
  as §6 says it should. Every number in §1–§5 remains single-seller.

## 10. Correction: the model was transcribed when it should have been imported

`economics.py` originally copied `MARKET_PARAMS`, `SHOPS`, the crop/animal tables and the price
function out of a reading of the env source. That was wrong in two ways, and the second one moved a
headline number.

**Redundant.** The referee exposes `market_price` publicly. The reimplementation agreed with it at
54 of 54 checked points — correct, and pointless.

**Stale.** Between kaggle-environments 1.32.4 (the version first read) and 1.32.6 (the version that
actually runs), the town centre changed from consuming every 12 turns with a multiplier rising to 4x
after day 20, to consuming every 24 turns with **no multiplier at all** — up to 8x less demand. Shops
also began being drawn *with replacement*, so one can unlock twice and consume twice. A transcription
cannot notice either change.

Everything is now imported. The corrected demand table:

| day | tiles of demand (corrected) | (stale figure) |
| --- | --- | --- |
| 10 | 170 | 245 |
| 20 | 315 | 491 |
| 29 | 460 | 588 |

**§1 survives.** Demand still runs to 170–460 tiles against a 100-tile board, so land still does not
saturate for most products — the conclusion held, at ~30% lower magnitude.

**§3 does not.** Melon appears in **no shop**, so its only drain is the town centre: 1 unit/day flat.
Sustainable melon is **1.8 tiles**, not the 7.3 first reported. The solo optimum moves from 20 tiles
to 15.

**§6 and §8 survive unchanged.** Re-running the two-seller game on the corrected drain gives the
same interior equilibrium of **10 melon tiles** (regret 0.0025). The equilibrium is robust to a
change that moved the solo optimum by a third — which is the more useful of the two numbers to be
robust. The head-to-head in §8 tested 10 against 20; under the corrected model 20 is still
over-committed (the solo optimum is 15), so the result stands, though "solo optimum" is now the
wrong label for the 20-tile arm.

The agent scores **$31,499 — identical before and after** de-duplication, since its own copied crop
table happened to match. That is luck, not vindication: it is exactly the kind of copy that would
have silently drifted at the next env release. `agent.py` now reads `CROPS` and `LAND_PRICES` from
the referee too.

**The lesson generalises past this study.** The simulator is provided; anything derived from it
should be imported from it, and only genuinely new quantities — here, per-day drain rates and
per-tile yield rates, which the referee models step-by-step rather than as rates — belong in
`economics.py`.

## 11. Livestock: the largest single gain, and it needed feed

§10's audit found five of eight revenue lines unbuilt, with milk ranked second by $/tile/day and
drained faster than the board can supply. Building it:

| | vs `starter` (8 seeds) | self-play |
| --- | --- | --- |
| crops only | $31,499 | $17,410 |
| **with 6 cows** | **$35,948** (median $35,753, range $31,642–$39,987) | **$26,599** |

**+14% against `starter`, +53% in self-play.** The self-play gain is the larger one and the more
meaningful: milk is the line the town under-supplies, so it holds its price even when both sellers
are working it — exactly what §4 predicted and the opposite of melon's behaviour.

Worst turn is 0.39 ms against the 1,000 ms budget.

### Feed is not optional

The first livestock version scored **$15,478 — worse than no animals at all.** Cows were bought,
placed, went unfed for two days and escaped, at $400 a head. The cause was that the crop planner
never planted wheat: `FEED` takes wheat from the *unit's* inventory (`_inv_take(inv, "WHEAT", 1)` in
the referee), and the farm had none to carry. Feed now outranks even melon in the planting order
while the herd is short of it, at 1.25 wheat tiles per head.

### The herd size is sharply peaked

| cows | | cows | |
| --- | --- | --- | --- |
| 0 | $31,803 | 7 | $30,778 |
| 3 | $30,615 | 8 | $28,812 |
| **6** | **$35,759** | 10 | $16,685 |
| | | 14 | $9,463 |

Past six, the pastures plus their feed tiles crowd out the crops and the labour needed to service
both. Cow beats sheep ($32,631) and beats goose badly ($17,009) — a goose caps at 4 unharvested
units and produces daily, so it costs the most labour per dollar earned.

Melon is now flat between 10 and 15 tiles ($35,759 vs $35,761); the equilibrium cap of 10 is kept,
since §8 showed it is the value that survives a melon-growing opponent.

### Where this leaves us

Against the town's whole-season absorption of $210,262 (about $105,131 per seller), $35,948 is
**~34% of the available money**, up from 30%. Three lines remain unbuilt — strawberry ($28.80/tile/
day), wool ($29.63) and tomato ($19.80) — plus greedy routing that still thrashes and an untuned
opening. This is a stronger prototype. It is not yet evidence of a competitive entry, and nothing
here has been measured against a single real leaderboard agent.
