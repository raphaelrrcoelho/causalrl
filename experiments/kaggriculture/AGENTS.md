# Brief for an agent continuing this work

You are picking up a study of the Kaggle **Kaggriculture** simulation competition. Human setup and
commands are in [README.md](README.md); every result and how it was obtained is in
[FINDINGS.md](FINDINGS.md). This file is the handoff: what is settled, what is not, what has already
failed, and what to do next.

**Goal.** Maximise the agent's final bank in a two-player, 720-turn farming sim. The deliverable is
`agent.py`, submitted to Kaggle as a single file.

---

## Orient yourself in five minutes

```bash
pip install --no-deps kaggle-environments && pip install numpy   # --no-deps is required
python experiments/kaggriculture/run.py economics                # the model
python experiments/kaggriculture/run.py eval --seeds 8           # ~$42,177 vs starter
python experiments/kaggriculture/run.py selfplay --seeds 5       # ~$29,285
```

Read `FINDINGS.md` §10, §11 and §12 before changing anything. They contain the three corrections
that cost the most time.

---

## Invariants — do not break these

1. **Import from the referee; never transcribe it.** `economics.py` and `agent.py` both read
   `MARKET_PARAMS`, `CROPS`, `ANIMALS`, `LAND_PRICES`, `market_price` from
   `kaggle_environments.envs.kaggriculture.kaggriculture`. An earlier version copied them and went
   stale across an env release, moving a headline conclusion by 30% (§10). Only derived quantities
   the referee does not expose — per-day drain rates, per-tile yield rates — belong to us.
2. **Nothing here may move into `src/causalrl/`.** The repo runs a generality lint that fails CI on
   domain nouns in the library's public surface. `experiments/kaggriculture/` is explicitly
   un-ignored in `.gitignore`; the rest of `experiments/` is not.
3. **Lint before you measure.** A sweep was once run with an `F821` outstanding and produced five
   identical `$3,000` rows — the untouched starting stake — costing a full diagnosis cycle for a bug
   ruff had already located. `uv run ruff check .` first, every time.
4. **Never compare on one seed.** Weed spawns and shop-unlock order are random. Use
   `harness.compare()` for a *paired* difference on identical seeds.
5. **`harness.compare()`'s `clear` flag is a screen, not a test.** Two standard errors does not
   survive peeking. Do not add seeds until it goes green and then stop; fix the seed count in
   advance, or report the flag as the weak evidence it is.

---

## Already tried, already failed — do not repeat

| attempt | result | why |
| --- | --- | --- |
| Cash floor + melon-capital gate | **$25,202 → $2,079** | The bank pinned at the floor, so no land tier was ever affordable. The mid-season dip to $0 it "fixed" was not fatal. |
| Buying all four quadrants | $24,970 vs $31,793 at two | Labour cannot service a full board at Fibonacci hire prices; surplus tiles decay to weeds that cost actions to clear. |
| Livestock without a feed crop | **$31,499 → $15,478** | `FEED` takes wheat from the *unit's* inventory, not the shed. Unfed cows escape at $400 a head. |
| Metering premium sales | ~neutral to −$2,500 | The 100-item shed cap makes hoarding impossible, so the burst-harvest price walk cannot be dodged. |
| Goose over cow | $17,009 vs $35,759 | Caps at 4 unharvested units and produces daily — the most labour-hungry animal per dollar. |
| Melon at the *solo* optimum (20) | loses 12/12 to melon at 10 | Against any melon-growing opponent, over-committing costs ~$7,845 (§8). |

**The recurring error is over-valuing land.** §1 said buy every quadrant, §7 revised to two, §12 to
one. Each sweep was right about its own setup and each under-corrected in the same direction. If a
change makes the farm bigger, be suspicious.

---

## Ranked next steps

1. **Build the three missing revenue lines** — strawberry ($28.80/tile/day), wool ($29.63), tomato
   ($19.80). Highest expected value. §4 shows strawberry and wool are drained faster than the board
   can supply, so like milk they sell *above* base all season and hold their price under
   competition. Livestock was worth +14% against `starter` and **+53% in self-play**; expect the
   same shape.
2. **Fix routing.** Assignment is greedy nearest-work and thrashes when two units want the same
   tile. A proper per-day assignment (Hungarian, or a claim-and-hold plan) should raise utilisation.
   The `~9 tiles per unit` figure in §5 was never verified — verify it.
3. **Solve the opening.** Days 0–10 are cash- and growth-constrained and were tuned by sweep, not
   reasoned. Melon pays at day 10; carrot turns over every three.
4. **Widen the opponent pool.** Every bot in `opponents.py` shares this agent's routing, so the pool
   measures strategy and not craft. A structurally different bot would test more.
5. **Get real leaderboard data.** The public `episodes.csv`
   (`kaggle.com/datasets/georgymamarin/kaggriculture-episodes`) carries final banks and ratings from
   real ladder games. Kaggle was network-blocked where this was developed. **Nothing here has ever
   been measured against a real competitor** — until that changes, no claim about competitiveness is
   supported.

---

## How to report

State the seed count and the paired delta, not a single number. When a change is inside the noise,
say so — §12's tuned config beats the previous default 8/8 but only 6/8 against the pool's best, and
is written up as *not established* on the second comparison. Record failures in `FINDINGS.md` with
the measurement that exposed them; four of that file's most useful sections are corrections.
