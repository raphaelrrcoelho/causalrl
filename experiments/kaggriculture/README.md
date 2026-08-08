# Kaggriculture — running this locally

A study of the [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) simulation
competition: an economic model read from the referee, a two-seller game analysis, a playable agent,
and an opponent pool to measure against. Findings and their evidence are in **[FINDINGS.md](FINDINGS.md)**.

Handing this to a coding agent rather than running it yourself? Point it at **[AGENTS.md](AGENTS.md)**,
which carries the invariants, the dead ends already explored, and the ranked next steps.

## Setup

```bash
git clone https://github.com/raphaelrrcoelho/causalrl.git
cd causalrl
git checkout claude/kaggle-agriculture-competitiveness-b0f6f7
```

Then install the simulator. **`--no-deps` is required**: `kaggle-environments` depends on `pygame`
for its visualiser, which frequently fails to build and is not needed to run episodes.

```bash
python -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install --no-deps kaggle-environments
pip install numpy
```

`import kaggle_environments` prints `Loading environment open_spiel_env failed: No module named
'pyspiel'` on startup. That is unrelated to this competition and harmless.

Only the `equilibrium` command needs the library in this repo:

```bash
pip install -e .          # optional, for `run.py equilibrium`
```

## Commands

Run everything from the repository root.

```bash
python experiments/kaggriculture/run.py --help
```

| command | what it does | time |
| --- | --- | --- |
| `economics` | demand and per-tile tables, from the referee's own constants | instant |
| `eval --seeds 8` | score the agent against `starter` | ~20 s |
| `selfplay --seeds 5` | score the agent against itself — the shared-market read | ~15 s |
| `timing` | per-turn decision cost against the 1,000 ms budget | ~3 s |
| `pool --seeds 3` | round-robin over the six-strategy pool | **~6 min** |
| `equilibrium --seeds 3` | solve the empirical game over the pool (needs `causalrl`) | ~6 min |
| `submission --out submission.py` | write a single-file agent to upload | instant |

Expected output for the two quick ones, at the committed defaults:

```
$ python experiments/kaggriculture/run.py eval --seeds 8
agent vs starter  (8 seeds, 20s)
  mean   $42,177        # starter's own baseline is $3,510
  ...

$ python experiments/kaggriculture/run.py timing
turns: 719  mean 0.103 ms  worst 0.375 ms
budget: 1000 ms per turn
```

## Submitting to Kaggle

`agent.py` imports only from `kaggle_environments`, so it is already self-contained — no bundling
step is needed.

```bash
python experiments/kaggriculture/run.py submission --out submission.py
```

Upload `submission.py` and use `agent` as the entry point. It matches the signature the referee
calls, `agent(obs, config)`.

## Tuning

The policy is a handful of module-level constants at the top of `agent.py`, each carrying the
measurement that set it:

| constant | value | what moved it |
| --- | --- | --- |
| `MAX_QUADRANTS` | 1 | land was over-valued twice; see FINDINGS §7 and §12 |
| `MELON_TILES` | 8 | the contested line; §6, §8, §12 |
| `ANIMAL_TARGET` | 2 | sharply peaked; §11, §12 |
| `MAX_HANDS` | 6 | +$4,208 over 4; §12 |

To sweep one, set it on the module and re-evaluate:

```python
import sys; sys.path.insert(0, "experiments/kaggriculture")
import agent
from harness import evaluate

for n in (4, 6, 8):
    agent.MAX_HANDS = n
    print(n, evaluate(agent.agent, "starter", seeds=range(5))["mean"])
```

`harness.compare()` does the same as a **paired** difference on identical seeds, which removes the
shared episode randomness an unpaired comparison leaves in. Its `clear` flag is a two-standard-error
screen, not a valid sequential test — it does not survive peeking, so do not add seeds until it
turns green and then stop.

## What this is not

The agent has never been played against a real leaderboard entry. `starter` farms a single tile,
and the opponent pool shares this agent's routing and market code — it varies portfolio and selling
policy, which is where the strategy lives, but says nothing about whether the underlying
implementation is any good.

Three revenue lines are unbuilt (strawberry, wool, tomato), routing is greedy nearest-work and
thrashes when two units want the same tile, and the agent captures roughly 40% of the town's
per-seller absorption. Treat the numbers here as a measurement rig, not a competitiveness claim.

If you can get the public [`episodes.csv`](https://www.kaggle.com/datasets/georgymamarin/kaggriculture-episodes)
(final banks and ratings from real ladder games), that is the missing reference distribution —
Kaggle is unreachable from the environment this was developed in.

## Files

| file | |
| --- | --- |
| `economics.py` | the referee's model, imported not transcribed (see FINDINGS §10) |
| `market_game.py` | the two-seller contested-product game |
| `agent.py` | the playable agent — the submission |
| `harness.py` | evaluation across seeds, paired comparison, diagnostics |
| `opponents.py` | the six-strategy reference pool |
| `run.py` | the command line above |
| `FINDINGS.md` | every result and the errors made getting there |
| `AGENTS.md` | handoff brief for a coding agent continuing the work |
