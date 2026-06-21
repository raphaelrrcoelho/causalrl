# Causal RL, hands-on — presentation

A practical, visual walk through `causalrl` for a math-literate audience, organised by the
three rungs of the **Pearl Causal Hierarchy**. Each demo is a small bandit where the
*interventional* signal alone is degenerate (every arm looks equal under `do`) yet a causal
agent still wins:

| Demo | Rung | Causal agent | Naive agent |
|------|------|:------------:|:-----------:|
| 1. MABUC — equal `do()`-means | L1→L2 | **0.76** | 0.50 |
| 2. POMIS — where to intervene (27 arms → 2) | L2 | **1.00** | 0.50 |
| 3. Counterfactual policy — `E[Y_do(a) \| intent]` | L3 | **0.80** | 0.37 |

## Files

- `causal_rl_presentation.ipynb` — the live notebook (run it in front of the room).
- `figures/` — pre-rendered PNGs for slides / projecting without running code.
- `_make_notebook.py` — regenerates the notebook from source cells.
- `_build_figures.py` — runs the same three demos and writes `figures/*.png`.

## Run it

```bash
uv sync --extra dev --extra torch
uv pip install matplotlib

# live notebook
uv run jupyter lab examples/presentation/causal_rl_presentation.ipynb

# or just rebuild the static figures
uv run python examples/presentation/_build_figures.py
```

Every cell is tabular and finishes in a few seconds. The notebook and the figure builder run
identical code, so the static PNGs and the live output cannot disagree.
