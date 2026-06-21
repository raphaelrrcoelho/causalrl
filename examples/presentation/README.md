# Causal RL you can *see and touch* — presentation

A hands-on `causalrl` walkthrough whose goal is to make an RL agent's **acting and learning**
tangible — not to teach causal theory. Three tiny bandit games, one per rung of the Pearl
hierarchy; in each you watch *where* the causal information enters the decision.

| Demo | What you see & touch | Causal | Naive |
|------|----------------------|:------:|:-----:|
| 1. MABUC | beliefs **split by mood** — animate with ▶ Play; pull once and compare samples | 0.76 | 0.50 |
| 2. POMIS | **hover the 27 levers** for their value; the graph keeps only {∅, {X3}} | 1.00 | 0.50 |
| 3. Counterfactual | **play single rounds**; the agent reads its intent off the decision table | 0.80 | 0.37 |

The figures are **plotly** (hover, zoom, and a play/slider animation for the belief-forming);
the **🎲 buttons** are `ipywidgets` for live stochastic pulls.

## Three ways to present

| Artifact | Command | Best for |
|----------|---------|----------|
| **Standalone HTML** | `uv run python examples/presentation/_build_html.py` | projecting / handing out — interactive (hover, ▶ Play), **no kernel needed** |
| **Live notebook** | `uv run jupyter lab examples/presentation/causal_rl_presentation.ipynb` | presenting live — plotly figures **plus** the 🎲 buttons |
| **Static PNGs** | `uv run python examples/presentation/_build_figures.py` | dropping into slide decks / PDF (matplotlib, no browser) |

## Setup

```bash
uv sync --extra viz        # matplotlib + ipywidgets + plotly + scipy (+ torch)
```

## Files

- `causal_rl_presentation.ipynb` / `.html` — the built notebook and standalone interactive page.
- `_demos.py` — pure compute (training, lever values, decision table); no plotting.
- `_viz.py` — plotly figure builders (take `_demos` data, return figures).
- `_build_html.py` — assembles the standalone HTML from the plotly figures.
- `_make_notebook.py` — regenerates the notebook from source cells.
- `_build_figures.py` — matplotlib static PNGs into `figures/` (slide backup).

The notebook, the HTML, and the PNGs all pull their numbers from `_demos.py`, so they cannot
disagree. Everything is tabular and finishes in seconds.
