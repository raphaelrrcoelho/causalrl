# Causal RL you can *see and touch* — presentation

A hands-on `causalrl` walkthrough whose goal is to make an RL agent's **acting and learning**
tangible — not to teach causal theory. Three tiny bandit games (one per rung of the Pearl
hierarchy); in each you watch *where* the causal information enters the decision.

| Demo | What you see & touch | Causal | Naive |
|------|----------------------|:------:|:-----:|
| 1. MABUC | beliefs **split by mood** as you scrub training; pull once and compare samples | 0.76 | 0.50 |
| 2. POMIS | **poke any of 27 levers**, read its value; graph keeps only 2 | 1.00 | 0.50 |
| 3. Counterfactual | **play single rounds**; agent reads its intent off the decision table | 0.80 | 0.37 |

Each demo has a **▶ live** cell (interactive `ipywidgets`: sliders, buttons, dropdowns) and a
**▦ static** cell that runs the same logic once — so the notebook validates headlessly and
projects even where widgets don't render.

## Files

- `causal_rl_presentation.ipynb` — the live notebook (run it in front of the room).
- `_make_notebook.py` — source of the notebook (regenerates it).
- `_build_figures.py` — renders standalone PNGs into `figures/` for slides.
- `figures/` — pre-rendered scoreboard figures (static backup).

## Run it

```bash
uv sync --extra dev --extra torch
uv pip install matplotlib          # not pulled by any extra

# live, interactive (recommended for presenting)
uv run jupyter lab examples/presentation/causal_rl_presentation.ipynb
```

The `▶ live` cells need a running kernel with `ipywidgets` (drag the sliders, click the
buttons). The `▦ static` cells and the `figures/` PNGs cover the case where you project from a
static HTML/PDF export. Everything is tabular and finishes in seconds; the notebook and
`_build_figures.py` run identical code.
