"""Generate causal_rl_presentation.ipynb — the hands-on "see & touch the agent" notebook.

The notebook is a build artifact; this generator is the reviewable source. Run::

    uv run python examples/presentation/_make_notebook.py

The compute lives in ``_demos.py`` and the interactive figures in ``_viz.py`` (plotly), so the
notebook and the standalone-HTML build (``_build_html.py``) share one source of truth. The
plotly figures give hover/zoom and a play/slider animation; the ``🎲`` buttons stay in
ipywidgets for live stochastic pulls.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells: list = []

cells.append(md(r"""# Causal RL you can *see and touch*

Not a course on causality — a chance to **watch an RL agent act and learn**, and see exactly
where the causal information enters the decision. Three tiny bandit games, one per rung of the
Pearl hierarchy. The plots are **interactive** (hover to read values, zoom, and press ▶ Play to
watch a belief form); the **🎲 buttons** play live rounds.

| Demo | What you see & touch | Causal | Naive |
|------|----------------------|:------:|:-----:|
| 1. MABUC | beliefs **split by mood** (animate it); pull once and compare samples | 0.76 | 0.50 |
| 2. POMIS | **hover the 27 levers**; the graph keeps only {∅, {X3}} | 1.00 | 0.50 |
| 3. Counterfactual | **play single rounds**; the agent reads its intent off the table | 0.80 | 0.37 |

> A standalone interactive **HTML** (no kernel needed, great for projecting/sharing) is built by
> `uv run python examples/presentation/_build_html.py`."""))

cells.append(code("""import sys
sys.path.insert(0, ".")            # so `_demos` / `_viz` import when run from this folder
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import plotly.io as pio

import _demos as d
import _viz as v

pio.renderers.default = "notebook_connected"  # plotly.js from CDN — keeps the .ipynb small

# Train once; every figure and button below reuses these objects.
SNAP_STEPS, causal_snaps, naive_snaps, first_decisions, (causal, naive) = d.mabuc_snapshots()
pomis = d.pomis_data()
cf_agent, M = d.counterfactual_agent_and_table()"""))

# ---- Demo 1 ---------------------------------------------------------------------------
cells.append(md(r"""## Demo 1 — MABUC: watch the belief split by mood

A hidden mood drives both the lucky arm and the agent's gut *intuition* `I`. One line of theory:
the arms are identical under intervention, $\mathbb{E}[Y\mid do(X{=}0)]=\mathbb{E}[Y\mid do(X{=}1)]=0.5$,
so choosing well is impossible without reading `I`.

- **Causal**: one belief per **(mood, arm)** → learns "in mood 0, arm 0 is lucky".
- **Naive**: one belief per **arm** → pools the moods, both arms stuck at 0.5.

Press **▶ Play** (or drag) and watch the causal beliefs pull apart while the naive ones stay glued."""))

cells.append(code("""print("First decisions of the causal agent (sampled thetas → arm played → reward):")
for t, ctx, thetas, action, r in first_decisions:
    print(f"  step {t}: intuition={ctx}  sampled={thetas}  -> arm {action}  reward={r:.0f}")"""))

cells.append(code("""v.belief_animation(SNAP_STEPS, causal_snaps, naive_snaps)"""))

cells.append(code("""# 🎲 pull once: both agents decide on the same fresh mood, samples shown
from causalrl.envs.suite.mabuc import MABUCEnv

_env, _rng = MABUCEnv(seed=99), np.random.default_rng(7)
_tally = {"causal": 0.0, "naive": 0.0, "n": 0}
_out = widgets.Output()


def _pull_once(_=None):
    obs, _i = _env.reset()
    cctx, cth, ca, cr = d.transparent_pull(causal, obs, _env, _rng)
    nctx, nth, na_, nr = d.transparent_pull(naive, {"intuition": cctx}, _env, _rng)
    _tally["causal"] += cr; _tally["naive"] += nr; _tally["n"] += 1
    with _out:
        print(f"intuition={cctx} | CAUSAL sampled {cth.round(2)} → arm {ca} (r={cr:.0f})   "
              f"NAIVE sampled {nth.round(2)} → arm {na_} (r={nr:.0f})   "
              f"|| avg reward  causal={_tally['causal']/_tally['n']:.2f} "
              f"naive={_tally['naive']/_tally['n']:.2f}")


_btn = widgets.Button(description="🎲 pull once", button_style="primary")
_btn.on_click(_pull_once)
display(_btn, _out)"""))

# ---- Demo 2 ---------------------------------------------------------------------------
cells.append(md(r"""## Demo 2 — POMIS: touch the 27 levers, keep the 2 sets that matter

The lever is now *which variables to set*. Chain $X_1\to X_2\to X_3\to Y$ with a hidden
back-door $X_1\leftrightarrow Y$; each of $X_1,X_2,X_3$ is idle or forced to 0/1 → **27 arms**.
One line of theory: setting an upstream variable destroys the $X_3\to Y$ effect you want, so
POMIS proves only $\{\varnothing,\{X_3\}\}$ can be optimal.

**Hover any bar** to read a lever's true value and whether POMIS kept it — and notice that plain
*observing* ($\varnothing$) is the winner at value 1.0."""))

cells.append(code("""v.lever_bar(pomis["labels"], pomis["values"], pomis["in_pomis"], pomis["optimal"])"""))

cells.append(code("""v.scoreboard(pomis["curves"], pomis["optimal"])  # POMIS converges at once; brute force crawls"""))

# ---- Demo 3 ---------------------------------------------------------------------------
cells.append(md(r"""## Demo 3 — Counterfactual policy: play one round at a time

A 3-arm bandit where the lucky arm *is* the hidden state $U$ and intent $I=U$. One line of
theory: every fixed intervention averages to chance, $\mathbb{E}[Y\mid do(X{=}a)]\approx0.37$,
but the counterfactual $\mathbb{E}[Y_{do(a)}\mid I{=}i]$ is sharp — play the arm matching your
intent (the diagonal). Hover the table, then play rounds and watch it beat the naive guesser."""))

cells.append(code("""v.decision_heatmap(M)"""))

cells.append(code("""# 🎲 play a round: counterfactual policy follows intent, naive guesses
from causalrl.agents.bandits import NaiveThompsonSampling

_cf_env = d.make_cf_env(seed=3)
_cf_naive = NaiveThompsonSampling(n_arms=3, seed=0)
_cf_tally = {"cf": 0.0, "naive": 0.0, "n": 0}
_cf_out = widgets.Output()


def _play_round(_=None):
    obs, _i = _cf_env.reset()
    intent = obs["intuition"]
    a_cf = cf_agent.act(obs)
    _, r_cf, _, _, _ = _cf_env.step(a_cf)
    a_nv = _cf_naive.act(obs)
    _cf_env.reset(); _cf_env._u = intent            # evaluate naive on the SAME hidden state
    _, r_nv, _, _, _ = _cf_env.step(a_nv)
    _cf_naive.update(obs, a_nv, r_nv)
    _cf_tally["cf"] += r_cf; _cf_tally["naive"] += r_nv; _cf_tally["n"] += 1
    with _cf_out:
        print(f"intent={intent} | COUNTERFACTUAL → arm {a_cf} (table {M[intent].round(2)}) "
              f"r={r_cf:.0f}   NAIVE → arm {a_nv} r={r_nv:.0f}   "
              f"|| avg  cf={_cf_tally['cf']/_cf_tally['n']:.2f} naive={_cf_tally['naive']/_cf_tally['n']:.2f}")


_cf_btn = widgets.Button(description="🎲 play a round", button_style="primary")
_cf_btn.on_click(_play_round)
display(_cf_btn, _cf_out)"""))

cells.append(md(r"""## What you just watched

In all three games the **intervention** signal alone was useless — every arm looked identical
under $do$. The causal agent won because it could *see and use* something extra, and you watched
where:

1. **MABUC** — beliefs **split by mood**: the causal agent's arms pulled apart while the naive ones stayed glued at 0.5.
2. **POMIS** — the graph **pruned 27 levers to {∅, {X3}}** before any data; you hovered the rest and confirmed they're worse.
3. **Counterfactual** — the agent **read its intent off a precomputed table** and played the matching arm, round after round.

### Next stops in `causalrl`
- `examples/offline_to_online.ipynb` — learning from confounded logs (Manski bounds).
- `identify_effect`, `certify_decision`, `manski_bounds` — the conservative identification layer.
- The docs **Tour by Task** covers transportability, discovery, imitation, curriculum, shaping and games.

*Standalone interactive HTML for projecting/sharing:* `uv run python examples/presentation/_build_html.py`."""))

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

out = Path(__file__).parent / "causal_rl_presentation.ipynb"
nbf.write(nb, out)
print("wrote", out)
