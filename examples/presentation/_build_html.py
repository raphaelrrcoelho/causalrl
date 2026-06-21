"""Build a single standalone, kernel-free interactive HTML from the plotly figures.

    uv run python examples/presentation/_build_html.py

The output ``causal_rl_presentation.html`` keeps hover, zoom and the MABUC play/slider
animation with no Python running — ideal for projecting or handing to the audience.
"""

from __future__ import annotations

from pathlib import Path

import _demos as d
import _viz as v

OUT = Path(__file__).parent / "causal_rl_presentation.html"

_SECTIONS = [
    ("1 — MABUC: watch the belief split by mood",
     "The two arms are identical under intervention (E[Y|do(X=0)]=E[Y|do(X=1)]=0.5), so "
     "choosing well is impossible without reading the gut <i>intuition</i>. The causal agent "
     "keeps one belief per (mood, arm) and pulls them apart; the naive agent pools the moods "
     "and stays glued at 0.5. Press ▶ Play."),
    ("2 — POMIS: touch the 27 levers, keep the 2 that matter",
     "Each of X1, X2, X3 can be left alone or forced to 0/1 → 27 candidate arms. POMIS proves "
     "only {∅, {X3}} can be optimal. Hover any lever to read its true value; note that plain "
     "<i>observing</i> (∅) wins."),
    ("3 — Counterfactual policy: read your intent off the table",
     "Every fixed do(X=a) averages ~0.37, but E[Y_do(a) | intent] is sharp: play the arm "
     "matching your intent (the diagonal)."),
]


def main() -> None:
    _, causal_snaps, naive_snaps, _, _ = d.mabuc_snapshots()
    fig1 = v.belief_animation(d.SNAP_STEPS, causal_snaps, naive_snaps)

    p = d.pomis_data()
    fig2a = v.lever_bar(p["labels"], p["values"], p["in_pomis"], p["optimal"])
    fig2b = v.scoreboard(p["curves"], p["optimal"])

    _, M = d.counterfactual_agent_and_table()
    fig3 = v.decision_heatmap(M)

    figs = [fig1, fig2a, fig2b, fig3]
    headers = [_SECTIONS[0], _SECTIONS[1], ("", ""), _SECTIONS[2]]

    parts = [
        "<html><head><meta charset='utf-8'>",
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:1100px;margin:24px auto;"
        "padding:0 16px;color:#222} h1{font-size:1.6em} h2{margin-top:1.4em} "
        "p{color:#444;line-height:1.5}</style></head><body>",
        "<h1>Causal RL you can <i>see and touch</i></h1>",
        "<p>Three bandit games, one per rung of the Pearl hierarchy. Interactive — hover, zoom, "
        "and press ▶ Play on the first one. No kernel required.</p>",
    ]
    for (title, blurb), fig in zip(headers, figs):
        if title:
            parts.append(f"<h2>{title}</h2><p>{blurb}</p>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if fig is fig1 else False))
    parts.append("</body></html>")

    OUT.write_text("\n".join(parts), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
