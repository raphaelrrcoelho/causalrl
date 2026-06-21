"""Plotly figure builders for the see-and-touch presentation.

Pure: every function takes already-computed data (from ``_demos``) and returns a
``plotly.graph_objects.Figure``. No training, no globals. The figures are interactive
(hover, zoom) and the MABUC one animates over training so you can *watch the belief form*;
all of it survives ``fig.write_html`` as a standalone, kernel-free artifact.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import beta as beta_dist

BLUE, GOLD, RED, GREY = "#1f77b4", "#e8a33d", "#d62728", "#9aa0a6"
_XS = np.linspace(0.001, 0.999, 240)


# --------------------------------------------------------------------------------------
# Demo 1 — animated belief densities (play / slider over training)
# --------------------------------------------------------------------------------------
def belief_animation(snap_steps, causal_snaps, naive_snaps):
    """Three panels (causal|intuition=0, causal|intuition=1, naive); animate over pulls."""
    titles = ("CAUSAL · intuition = 0", "CAUSAL · intuition = 1", "NAIVE · (moods pooled)")
    axes = ("x", "x2", "x3"), ("y", "y2", "y3")

    def curve(a, b, color, name, col, show):
        return go.Scatter(
            x=_XS, y=beta_dist.pdf(_XS, a, b), mode="lines",
            line=dict(color=color, width=2.5), name=name, legendgroup=name,
            showlegend=show, xaxis=axes[0][col], yaxis=axes[1][col],
            hovertemplate=name + "<br>believed P(reward)=%{x:.2f}<extra></extra>",
        )

    def frame_traces(step, show_legend):
        ca, cb = causal_snaps[step]
        na, nb = naive_snaps[step]
        return [
            curve(ca[0, 0], cb[0, 0], BLUE, "arm 0", 0, show_legend),
            curve(ca[0, 1], cb[0, 1], GOLD, "arm 1", 0, show_legend),
            curve(ca[1, 0], cb[1, 0], BLUE, "arm 0", 1, False),
            curve(ca[1, 1], cb[1, 1], GOLD, "arm 1", 1, False),
            curve(na[0], nb[0], BLUE, "arm 0", 2, False),
            curve(na[1], nb[1], GOLD, "arm 1", 2, False),
        ]

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles, horizontal_spacing=0.06)
    for tr in frame_traces(snap_steps[0], True):
        fig.add_trace(tr)
    fig.frames = [go.Frame(data=frame_traces(s, True), name=str(s)) for s in snap_steps]

    steps = [dict(method="animate", label=str(s),
                  args=[[str(s)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True),
                                       transition=dict(duration=0))])
             for s in snap_steps]
    fig.update_layout(
        height=440, template="plotly_white",
        title=dict(text="Watch the belief form — drag the slider, or press ▶ Play", x=0.02, y=0.97),
        margin=dict(t=110, b=40),
        updatemenus=[dict(type="buttons", showactive=False, x=1.0, y=1.22, xanchor="right",
                          direction="right",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=550, redraw=True),
                                                    transition=dict(duration=200),
                                                    fromcurrent=True)]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], dict(mode="immediate",
                                                      frame=dict(duration=0, redraw=False))]),
                          ])],
        sliders=[dict(active=0, x=0.12, len=0.85, currentvalue=dict(prefix="pulls: "),
                      steps=steps)],
    )
    for ax in ("xaxis", "xaxis2", "xaxis3"):
        fig.layout[ax].update(range=[0, 1], title="believed P(reward)")
    for ax in ("yaxis", "yaxis2", "yaxis3"):
        fig.layout[ax].update(range=[0, 14])
    return fig


# --------------------------------------------------------------------------------------
# Demo 2 — touch the 27 levers (hover = value + POMIS membership) + learning scoreboard
# --------------------------------------------------------------------------------------
def lever_bar(labels, values, in_pomis, optimal):
    order = sorted(range(len(values)), key=lambda i: values[i])
    y = [labels[i] for i in order]
    x = [values[i] for i in order]
    colors = [BLUE if in_pomis[i] else GREY for i in order]
    tags = ["POMIS ✓" if in_pomis[i] else "pruned" for i in order]
    fig = go.Figure(go.Bar(
        x=x, y=y, orientation="h", marker_color=colors,
        customdata=tags,
        hovertemplate="%{y}<br>value = %{x:.3f}<br>%{customdata}<extra></extra>",
    ))
    fig.add_vline(x=optimal, line=dict(color="black", dash="dash", width=1),
                  annotation_text="optimal", annotation_position="top")
    fig.update_layout(
        height=620, template="plotly_white",
        title="Hover any lever — blue = in a POMIS set {∅, {X3}}, grey = pruned",
        xaxis_title="true value  E[reward]", margin=dict(l=180, t=70),
    )
    return fig


def scoreboard(curves, optimal):
    fig = go.Figure()
    palette = {"POMIS (2 arms)": BLUE, "brute force (27 arms)": GREY, "naive do(X3)": RED}
    for name, curve in curves.items():
        step = max(1, len(curve) // 400)
        xs = np.arange(0, len(curve), step)
        fig.add_trace(go.Scatter(x=xs, y=curve[::step], mode="lines", name=name,
                                 line=dict(color=palette.get(name))))
    fig.add_hline(y=optimal, line=dict(color="black", dash="dash", width=1))
    fig.update_layout(height=380, template="plotly_white",
                      title="Knowing where to intervene = converge immediately",
                      xaxis_title="step", yaxis_title="running avg reward")
    return fig


# --------------------------------------------------------------------------------------
# Demo 3 — counterfactual decision table (hover-able heatmap)
# --------------------------------------------------------------------------------------
def decision_heatmap(M):
    arms = ["do(X=0)", "do(X=1)", "do(X=2)"]
    intents = ["intent=0", "intent=1", "intent=2"]
    fig = go.Figure(go.Heatmap(
        z=M, x=arms, y=intents, colorscale="Viridis", zmin=0.1, zmax=0.85,
        text=[[f"{v:.2f}" for v in row] for row in M], texttemplate="%{text}",
        hovertemplate="%{y}, %{x}<br>E[Y_do(a)|intent] = %{z:.2f}<extra></extra>",
        colorbar=dict(title="E[Y]"),
    ))
    fig.update_layout(height=420, template="plotly_white",
                      title="E[Y_do(a) | intent] — argmax per row is the policy (the diagonal)")
    fig.update_yaxes(autorange="reversed")
    return fig
