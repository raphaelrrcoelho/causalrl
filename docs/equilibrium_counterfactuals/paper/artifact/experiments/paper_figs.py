"""Figures for paper 1 (the CLeaR-shaped causal-semantics paper).

Regenerates everything deterministically from the E4/E7 experiment code (same seeds as the
2026-07-15 runs recorded in RESULTS.md) and writes PDFs into
docs/equilibrium_counterfactuals/paper/figs/.

Fig e4: the certified sign flip — stability margin vs the policy coefficient phi (crossing zero
at phi*), and naive-expectations learning paths vs the equilibrium do() predictions.
Fig e7: bistable mean field under do(u) — locally certified roots, basin-boundary shift, and the
selection effect as a function of initial dispersion.

Run:  cd experiments/eqcf && uv run python paper_figs.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import e4_macro_loop as e4
import e7_basins as e7

OUT = Path(__file__).resolve().parents[1] / "figs"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

C_EQ = "#0072B2"  # equilibrium blue
C_LEARN = "#D55E00"  # learning orange
C_GRAY = "#666666"


def fig_e4() -> None:
    phi_star = 1.0 - (1.0 - e4.BETA) / (e4.KAPPA * e4.SIGMA)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.0))

    # Panel A: stability margin vs phi.
    phis = np.linspace(0.3, 1.8, 300)
    margins = [e4.macro_scm(p).stability_margin() for p in phis]
    ax1.axhline(0.0, color=C_GRAY, lw=0.8)
    ax1.axvline(phi_star, color=C_GRAY, lw=0.8, ls=":")
    ax1.plot(phis, margins, color="black", lw=1.4)
    ax1.fill_between(phis, margins, 0, where=np.array(margins) > 0, color=C_EQ, alpha=0.12)
    ax1.fill_between(phis, margins, 0, where=np.array(margins) < 0, color=C_LEARN, alpha=0.15)
    for phi, va in ((0.5, "top"), (1.5, "bottom")):
        m = e4.macro_scm(phi).stability_margin()
        ax1.plot([phi], [m], "o", color="black", ms=4)
        ax1.annotate(
            rf"$\varphi={phi}$", (phi, m), textcoords="offset points",
            xytext=(6, -10 if va == "top" else 6),
        )
    ax1.annotate(
        rf"$\varphi^*={phi_star:.3f}$", (phi_star, ax1.get_ylim()[0]),
        textcoords="offset points", xytext=(4, 8), color=C_GRAY,
    )
    ax1.set_xlabel(r"policy coefficient $\varphi$")
    ax1.set_ylabel("stability margin")
    ax1.set_title("(a) E-stability margin of the intervened SCM")

    # Panel B: learning paths vs equilibrium predictions under do(u = +1).
    steps = np.arange(1, 31)
    for phi, color in ((1.5, C_EQ), (0.5, C_LEARN)):
        t = e4.t_slope(phi)
        eq_pi = e4.SHOCK / (1.0 - t)
        path = e4.learning_path(phi, e4.SHOCK)
        ax2.plot(steps, path, color=color, lw=1.4, label=rf"learning path, $\varphi={phi}$")
        ax2.axhline(eq_pi, color=color, lw=1.0, ls="--")
        ax2.annotate(
            rf"equilibrium $do$: $\pi^*={eq_pi:+.1f}$",
            (steps[-1], eq_pi), ha="right",
            textcoords="offset points", xytext=(0, 4 if phi == 1.5 else -11), color=color,
        )
    ax2.axhline(0.0, color=C_GRAY, lw=0.8)
    ax2.set_yscale("symlog", linthresh=10.0)
    ax2.set_xlabel("learning step $k$")
    ax2.set_ylabel(r"inflation response $\pi_k$")
    ax2.set_title(r"(b) do(u=+1): the sign of the policy conclusion flips")
    ax2.legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_e4.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_e4.pdf'}")


def fig_e7() -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10.8, 3.0))

    # Panel A: mean field and roots under u=0 vs do(u=0.2).
    xs = np.linspace(-2.0, 2.0, 600)
    for u, color, label in ((0.0, C_GRAY, "$u=0$"), (0.2, C_EQ, "$do(u=0.2)$")):
        ax1.plot(xs, e7.mean_field(xs, u), color=color, lw=1.4, label=label)
        for root, margin in e7.equilibria(u):
            stable = margin > 0
            ax1.plot(
                [root], [0.0], "o" if stable else "o", ms=5,
                mfc=color if stable else "white", mec=color,
            )
    ax1.axhline(0.0, color="black", lw=0.6)
    ax1.set_xlabel("$x$")
    ax1.set_ylabel(r"mean field $F(x)=\tanh(3x)-x+u$")
    ax1.set_title("(a) roots move and the basin boundary moves")
    ax1.legend(loc="lower right", frameon=False)

    # Panels B/C: selection effect vs initial dispersion (same seeds as RESULTS.md).
    spreads = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    moved, mix_after = [], []
    tracked = e7.equilibria(0.2)[-1][0]
    for spread in spreads:
        b = e7.ensemble_basin_mass(0.0, spread=spread)
        s = e7.ensemble_basin_mass(0.2, spread=spread)
        moved.append(100.0 * (s["positive"] - b["positive"]))
        mix_after.append(s["negative"] * s["mean_neg"] + s["positive"] * s["mean_pos"])
        print(f"spread {spread}: moved {moved[-1]:+.1f}%  mixture {mix_after[-1]:+.3f}")

    ax2.plot(spreads, moved, "o-", color=C_LEARN, lw=1.4, ms=4)
    ax2.set_xlabel(r"initial dispersion $s_0$  ($x_0\sim\mathcal{N}(0,s_0^2)$)")
    ax2.set_ylabel("mass crossing the boundary (%)")
    ax2.set_title("(b) selection: basin mass moved by $do(u=0.2)$")
    ax2.set_ylim(bottom=0)

    ax3.plot(spreads, mix_after, "o-", color=C_LEARN, lw=1.4, ms=4, label="population mixture mean")
    ax3.axhline(tracked, color=C_EQ, lw=1.2, ls="--", label=rf"tracked root $x_+^*={tracked:+.2f}$")
    ax3.set_xlabel(r"initial dispersion $s_0$")
    ax3.set_ylabel(r"$\mathbb{E}[x_\infty]$ under $do(u=0.2)$")
    ax3.set_title("(c) the gap no local certificate can see")
    ax3.set_ylim(0.0, 1.35)
    ax3.legend(loc="center right", frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_e7.pdf")
    plt.close(fig)
    print(f"wrote {OUT / 'fig_e7.pdf'}")


if __name__ == "__main__":
    fig_e4()
    fig_e7()
