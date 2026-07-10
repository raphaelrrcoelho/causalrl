"""Plan §8 acceptance (b): single-learner-in-population OPE (Phase-1 DR) matches MC ground truth.

The per-agent view exposes one ego agent inside a fixed population as a CausalEnvProtocol; its
logged play is confounded by an observed context, so the Phase-1 back-door DR estimator recovers the
ego's action effect — and it agrees with the Monte-Carlo truth obtained by intervening (``do``).
numpy; fully local.
"""

from __future__ import annotations

from causalrl.estimate.compiler import certify_effect
from causalrl.magames.views import PopulationAgentView, agent_causal_env_view
from causalrl.protocols import CausalEnvProtocol
from causalrl.scm.graph import CausalGraph


def _view_graph() -> CausalGraph:
    return CausalGraph(
        directed_edges=[("Z", "ego"), ("Z", "Y"), ("Z", "co"), ("ego", "Y"), ("co", "Y")],
        nodes=["Z", "ego", "co", "Y"],
    )


def test_view_conforms_to_causal_env_protocol() -> None:
    view = agent_causal_env_view()
    assert isinstance(view, CausalEnvProtocol)
    log = view.sample(50, seed=0)
    assert set(log.column("name").tolist()) == {"Z", "ego", "co", "Y"}
    assert view.noise_ledger() is None  # white-box ledger not registered for this view


def test_single_learner_ope_matches_monte_carlo() -> None:
    view = PopulationAgentView(ego_effect=1.5, coplayer_effect=0.8, confound=1.0, noise=0.5)

    # Phase-1 DR on the confounded logged play (adjust for the observed context via the graph).
    _, frame = view.sample(8000, seed=0).pivot()
    data = {k: frame[k] for k in ("Z", "ego", "co", "Y")}
    cert = certify_effect(_view_graph(), "ego", "Y", data, method="dml", seed=0)
    assert cert.value is not None and cert.hedge is None

    # Monte-Carlo ground truth: intervene on the ego and difference the mean rewards.
    y1 = view.do({"ego": 1}, 40000, seed=1).values_by_name("Y").astype(float)
    y0 = view.do({"ego": 0}, 40000, seed=2).values_by_name("Y").astype(float)
    mc_ate = float(y1.mean() - y0.mean())

    assert abs(mc_ate - 1.5) < 0.1  # MC recovers the ego action effect
    assert abs(cert.value - mc_ate) < 0.1  # DR estimate matches MC within tolerance
    assert cert.ci.lower <= 1.5 <= cert.ci.upper
