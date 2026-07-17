"""Equilibrium certificates for causal games (plan §8.1).

:func:`certify_equilibrium` certifies that a pure profile is a (tol-)Nash equilibrium of a
:class:`~eqcert.games.CausalGame`, optionally **under an intervention** ``do`` that fixes some
agents' actions (a policy intervention): the profile is *robust* iff every free agent still best-
responds when the intervened agents are forced. The check is exact on finite games (best responses),
so a single-learner topology may claim ``IDENTIFIED``; other topologies are capped at ``EMPIRICAL``
(I2). Requesting a ``Kind`` the topology does not license is refused up front (I2 / acceptance d) —
never silently downgraded to a mislabelled certificate.
"""

from __future__ import annotations

from collections.abc import Mapping

from eqcert.certify.certificate import (
    Assumption,
    Certificate,
    EstimandSpec,
    Hedge,
    Kind,
    Provenance,
    Witness,
)
from eqcert.games import CausalGame
from eqcert.graphs import graph_hash
from eqcert.magames.population import LearnerTopology, topology_max_kind

__all__ = ["KindNotLicensedError", "certify_equilibrium"]


class KindNotLicensedError(TypeError):
    """Raised when a stronger ``Kind`` is requested than the learner topology licenses (I2)."""


def _max_regret(
    game: CausalGame, profile_eff: Mapping[str, int], free_agents: list[str]
) -> tuple[str | None, float]:
    """Largest unilateral-deviation gain among ``free_agents`` at the effective profile."""
    order = game.agents
    base = [profile_eff[a] for a in order]
    worst_agent: str | None = None
    worst_gain = 0.0
    for agent in free_agents:
        idx = order.index(agent)
        current = game.utilities[agent][tuple(base)]
        best = max(
            game.utilities[agent][tuple([*base[:idx], act, *base[idx + 1 :]])]
            for act in game.actions[agent]
        )
        gain = best - current
        if gain > worst_gain:
            worst_gain, worst_agent = gain, agent
    return worst_agent, worst_gain


def certify_equilibrium(
    game: CausalGame,
    profile: Mapping[str, int],
    *,
    do: Mapping[str, int] | None = None,
    topology: LearnerTopology = LearnerTopology.SINGLE_LEARNER,
    tol: float = 1e-9,
    require_kind: Kind | None = None,
) -> Certificate:
    """Certify ``profile`` is a robust (tol-)Nash equilibrium of ``game``, optionally under ``do``.

    ``do`` fixes the given agents' actions; the remaining *free* agents must each best-respond
    within ``tol`` for the profile to be certified. ``kind`` is capped by ``topology`` (I2). If
    ``require_kind`` is stronger than the topology licenses (e.g. ``IDENTIFIED`` under
    ``INDEPENDENT_LEARNERS``), raise :class:`KindNotLicensedError` — the requested guarantee is not
    available, so no certificate is returned (acceptance d).
    """
    kind = topology_max_kind(topology)
    if require_kind is Kind.IDENTIFIED and kind is not Kind.IDENTIFIED:
        raise KindNotLicensedError(
            f"topology {topology.value!r} licenses at most {kind.name}, not IDENTIFIED; "
            "single-agent identification does not hold for simultaneous/centralised learners"
        )

    do = dict(do or {})
    profile_eff = {a: do.get(a, profile[a]) for a in game.agents}
    free_agents = [a for a in game.agents if a not in do]
    deviator, regret = _max_regret(game, profile_eff, free_agents)
    certified = regret <= tol

    assumptions = (
        Assumption(name="finite-game", params={"agents": list(game.agents)}, checkable=True),
        Assumption(name="tol", params={"tol": tol}, checkable=True),
    )
    provenance = Provenance.create(graph_hash=graph_hash(game.graph))
    intervened = f" | do={dict(do)}" if do else ""
    if not certified:
        return Certificate(
            claim=f"profile {dict(profile)} is NOT a robust equilibrium{intervened}",
            estimand=EstimandSpec(query="equilibrium", target="mean"),
            kind=kind,
            value=None,
            alpha=None,
            assumptions=assumptions,
            method="best-response-exact",
            witness=None,
            hedge=Hedge(
                reason="not-an-equilibrium",
                detail={"deviating_agent": deviator, "regret": regret, "do": dict(do)},
            ),
            provenance=provenance,
        )
    return Certificate(
        claim=f"profile {dict(profile)} is a robust equilibrium{intervened}",
        estimand=EstimandSpec(query="equilibrium", target="mean"),
        kind=kind,
        value=None,
        alpha=None,
        assumptions=assumptions,
        method="best-response-exact",
        witness=Witness(
            kind="nash-equilibrium",
            detail={"profile": dict(profile), "do": dict(do), "max_regret": regret},
        ),
        hedge=None,
        provenance=provenance,
    )
