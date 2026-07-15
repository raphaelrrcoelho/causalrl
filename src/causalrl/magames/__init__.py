"""Multi-agent causal games (plan §8): typed populations, equilibrium certificates, per-agent views.

Builds on the shipped one-shot :class:`~causalrl.games.CausalGame`. :class:`AgentType` /
:class:`Population` are typed agent templates with parameter sharing; :func:`certify_equilibrium`
certifies (robust, possibly intervened) equilibria with an honest epistemic ``Kind`` capped by the
:class:`LearnerTopology` (I2). Per-agent :class:`~causalrl.protocols.CausalEnvProtocol` views (added
next) let the Phase-1 estimation machinery apply to one agent inside a population.
"""

from causalrl.magames.cce import (
    CCEPolytope,
    cce_bounds,
    cce_polytope,
    cce_regret,
    certify_cce_do,
)
from causalrl.magames.equilibrium import KindNotLicensedError, certify_equilibrium
from causalrl.magames.population import (
    AgentType,
    LearnerTopology,
    Population,
    topology_max_kind,
)
from causalrl.magames.views import PopulationAgentView, agent_causal_env_view

__all__ = [
    "AgentType",
    "CCEPolytope",
    "KindNotLicensedError",
    "LearnerTopology",
    "Population",
    "PopulationAgentView",
    "agent_causal_env_view",
    "cce_bounds",
    "cce_polytope",
    "cce_regret",
    "certify_cce_do",
    "certify_equilibrium",
    "topology_max_kind",
]
