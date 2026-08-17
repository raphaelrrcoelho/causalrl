"""Multi-agent causal games (plan §8): typed populations, learning dynamics, certificates.

Builds on the shipped one-shot :class:`~causalrl.games.CausalGame`. :class:`AgentType` /
:class:`Population` are typed agent templates with parameter sharing; :func:`run_no_regret` plays a
population of no-regret learners and returns the realized empirical joint, which
:func:`cce_regret` / :func:`certify_cce_do` turn into a finite-time coarse-correlated-equilibrium
certificate; :class:`EmpiricalGame` is the same game when its payoff table had to be *measured*
rather than given, carrying the standard errors that become the :class:`PayoffError` those
certificates accept; :func:`certify_equilibrium` certifies (robust, possibly intervened)
equilibria with an honest epistemic ``Kind`` capped by the :class:`LearnerTopology` (I2).
:class:`LinearGaussianPopulationEnv` is a synthetic per-agent
:class:`~causalrl.protocols.CausalEnvProtocol` that lets the Phase-1 estimation machinery apply to
one agent inside a fixed population.
"""

from causalrl.magames.cce import (
    CCEPolytope,
    PayoffError,
    cce_bounds,
    cce_polytope,
    cce_regret,
    certify_cce_do,
)
from causalrl.magames.empirical import EmpiricalGame
from causalrl.magames.equilibrium import KindNotLicensedError, certify_equilibrium
from causalrl.magames.learning import NoRegretAlgorithm, NoRegretRun, run_no_regret
from causalrl.magames.population import (
    AgentType,
    LearnerTopology,
    Population,
    topology_max_kind,
)
from causalrl.magames.views import LinearGaussianPopulationEnv, linear_gaussian_population_env

__all__ = [
    "AgentType",
    "CCEPolytope",
    "EmpiricalGame",
    "KindNotLicensedError",
    "LearnerTopology",
    "LinearGaussianPopulationEnv",
    "NoRegretAlgorithm",
    "NoRegretRun",
    "PayoffError",
    "Population",
    "cce_bounds",
    "cce_polytope",
    "cce_regret",
    "certify_cce_do",
    "certify_equilibrium",
    "linear_gaussian_population_env",
    "run_no_regret",
    "topology_max_kind",
]
