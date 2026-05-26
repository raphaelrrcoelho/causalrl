"""causalrl: causal reinforcement learning.

Structural causal models (Pearl's Causal Hierarchy: ``see`` / ``do`` / ``counterfactual``)
and the causal RL algorithms built on top, organized around the 9-task taxonomy of causal RL
(https://crl.causalai.net/).

The curated names below are the public API and may be imported directly from ``causalrl``::

    from causalrl import DOVI, StructuralCausalModel, DTREnv, generate_logs

The full module paths (e.g. ``causalrl.agents.dovi``) remain importable and unchanged.
"""

from importlib.metadata import version as _pkg_version

from causalrl.agents.bandits import CausalThompsonSampling, NaiveThompsonSampling
from causalrl.agents.base import Agent
from causalrl.agents.baselines import NaiveOffline, OnlineOnlyUCB
from causalrl.agents.deep_deconfounded import DeepDeconfoundedQ
from causalrl.agents.dovi import DOVI
from causalrl.agents.offline_online import UCDTR
from causalrl.data.dataset import ConfoundedTrajectoryDataset, Transition, generate_logs
from causalrl.envs.base import CausalEnv, ConfoundedMDP
from causalrl.envs.suite.dtr import DTREnv
from causalrl.envs.suite.gridworld import ConfoundedGridworld
from causalrl.envs.suite.mabuc import MABUCEnv
from causalrl.envs.suite.seq_dtr import SequentialDTREnv
from causalrl.envs.suite.seq_mabuc import SequentialMABUCEnv
from causalrl.eval.harness import run_episodes
from causalrl.eval.metrics import cumulative_regret, finite_horizon_regret
from causalrl.eval.ope import confounding_sensitivity_bounds, ipw_value
from causalrl.exceptions import (
    CausalGraphError,
    CausalRLError,
    NotIdentifiableError,
    RealizabilityError,
)
from causalrl.identification.bounds import causal_q_bounds
from causalrl.identification.criteria import backdoor_adjustment_set, is_identifiable
from causalrl.scm.graph import CausalGraph
from causalrl.scm.mechanisms import (
    FunctionalMechanism,
    LinearGaussianMechanism,
    Mechanism,
    NeuralMechanism,
)
from causalrl.scm.scm import StructuralCausalModel

__version__ = _pkg_version("causalrl")

__all__ = [
    "DOVI",
    "UCDTR",
    "Agent",
    "CausalEnv",
    "CausalGraph",
    "CausalGraphError",
    "CausalRLError",
    "CausalThompsonSampling",
    "ConfoundedGridworld",
    "ConfoundedMDP",
    "ConfoundedTrajectoryDataset",
    "DTREnv",
    "DeepDeconfoundedQ",
    "FunctionalMechanism",
    "LinearGaussianMechanism",
    "MABUCEnv",
    "Mechanism",
    "NaiveOffline",
    "NaiveThompsonSampling",
    "NeuralMechanism",
    "NotIdentifiableError",
    "OnlineOnlyUCB",
    "RealizabilityError",
    "SequentialDTREnv",
    "SequentialMABUCEnv",
    "StructuralCausalModel",
    "Transition",
    "__version__",
    "backdoor_adjustment_set",
    "causal_q_bounds",
    "confounding_sensitivity_bounds",
    "cumulative_regret",
    "finite_horizon_regret",
    "generate_logs",
    "ipw_value",
    "is_identifiable",
    "run_episodes",
]
