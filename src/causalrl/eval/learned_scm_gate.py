"""Oracle kill gate for fit_scm: does the learned SCM get interventions right?

The baseline is the sharp one. A *complete* DAG on the reversed topological order is saturated, so
it reproduces the observational distribution exactly — yet it implies different interventions. Any
gap between the two therefore isolates the contribution of causal structure, not of fit quality.

Ground truth (``build_discovery_scm``): ``W`` is a noisy copy of ``Z``, so ``E[W | do(Z=1)] = 0.9``.
Under the reversed structure ``W`` is a parent of ``Z``, so ``do(Z=1)`` leaves it at
``E[W] = 0.66``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from causalrl.envs.suite.discovery import build_discovery_scm
from causalrl.scm.fit import fit_scm
from causalrl.scm.graph import CausalGraph

_TRUE_EDGES = [("X", "Z"), ("Y", "Z"), ("Z", "W")]
_REVERSED_ORDER = ["W", "Z", "X", "Y"]
_ORACLE_E_W_DO_Z1 = 0.9


class OracleGateResult(NamedTuple):
    """Mean absolute do-query error of each model, and the verdict."""

    causal_error: float
    correlational_error: float
    gap: float
    passed: bool

    def summary(self) -> str:
        return (
            f"causal={self.causal_error:.4f}  correlational={self.correlational_error:.4f}  "
            f"gap={self.gap:+.4f}  passed={self.passed}"
        )


def _complete_dag(order: Sequence[str]) -> CausalGraph:
    """Saturated DAG on ``order``: every node's parents are all earlier nodes."""
    edges = [(order[i], order[j]) for j in range(len(order)) for i in range(j)]
    return CausalGraph(directed_edges=edges, nodes=list(order))


def run_learned_scm_oracle_gate(
    *, seeds: Sequence[int] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), n: int = 20_000
) -> OracleGateResult:
    """Fit on observational draws, then score ``E[W | do(Z=1)]`` against the oracle 0.9.

    PASS = the true-graph fit lands within 0.03 of the oracle while the L1-equivalent complete DAG
    on the reversed order stays above 0.15 off, on every seed.
    """
    truth = build_discovery_scm()
    causal_errors: list[float] = []
    correlational_errors: list[float] = []
    for seed in seeds:
        observed = {k: v.numpy() for k, v in truth.see(n, seed=seed).items()}
        data = {k: np.asarray(v).astype(int) for k, v in observed.items()}

        causal = fit_scm(data, graph=CausalGraph(directed_edges=_TRUE_EDGES), seed=seed)
        correlational = fit_scm(data, graph=_complete_dag(_REVERSED_ORDER), seed=seed)

        for model, sink in ((causal, causal_errors), (correlational, correlational_errors)):
            predicted = float(model.do({"Z": 1.0}).see(n, seed=seed)["W"].mean())
            sink.append(abs(predicted - _ORACLE_E_W_DO_Z1))

    causal_error = float(np.mean(causal_errors))
    correlational_error = float(np.mean(correlational_errors))
    return OracleGateResult(
        causal_error=causal_error,
        correlational_error=correlational_error,
        gap=correlational_error - causal_error,
        passed=bool(max(causal_errors) < 0.03 and min(correlational_errors) > 0.15),
    )
