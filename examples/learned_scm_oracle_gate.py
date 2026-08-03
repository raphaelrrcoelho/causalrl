"""Oracle kill gate for fit_scm -- 10 seeds, full n.

    uv run python examples/learned_scm_oracle_gate.py

Fits an SCM on observational draws from a known world, then asks both it and an L1-equivalent
wrong-structure model for E[W | do(Z=1)]. The wrong-structure model is a *saturated* DAG on the
reversed order: it reproduces the observational distribution exactly, so any gap is attributable
to causal structure alone, not to fit quality.
"""

from __future__ import annotations

from causalrl.eval.learned_scm_gate import run_learned_scm_oracle_gate


def main() -> None:
    result = run_learned_scm_oracle_gate()
    print(result.summary())
    print("oracle E[W | do(Z=1)] = 0.9; the reversed-order model answers with E[W] = 0.66")


if __name__ == "__main__":
    main()
