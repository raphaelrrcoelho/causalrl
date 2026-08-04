"""Smoke test for examples/learned_scm_policy.py.

Loads the example as a module and runs ``main()`` on a tiny budget, so the whole
fit -> world-model env -> plan -> score path executes without the full run's cost. The
assertions pin the example's *point*: on a confounded log the two world models disagree about
which arm is best, and only the causal one agrees with the truth.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


def _load_example_module() -> Any:
    """Load examples/learned_scm_policy.py without executing its __main__ block."""
    example_path = Path(__file__).parent.parent / "examples" / "learned_scm_policy.py"
    spec = importlib.util.spec_from_file_location("learned_scm_policy", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_learned_scm_policy_example_smoke() -> None:
    module = _load_example_module()
    results = module.main(n_logs=4_000, steps=40, seeds=(0,), n_mc=400)

    assert len(results) == 2
    confounded, randomized = results
    assert confounded.regime == "confounded log"
    assert randomized.regime.startswith("randomized log")

    # The logs set the trap: the naive conditional prefers action 0.
    assert confounded.empirical_conditional[0] > confounded.empirical_conditional[1]

    causal, blind = confounded.models
    assert causal.reward_parents == ("Z", "A")
    assert blind.reward_parents == ("A",)
    # Same rows, same fitter -- opposite verdicts, because of structure alone.
    assert causal.arm_values[2] > causal.arm_values[1]  # do(A=1) beats do(A=0): correct
    assert blind.arm_values[1] > blind.arm_values[2]  # ...and the blind model reverses it

    # On a randomized log the trap is gone and both models get the ordering right.
    for model in randomized.models:
        assert model.arm_values[2] > model.arm_values[1]
