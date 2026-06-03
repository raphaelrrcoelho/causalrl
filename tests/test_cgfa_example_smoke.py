"""Smoke test for examples/cgfa_ppo_example.py.

Imports the example as a module and calls main() with a tiny budget so the full
execution path is exercised without a long training run.

stable-baselines3 is an optional dependency (in ``causalrl[examples]``) and is
**not** required in CI.  The test is skipped automatically when SB3 is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example_module() -> object:
    """Load examples/cgfa_ppo_example.py as a module without executing its __main__ block."""
    example_path = Path(__file__).parent.parent / "examples" / "cgfa_ppo_example.py"
    spec = importlib.util.spec_from_file_location("cgfa_ppo_example", example_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def test_cgfa_example_smoke() -> None:
    """Run cgfa_ppo_example.main() with a minimal timestep budget.

    Skipped when stable-baselines3 is not installed.
    """
    import pytest

    pytest.importorskip("stable_baselines3")

    module = _load_example_module()
    main = module.main  # type: ignore[attr-defined]
    # Tiny budget: fast on any machine; n_mc=50 speeds env construction.
    main(total_timesteps=32, eval_steps=4, n_mc=50)
