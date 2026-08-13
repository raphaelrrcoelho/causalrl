"""Regressions for the two anti-conservative bugs the adversarial review reproduced."""
from __future__ import annotations

import numpy as np

from causalrl.scm.fitters import _r2


def test_r2_refuses_to_score_a_zero_variance_target_as_perfect() -> None:
    """A single-row holdout has no variance; a flat 1.0 there hides any error at all."""
    y = np.array([10.0])
    assert np.isnan(_r2(y, np.array([999.0]))), "a wrong prediction must not score 1.0"
    assert _r2(y, np.array([10.0])) == 1.0, "an exact prediction is still a perfect fit"


def test_functional_manski_divides_by_per_action_fold_counts() -> None:
    """An action absent from a fold must not have its outcome prediction shrunk toward zero."""
    from causalrl.bounds.functional import FunctionalManskiBounds

    class Constant:
        def __init__(self, value: float) -> None:
            self.value = value

        def predict(self, x):  # noqa: ANN001, ANN202
            return np.full(len(x), self.value)

        def predict_proba(self, x):  # noqa: ANN001, ANN202
            return np.full(len(x), 1.0)

    bounds = FunctionalManskiBounds(n_actions=2)
    # Action 1 is fitted in one fold only; action 0 in both.
    models = [
        ({0: Constant(0.5), 1: Constant(0.95)}, {0: Constant(1.0), 1: Constant(1.0)}),
        ({0: Constant(0.5)}, {0: Constant(1.0)}),
    ]
    mu, _ = bounds._predict(np.zeros((3, 1)), models)
    assert np.allclose(mu[:, 0], 0.5)
    assert np.allclose(mu[:, 1], 0.95), f"action 1 shrunk to {mu[0, 1]} by the absent fold"
