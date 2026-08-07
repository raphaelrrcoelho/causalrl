"""Off-policy evaluation: what would this policy have been worth on these logs?

One home for the question an RL practitioner actually asks of causal machinery. The surface runs
from the plain reweighting estimate to the certificate a deployment decision can rest on:

* :mod:`.ipw` — :func:`~causalrl.ope.ipw.ipw_value` (in-memory inverse-propensity weighting) and
  :func:`~causalrl.ope.ipw.stream_policy_value` (the self-normalised streaming certificate).
* :mod:`.sequential` — :func:`~causalrl.ope.sequential.estimate_sequential_value` /
  :func:`~causalrl.ope.sequential.certify_sequential_value`: finite-horizon g-computation and the
  cross-fitted sequentially doubly-robust (LTMLE-style) estimator, under sequential ignorability.
* :mod:`.bounds` — what the value could be when the logs are *confounded*: Manski
  :func:`~causalrl.ope.bounds.causal_q_bounds` and the marginal-sensitivity-model family
  (:func:`~causalrl.ope.bounds.msm_policy_value_bounds` and friends).
* :mod:`.certify` — :func:`~causalrl.ope.certify.certify_policy`, the decision at the end: does a
  learned policy's improvement over the behaviour policy survive hidden confounding?

Every name below is also a top-level export (``from causalrl import msm_policy_value_bounds``),
which is the canonical spelling; this package is the second one a reader reasonably reaches for.

Re-exports resolve lazily through module ``__getattr__``, and not as a style preference: importing
:mod:`causalrl.ope.bounds` executes *this* module first, and
:mod:`causalrl.identification.decision` imports from ``ope.bounds`` while ``ope.certify`` imports
from ``identification.decision``. Binding either eagerly here would close that loop into an
ImportError on some import orders. Adding a name means adding it to ``_LAZY`` *and* ``__all__``
together: a name in only one is either an export that cannot resolve or one ``dir()`` hides.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from causalrl.ope.bounds import (
        causal_q_bounds,
        ipw_sensitivity_bounds,
        msm_contribution_bounds,
        msm_per_step_bounds,
        msm_policy_value_bounds,
        msm_stratified_bounds,
    )
    from causalrl.ope.certify import certify_policy
    from causalrl.ope.ipw import ipw_value, stream_policy_value
    from causalrl.ope.sequential import (
        SequentialValueEstimate,
        certify_sequential_value,
        estimate_sequential_value,
        sequential_ice_values,
    )
    from causalrl.ope.sequential_test import (
        ConfidenceSequence,
        SequentialVerdict,
        confidence_sequence,
        sequential_policy_comparison,
    )

# name -> (submodule, attribute); resolved on first attribute access.
_LAZY: dict[str, tuple[str, str]] = {
    "SequentialValueEstimate": ("causalrl.ope.sequential", "SequentialValueEstimate"),
    "causal_q_bounds": ("causalrl.ope.bounds", "causal_q_bounds"),
    "certify_policy": ("causalrl.ope.certify", "certify_policy"),
    "certify_sequential_value": ("causalrl.ope.sequential", "certify_sequential_value"),
    "estimate_sequential_value": ("causalrl.ope.sequential", "estimate_sequential_value"),
    "ipw_sensitivity_bounds": ("causalrl.ope.bounds", "ipw_sensitivity_bounds"),
    "ipw_value": ("causalrl.ope.ipw", "ipw_value"),
    "msm_contribution_bounds": ("causalrl.ope.bounds", "msm_contribution_bounds"),
    "msm_per_step_bounds": ("causalrl.ope.bounds", "msm_per_step_bounds"),
    "msm_policy_value_bounds": ("causalrl.ope.bounds", "msm_policy_value_bounds"),
    "msm_stratified_bounds": ("causalrl.ope.bounds", "msm_stratified_bounds"),
    "sequential_ice_values": ("causalrl.ope.sequential", "sequential_ice_values"),
    "ConfidenceSequence": ("causalrl.ope.sequential_test", "ConfidenceSequence"),
    "SequentialVerdict": ("causalrl.ope.sequential_test", "SequentialVerdict"),
    "confidence_sequence": ("causalrl.ope.sequential_test", "confidence_sequence"),
    "sequential_policy_comparison": (
        "causalrl.ope.sequential_test",
        "sequential_policy_comparison",
    ),
    "stream_policy_value": ("causalrl.ope.ipw", "stream_policy_value"),
}

__all__ = [
    "ConfidenceSequence",
    "SequentialValueEstimate",
    "SequentialVerdict",
    "causal_q_bounds",
    "certify_policy",
    "certify_sequential_value",
    "confidence_sequence",
    "estimate_sequential_value",
    "ipw_sensitivity_bounds",
    "ipw_value",
    "msm_contribution_bounds",
    "msm_per_step_bounds",
    "msm_policy_value_bounds",
    "msm_stratified_bounds",
    "sequential_ice_values",
    "sequential_policy_comparison",
    "stream_policy_value",
]


def __getattr__(name: str) -> object:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(_import_module(module_name), attribute)
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
