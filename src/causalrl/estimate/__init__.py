"""Identification-aware estimation of identified functionals (plan §7.2).

The continuous/DR counterpart to the shipped discrete estimand evaluator
(:func:`causalrl.identification.id_algorithm.estimate_effect`): it compiles a graph query into a
back-door adjustment plan and estimates it with plug-in, self-normalised IPW, AIPW, or cross-fitted
DML, returning a unified :class:`~causalrl.certify.certificate.Certificate`. Non-identifiable or
unsupported queries are hedged, never silently point-estimated.

The sequential-value estimators and the streaming importance-sampling kernel moved to
:mod:`causalrl.ope`. ``stream_policy_value`` is re-exported here for its old import path, and
*lazily*: it now lives in :mod:`causalrl.ope.ipw`, which imports this package's private
:mod:`._stats`, so binding it eagerly would make ``import causalrl.ope.ipw`` re-enter a
half-initialised ``causalrl.estimate``.

``Classifier`` / ``LogisticRegressor`` / ``Regressor`` / ``RidgeRegressor`` (:mod:`.nuisance`) back
several of the estimators above but are internal: import them from ``causalrl.estimate.nuisance``
directly rather than from this package.
"""

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

from causalrl.estimate.compiler import (
    EstimandNotSupportedError,
    EstimatorPlan,
    certify_effect,
    compile_estimand,
)
from causalrl.estimate.estimators import EffectEstimate, estimate_ate

if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from causalrl.ope.ipw import stream_policy_value

__all__ = [
    "EffectEstimate",
    "EstimandNotSupportedError",
    "EstimatorPlan",
    "certify_effect",
    "compile_estimand",
    "estimate_ate",
    "stream_policy_value",
]


def __getattr__(name: str) -> object:
    if name != "stream_policy_value":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = _import_module("causalrl.ope.ipw").stream_policy_value
    globals()[name] = value  # cache so subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
