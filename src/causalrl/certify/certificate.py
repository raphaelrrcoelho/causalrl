"""The unified :class:`Certificate` and its parts (plan §5.2; invariants I1-I3).

One serializable certificate type: it records the claim, the structured estimand, an epistemic
``kind`` (I2), the numeric result, the assumptions consumed, and either a ``witness`` (why the
claim holds) or a ``hedge`` (why it was refused or weakened, incl. I3 target downgrades). Shipped
bespoke certificates adapt into this type; every new inferential routine returns it.
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from causalrl.identification.bounds import Interval

_SCHEMA_VERSION = 1


def _empty_dict() -> dict[str, Any]:
    return {}


class Kind(enum.Enum):
    """Epistemic status of a certificate's claim (I2). Never conflate the three."""

    IDENTIFIED = "identified"  # point-identified under the stated graph/assumptions
    BOUNDED = "bounded"  # partial identification under an explicit sensitivity budget
    EMPIRICAL = "empirical"  # simulation/sample evidence only; no identification guarantee


@dataclass(frozen=True)
class EstimandSpec:
    """What a certificate is about.

    Named ``EstimandSpec`` to avoid colliding with the shipped identification
    :class:`causalrl.identification.id_algorithm.Estimand`. ``target`` is the functional
    (``mean``/``quantile``/``tail``); only ``mean`` is used before Phase 1.
    """

    query: str  # see | do | counterfactual | policy_value | transport | equilibrium
    target: str = "mean"
    policy: str | None = None
    domains: tuple[str, ...] = ()


@dataclass(frozen=True)
class Assumption:
    """A consumed assumption, optionally with a falsification/diagnostic result."""

    name: str  # e.g. "MSM", "backdoor", "overlap", "mi-cap", "moment-condition"
    params: dict[str, Any] = field(default_factory=_empty_dict)
    checkable: bool = False
    diagnostic: dict[str, Any] | None = None


@dataclass(frozen=True)
class Witness:
    """Why an identified/bounded claim holds: adjustment set, transport formula, ID trace, ..."""

    kind: str
    detail: dict[str, Any] = field(default_factory=_empty_dict)


@dataclass(frozen=True)
class Hedge:
    """Why a claim was refused or weakened (subsumes the shipped act/abstain verdict; I3)."""

    reason: str
    detail: dict[str, Any] | None = None
    downgraded_from: str | None = None  # original target when downgraded (e.g. "mean" -> quantile)


@dataclass(frozen=True)
class Provenance:
    """Reproducibility record (I8): library version, seeds, data/graph fingerprints, timestamp."""

    library_version: str
    seeds: tuple[int, ...] = ()
    data_fingerprint: str | None = None
    graph_hash: str | None = None
    timestamp: str = ""  # ISO-8601 UTC

    @staticmethod
    def create(
        *,
        seeds: tuple[int, ...] = (),
        data_fingerprint: str | None = None,
        graph_hash: str | None = None,
    ) -> Provenance:
        """Fill ``library_version`` and a UTC ``timestamp`` automatically."""
        return Provenance(
            library_version=_library_version(),
            seeds=tuple(seeds),
            data_fingerprint=data_fingerprint,
            graph_hash=graph_hash,
            timestamp=_dt.datetime.now(_dt.UTC).isoformat(),
        )


def _library_version() -> str:
    try:
        return _pkg_version("causalrl")
    except PackageNotFoundError:  # pragma: no cover - only when run from an uninstalled tree
        return "unknown"


def _value_to_json(v: float | Interval | None) -> dict[str, Any]:
    if v is None:
        return {"t": "none"}
    if isinstance(v, Interval):
        return {"t": "interval", "lower": v.lower, "upper": v.upper}
    return {"t": "float", "v": float(v)}


def _value_from_json(d: Mapping[str, Any]) -> float | Interval | None:
    t = d["t"]
    if t == "none":
        return None
    if t == "interval":
        return Interval(float(d["lower"]), float(d["upper"]))
    return float(d["v"])


def _format_value(v: float | Interval | None) -> str | None:
    if v is None:
        return None
    if isinstance(v, Interval):
        return f"[{v.lower:.4g}, {v.upper:.4g}]"
    return f"{v:.4g}"


@dataclass(frozen=True)
class Certificate:
    """A serializable certified claim (plan §5.2)."""

    claim: str
    estimand: EstimandSpec
    kind: Kind
    value: float | Interval | None
    alpha: float | None
    assumptions: tuple[Assumption, ...]
    method: str
    witness: Witness | None
    hedge: Hedge | None
    provenance: Provenance
    ci: Interval | None = None  # optional confidence interval for `value` at level `alpha`

    def __str__(self) -> str:
        parts = [f"[{self.kind.name}] {self.claim}"]
        v = _format_value(self.value)
        if v is not None:
            parts.append(f"value={v}")
        if self.ci is not None:
            parts.append(f"ci={_format_value(self.ci)}")
        if self.hedge is not None:
            parts.append(f"HEDGE: {self.hedge.reason}")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "claim": self.claim,
            "estimand": {
                "query": self.estimand.query,
                "target": self.estimand.target,
                "policy": self.estimand.policy,
                "domains": list(self.estimand.domains),
            },
            "kind": self.kind.value,
            "value": _value_to_json(self.value),
            "ci": _value_to_json(self.ci),
            "alpha": self.alpha,
            "assumptions": [
                {
                    "name": a.name,
                    "params": a.params,
                    "checkable": a.checkable,
                    "diagnostic": a.diagnostic,
                }
                for a in self.assumptions
            ],
            "method": self.method,
            "witness": (
                None
                if self.witness is None
                else {"kind": self.witness.kind, "detail": self.witness.detail}
            ),
            "hedge": (
                None
                if self.hedge is None
                else {
                    "reason": self.hedge.reason,
                    "detail": self.hedge.detail,
                    "downgraded_from": self.hedge.downgraded_from,
                }
            ),
            "provenance": {
                "library_version": self.provenance.library_version,
                "seeds": list(self.provenance.seeds),
                "data_fingerprint": self.provenance.data_fingerprint,
                "graph_hash": self.provenance.graph_hash,
                "timestamp": self.provenance.timestamp,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Certificate:
        est = d["estimand"]
        prov = d["provenance"]
        wit = d.get("witness")
        hed = d.get("hedge")
        ci_raw = d.get("ci")
        ci: Interval | None = None
        if ci_raw is not None:
            parsed = _value_from_json(ci_raw)
            ci = parsed if isinstance(parsed, Interval) else None
        return cls(
            claim=d["claim"],
            estimand=EstimandSpec(
                query=est["query"],
                target=est.get("target", "mean"),
                policy=est.get("policy"),
                domains=tuple(est.get("domains", ())),
            ),
            kind=Kind(d["kind"]),
            value=_value_from_json(d["value"]),
            alpha=d["alpha"],
            assumptions=tuple(
                Assumption(
                    name=a["name"],
                    params=dict(a.get("params", {})),
                    checkable=a.get("checkable", False),
                    diagnostic=a.get("diagnostic"),
                )
                for a in d["assumptions"]
            ),
            method=d["method"],
            witness=None
            if wit is None
            else Witness(kind=wit["kind"], detail=dict(wit.get("detail", {}))),
            hedge=(
                None
                if hed is None
                else Hedge(
                    reason=hed["reason"],
                    detail=hed.get("detail"),
                    downgraded_from=hed.get("downgraded_from"),
                )
            ),
            provenance=Provenance(
                library_version=prov["library_version"],
                seeds=tuple(prov.get("seeds", ())),
                data_fingerprint=prov.get("data_fingerprint"),
                graph_hash=prov.get("graph_hash"),
                timestamp=prov.get("timestamp", ""),
            ),
            ci=ci,
        )

    @staticmethod
    def from_json(s: str) -> Certificate:
        return Certificate.from_dict(json.loads(s))
