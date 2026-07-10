"""Regime: a labeled data-generating configuration (plan §5.3).

Named selection-marked variables (mechanisms that differ from the reference) plus parameters.
Built on the transport :class:`~causalrl.identification.id_algorithm.Domain`: ``selection``
projects directly to a Domain's selection set. Hashable, serializable, and composable (``a | b``
merges the two, detecting conflicts on shared parameters). Parameter values should be JSON scalars
(int/float/str/bool) so certificates and logs can serialize them.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from causalrl.identification.id_algorithm import Domain


@dataclass(frozen=True)
class Regime:
    """A labeled data-generating configuration (§5.3)."""

    name: str
    selection: frozenset[str] = frozenset()
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        # normalize parameter order so equality/hashing ignore insertion order
        ordered = tuple(sorted(self.parameters, key=lambda kv: kv[0]))
        object.__setattr__(self, "parameters", ordered)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        selection: Iterable[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
    ) -> Regime:
        params = parameters or {}
        pairs = tuple((k, params[k]) for k in sorted(params))
        return cls(name, frozenset(selection or ()), pairs)

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.parameters)

    def __or__(self, other: Regime) -> Regime:
        """Merge two regimes; raise on a shared parameter set to conflicting values."""
        a, b = self.params, other.params
        for key in a.keys() & b.keys():
            if a[key] != b[key]:
                raise ValueError(f"regime conflict on parameter {key!r}: {a[key]!r} vs {b[key]!r}")
        return Regime.create(
            f"{self.name}|{other.name}",
            selection=self.selection | other.selection,
            parameters={**a, **b},
        )

    def to_domain(self, *, experiments: frozenset[frozenset[str]] = frozenset()) -> Domain:
        """Project to a transport :class:`Domain` (selection-marked variables carry over)."""
        return Domain(self.name, self.selection, experiments)

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "selection": sorted(self.selection),
                "parameters": [list(kv) for kv in self.parameters],
            }
        )

    @staticmethod
    def from_json(s: str) -> Regime:
        d: dict[str, Any] = json.loads(s)
        raw_params: Any = d.get("parameters", [])
        params: dict[str, Any] = {str(k): v for k, v in raw_params}
        raw_selection: Any = d.get("selection", [])
        return Regime.create(
            d["name"], selection={str(x) for x in raw_selection}, parameters=params
        )
