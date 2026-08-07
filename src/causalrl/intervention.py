"""Set-valued interventions: the action type the identification layer already speaks.

``StructuralCausalModel.do`` takes a ``Mapping[str, value]``; :func:`causalrl.pomis` returns
*sets* of variables; :class:`causalrl.Regime` carries a set of selection-marked names. The one
place that could not express a multi-variable intervention was the agent layer, whose
:class:`causalrl.Agent` selects an arm index. This module supplies the missing vocabulary so the
two ends meet:

* :data:`Intervention` — an assignment ``{variable: value}``, exactly what ``do`` accepts.
* :class:`InterventionSpace` — which variables may be set *in a given context*, and to which
  values. Feasibility is rarely a property of the graph alone (a lever can be unavailable in one
  state and available in the next), so the space is a per-decision object rather than a static
  attribute of the model.

The bridge between them is :meth:`InterventionSpace.assignments`: an intervention *set* from
POMIS names the variables worth intervening on, and the assignments of those variables within the
space are the arms an agent actually chooses between.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

Intervention = Mapping[str, Any]
"""An assignment ``{variable: value}`` — the argument ``StructuralCausalModel.do`` accepts.

The empty mapping is the observational regime (``do()`` on nothing), matching the ``frozenset()``
that :func:`causalrl.pomis` emits when not intervening is possibly optimal.
"""

_MAX_ASSIGNMENTS = 100_000
"""Largest number of arms :meth:`InterventionSpace.assignments` will enumerate.

The count is the PRODUCT of the chosen variables' domain sizes, so it grows explosively: six
variables with ten values each is already a million arms. Enumerating them is essentially never
what a caller wants -- at that size the decision problem needs structure (a factored value model,
a search) rather than an arm list -- so this refuses instead of materialising the product.
"""


def canonical(intervention: Intervention) -> tuple[tuple[str, Any], ...]:
    """A hashable, order-independent key for an intervention.

    ``Intervention`` is a ``Mapping`` and therefore unhashable, which makes the obvious things --
    deduplicating arms, using an intervention as a dict key, comparing two of them for identity --
    awkward at every call site. This is the one canonical form, sorted by variable name so
    ``{"A": 1, "B": 0}`` and ``{"B": 0, "A": 1}`` produce the same key.
    """
    return tuple(sorted(intervention.items()))


@dataclass(frozen=True)
class InterventionSpace:
    """Which variables may be intervened on right now, and the values each may take.

    Hashable and composable, following :class:`causalrl.Regime`'s idiom: the domains are stored as
    a normalised tuple of pairs so that two spaces built from equal mappings compare equal
    regardless of insertion order. Build one with :meth:`create`.

    A space is a *context-dependent* object. The intended pattern is to construct a fresh one per
    decision from the current state -- dropping the levers this state does not permit and
    narrowing the values the others may take -- and hand it to the agent for that decision. What
    stays fixed across decisions is the graph; what varies is this.
    """

    domains: tuple[tuple[str, tuple[Any, ...]], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "domains", tuple(sorted(self.domains, key=lambda kv: kv[0])))

    @classmethod
    def create(cls, domains: Mapping[str, Iterable[Any]]) -> InterventionSpace:
        """Build a space from ``{variable: allowed values}``.

        A variable mapped to an empty domain is refused rather than silently retained: it would be
        a variable that is nominally manipulable but has nothing it can be set to, which makes
        every intervention set containing it unsatisfiable while still counting as admissible.
        Drop the variable instead -- that is what "not manipulable here" means.
        """
        pairs: list[tuple[str, tuple[Any, ...]]] = []
        for name in sorted(domains):
            values = tuple(domains[name])
            if not values:
                raise ValueError(
                    f"variable {name!r} was given an empty domain: a variable that may be "
                    "intervened on but has no value it can take makes every intervention set "
                    "containing it unsatisfiable while still counting as admissible. Omit the "
                    "variable to say it is not manipulable in this context."
                )
            pairs.append((name, values))
        return cls(tuple(pairs))

    @property
    def variables(self) -> frozenset[str]:
        """The manipulable variables — what :func:`causalrl.pomis` takes as ``manipulable``."""
        return frozenset(name for name, _ in self.domains)

    def values(self, variable: str) -> tuple[Any, ...]:
        """The admissible values of ``variable``; raises ``KeyError`` if it is not manipulable."""
        for name, values in self.domains:
            if name == variable:
                return values
        raise KeyError(
            f"{variable!r} is not manipulable in this InterventionSpace "
            f"(manipulable: {sorted(self.variables)})"
        )

    def permits(self, intervention: Intervention) -> bool:
        """Whether every variable in ``intervention`` is manipulable and set to an allowed value."""
        lookup = dict(self.domains)
        return all(name in lookup and value in lookup[name] for name, value in intervention.items())

    def restrict(self, variables: Iterable[str]) -> InterventionSpace:
        """The sub-space over ``variables``, ignoring any that are not manipulable here."""
        keep = set(variables)
        return InterventionSpace(tuple(kv for kv in self.domains if kv[0] in keep))

    def __and__(self, other: InterventionSpace) -> InterventionSpace:
        """Intersect two spaces: variables in both, each with the values both allow.

        A variable whose domains do not overlap drops out entirely, since retaining it with an
        empty domain is exactly what :meth:`create` refuses.
        """
        theirs = dict(other.domains)
        pairs: list[tuple[str, tuple[Any, ...]]] = []
        for name, values in self.domains:
            if name not in theirs:
                continue
            allowed = tuple(v for v in values if v in theirs[name])
            if allowed:
                pairs.append((name, allowed))
        return InterventionSpace(tuple(pairs))

    def assignments(self, variables: Iterable[str]) -> Iterator[Intervention]:
        """Every assignment of ``variables`` admissible here — the arms of an intervention set.

        This is the step from "which variables to intervene on" (what POMIS answers) to "which
        interventions to choose between" (what an agent needs). The empty set yields exactly one
        assignment, the empty intervention, which is the observational regime rather than nothing.

        Raises ``KeyError`` naming any variable that is not manipulable here, and ``ValueError``
        when the product exceeds :data:`_MAX_ASSIGNMENTS`.
        """
        names = sorted(variables)
        domains = [self.values(name) for name in names]  # KeyError names the offending variable
        total = 1
        for values in domains:
            total *= len(values)
        if total > _MAX_ASSIGNMENTS:
            sizes = ", ".join(f"{n}={len(v)}" for n, v in zip(names, domains, strict=True))
            raise ValueError(
                f"enumerating assignments of {names} would produce {total} arms -- the product of "
                f"their domain sizes ({sizes}) -- above the _MAX_ASSIGNMENTS={_MAX_ASSIGNMENTS} "
                "limit. A decision over that many arms needs structure (a factored value model or "
                "a search over assignments), not an enumerated arm list."
            )
        for combination in product(*domains):
            yield dict(zip(names, combination, strict=True))
