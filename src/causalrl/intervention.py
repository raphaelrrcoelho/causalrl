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

import numpy as np

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
class Discrete:
    """A finite set of admissible values -- the only domain kind before continuous ones existed."""

    values: tuple[Any, ...]

    def contains(self, value: Any) -> bool:
        return value in self.values

    def project(self, value: Any) -> Any:
        """The admissible value closest to ``value``; ``value`` itself when already admissible.

        Closeness is numeric where the values are numeric and identity otherwise, so a
        non-admissible categorical projects to the first admissible value rather than raising --
        :meth:`InterventionSpace.project` is the "make this admissible" path, and a caller who
        wants a refusal should ask :meth:`InterventionSpace.permits` instead.
        """
        if self.contains(value):
            return value
        try:
            return min(self.values, key=lambda v: abs(float(v) - float(value)))
        except (TypeError, ValueError):
            return self.values[0]

    def sample(self, rng: np.random.Generator) -> Any:
        return self.values[int(rng.integers(len(self.values)))]


@dataclass(frozen=True)
class Continuous:
    """A closed real interval ``[low, high]`` of admissible values.

    The action-side counterpart of :mod:`causalrl.state`'s feature vectors. The estimation core
    never needed a discretisation -- cross-fitted DML, additive-noise mechanisms, the continuous
    bounds and the RBF-encoded function approximators all take real-valued inputs -- so a
    continuous *confounder* was always expressible while a continuous *treatment* was not. Dose,
    price, budget and duration are the ordinary cases.

    Deliberately NOT named ``Interval``: :class:`causalrl.Interval` is a bound on an estimand's
    value, and reusing the name for an action domain would make every docstring mentioning one
    ambiguous about which it meant.
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        low, high = float(self.low), float(self.high)
        if not low <= high:
            raise ValueError(
                f"Continuous(low={self.low}, high={self.high}) must satisfy low <= high: an "
                "inverted interval admits nothing, which makes every intervention on this "
                "variable inadmissible while the variable still counts as manipulable."
            )
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    def contains(self, value: Any) -> bool:
        try:
            return self.low <= float(value) <= self.high
        except (TypeError, ValueError):
            return False

    def project(self, value: Any) -> float:
        """``value`` clipped into ``[low, high]``."""
        return min(self.high, max(self.low, float(value)))

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


InterventionDomain = Discrete | Continuous
"""What a manipulable variable may be set to: a finite set, or a real interval.

Named in full because :class:`causalrl.Domain` is already taken by the *transport* domain -- a
source or target population in a selection diagram. The two are unrelated, and one of them had to
say which it was.
"""


def _as_domain(spec: InterventionDomain | Iterable[Any]) -> InterventionDomain:
    """Normalise a domain specification, so a raw value tuple still means :class:`Discrete`."""
    if isinstance(spec, Discrete | Continuous):
        return spec
    values = tuple(spec)
    if not values:
        raise ValueError(
            "a variable was given an empty domain: a variable that may be intervened on but has "
            "no value it can take makes every intervention set containing it unsatisfiable while "
            "still counting as admissible. Omit the variable to say it is not manipulable here."
        )
    return Discrete(values)


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

    domains: tuple[tuple[str, InterventionDomain], ...] = ()

    def __post_init__(self) -> None:
        # Raw value tuples are normalised to Discrete, so direct construction from the pre-3.1
        # ``(name, values)`` shape keeps working unchanged.
        normalised = tuple(
            sorted(((n, _as_domain(d)) for n, d in self.domains), key=lambda kv: kv[0])
        )
        object.__setattr__(self, "domains", normalised)

    @classmethod
    def create(cls, domains: Mapping[str, Iterable[Any]]) -> InterventionSpace:
        """Build a space from ``{variable: allowed values}``.

        A variable mapped to an empty domain is refused rather than silently retained: it would be
        a variable that is nominally manipulable but has nothing it can be set to, which makes
        every intervention set containing it unsatisfiable while still counting as admissible.
        Drop the variable instead -- that is what "not manipulable here" means.
        """
        pairs: list[tuple[str, InterventionDomain]] = []
        for name in sorted(domains):
            try:
                pairs.append((name, _as_domain(domains[name])))
            except ValueError as exc:
                raise ValueError(f"variable {name!r}: {exc}") from exc
        return cls(tuple(pairs))

    @property
    def variables(self) -> frozenset[str]:
        """The manipulable variables — what :func:`causalrl.pomis` takes as ``manipulable``."""
        return frozenset(name for name, _ in self.domains)

    def domain(self, variable: str) -> InterventionDomain:
        """The :data:`InterventionDomain` of ``variable``.

        Raises ``KeyError`` if it is not manipulable here.
        """
        for name, domain in self.domains:
            if name == variable:
                return domain
        raise KeyError(
            f"{variable!r} is not manipulable in this InterventionSpace "
            f"(manipulable: {sorted(self.variables)})"
        )

    def values(self, variable: str) -> tuple[Any, ...]:
        """The admissible values of a DISCRETE ``variable``.

        Raises ``KeyError`` if ``variable`` is not manipulable, and ``TypeError`` if its domain is
        continuous -- an interval has no value list, and returning a sampled or gridded stand-in
        would quietly turn an exact domain into an approximation of one.
        """
        domain = self.domain(variable)
        if isinstance(domain, Continuous):
            raise TypeError(
                f"{variable!r} has the continuous domain [{domain.low}, {domain.high}], which has "
                "no enumerable value list. Use permits/project/sample to work with it, or pass a "
                "Discrete domain if you want arms."
            )
        return domain.values

    def permits(self, intervention: Intervention) -> bool:
        """Whether every variable in ``intervention`` is manipulable and set to an allowed value."""
        lookup = dict(self.domains)
        return all(
            name in lookup and lookup[name].contains(value) for name, value in intervention.items()
        )

    def project(self, intervention: Intervention) -> Intervention:
        """``intervention`` moved into this space: each value clipped or snapped to its domain.

        The counterpart of :meth:`permits` for a search that proposes freely and needs its
        proposals made admissible -- clipping a dose to its safe range rather than discarding the
        candidate. Raises ``KeyError`` naming any variable that is not manipulable here, since no
        amount of clipping makes an unavailable lever available.
        """
        return {name: self.domain(name).project(value) for name, value in intervention.items()}

    def sample(self, variables: Iterable[str], rng: np.random.Generator) -> Intervention:
        """One admissible assignment of ``variables``, drawn uniformly from each domain.

        The continuous counterpart of :meth:`assignments`: where a finite space can be enumerated,
        an interval must be searched, and a search needs proposals.
        """
        return {name: self.domain(name).sample(rng) for name in sorted(variables)}

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
        pairs: list[tuple[str, InterventionDomain]] = []
        for name, mine in self.domains:
            if name not in theirs:
                continue
            merged = _intersect(mine, theirs[name])
            if merged is not None:
                pairs.append((name, merged))
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
        # KeyError names a non-manipulable variable; TypeError names a continuous one, which has
        # no arms to enumerate -- the case that needs a search rather than an arm list.
        domains = [self.values(name) for name in names]
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


def _intersect(a: InterventionDomain, b: InterventionDomain) -> InterventionDomain | None:
    """The domain admitting exactly what both admit, or ``None`` when nothing is left.

    ``None`` rather than an empty domain because :meth:`InterventionSpace.__and__` drops such a
    variable entirely -- retaining it with nothing it can be set to is what :meth:`create` refuses.
    """
    match a, b:
        case Discrete(), _:
            kept = tuple(v for v in a.values if b.contains(v))
            return Discrete(kept) if kept else None
        case Continuous(), Discrete():
            kept = tuple(v for v in b.values if a.contains(v))
            return Discrete(kept) if kept else None
        case Continuous(), Continuous():
            low, high = max(a.low, b.low), min(a.high, b.high)
            return Continuous(low, high) if low <= high else None
    return None
