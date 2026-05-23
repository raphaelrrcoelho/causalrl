"""Typed exceptions. Causal misuse fails loudly, never silently."""

from __future__ import annotations


class CausalRLError(Exception):
    """Base class for all causalrl errors."""


class CausalGraphError(CausalRLError):
    """Invalid graph operation (unknown node, cycle, malformed edge)."""


class NotIdentifiableError(CausalRLError):
    """A causal query is not identifiable from the available data."""

    def __init__(self, message: str, witness: object | None = None) -> None:
        super().__init__(message)
        self.witness = witness


class RealizabilityError(CausalRLError):
    """A counterfactual query cannot be realized from the given evidence."""
