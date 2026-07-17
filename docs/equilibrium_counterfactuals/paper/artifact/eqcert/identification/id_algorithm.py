"""Minimal stub of the library's identification engine: only ``Domain`` is needed here."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Domain:
    """A source domain relative to the target: which mechanisms differ, and what data it offers."""

    name: str
    selection: frozenset[str] = frozenset()
    experiments: frozenset[frozenset[str]] = frozenset()
