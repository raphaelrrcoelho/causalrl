"""Causal RL agents.

Re-exports are resolved lazily through module ``__getattr__``. The top-level package's own lazy
loader reaches an agent by importing its submodule -- ``causalrl.agents.dovi`` for ``DOVI``, say --
and that import executes *this* module first, so a name bound here eagerly would drag its
dependencies (for the torch-backed agents, PyTorch) into every torch-free import of the graph
surface. ``tests/test_public_api.py`` pins that: ``from causalrl import DOVI`` must succeed with
torch absent.

``__all__`` here is not the full agent roster. ``causalrl`` itself is the curated public surface and
``from causalrl import OnlineCausalMBRL`` is the canonical import; every agent is additionally
reachable by its module path. What this module adds is the second spelling a caller reasonably
reaches for, ``from causalrl.agents import X``, for the names listed below. Adding one means adding
it to ``_LAZY`` *and* ``__all__`` together: a name in only one of the two is either an export that
cannot resolve or one ``dir()`` hides.
"""

from __future__ import annotations

from importlib import import_module as _import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers / IDEs only; not executed at runtime
    from causalrl.agents.online_causal_mbrl import OnlineCausalMBRL

# name -> (submodule, attribute); resolved on first attribute access.
_LAZY: dict[str, tuple[str, str]] = {
    "OnlineCausalMBRL": ("causalrl.agents.online_causal_mbrl", "OnlineCausalMBRL"),
}

__all__ = [
    "OnlineCausalMBRL",
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
