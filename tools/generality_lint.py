#!/usr/bin/env python3
"""Generality lint (plan §12.4; enforces invariant I7 — domain-agnostic core).

Fail CI when an application-domain noun leaks into the *public* surface of ``src/causalrl/`` —
i.e. a public identifier name, a function/class signature, or a docstring. Domain-specific
demonstrations belong in ``examples/`` or a downstream package, never in the type-agnostic core.

Matching rules (kept deliberately narrow so genuine technical terms are not flagged):

* **Identifiers** (public ``def``/``class`` names and function argument names) are split on ``_``
  and camelCase; each whole sub-token is compared against the denylist. So ``market_price`` and
  ``PatientState`` are caught, but ``bankroll`` (no ``bank`` sub-token) is not.
* **Docstrings** (module / class / function) are matched on whole words with ``\\b`` boundaries,
  case-insensitively.
* Comments and private (leading-underscore) names are not the public surface and are not scanned.

The denylist is a configurable seed of common application nouns; extend :data:`DENYLIST` as new
domains are quarantined.

Usage::

    python tools/generality_lint.py [PATH ...]      # defaults to src/causalrl

Exit code ``0`` when clean, ``1`` when any violation is found.
"""

from __future__ import annotations

import ast
import re
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

# Seed denylist (plan §12.4). Configurable: add nouns as new application domains are quarantined.
DENYLIST: frozenset[str] = frozenset(
    {"market", "price", "trader", "firm", "household", "patient", "portfolio", "bank"}
)

# Split an identifier into camelCase / snake_case sub-tokens.
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


@dataclass(frozen=True)
class Violation:
    """A single domain-noun leak: which file/line, which word, and where it was found."""

    path: Path
    lineno: int
    word: str
    context: str  # "identifier" | "docstring"
    detail: str

    def render(self, root: Path) -> str:
        rel = self.path.relative_to(root) if self.path.is_relative_to(root) else self.path
        return f"{rel}:{self.lineno}: domain noun '{self.word}' in {self.context} -> {self.detail}"


def _identifier_tokens(name: str) -> set[str]:
    """Lowercased sub-tokens of ``name``, split on ``_`` and camelCase boundaries."""
    return {m.group(0).lower() for part in name.split("_") for m in _CAMEL.finditer(part)}


def _prose_pattern(denylist: Iterable[str]) -> re.Pattern[str]:
    alternation = "|".join(re.escape(w) for w in sorted(denylist))
    return re.compile(rf"\b({alternation})\b", re.IGNORECASE)


def _docstring_constant(node: ast.AST) -> ast.Constant | None:
    """The string-literal docstring node of a module/class/function, if any (for its ``lineno``)."""
    body = getattr(node, "body", None)
    if not body:
        return None
    first = body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value
    return None


def _iter_identifiers(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(lineno, name)`` for every public def/class name and every function argument name."""
    for node in ast.walk(tree):
        if isinstance(
            node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ) and not node.name.startswith("_"):
            yield node.lineno, node.name
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg):
                if arg is not None and not arg.arg.startswith("_"):
                    yield arg.lineno, arg.arg


def _iter_docstrings(tree: ast.AST) -> Iterator[tuple[int, str]]:
    """``(lineno, text)`` for the module docstring and every class/function docstring."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = _docstring_constant(node)
            if doc is not None and isinstance(doc.value, str):
                yield doc.lineno, doc.value


def scan_file(path: Path, denylist: frozenset[str] = DENYLIST) -> list[Violation]:
    """Every domain-noun leak in ``path`` (identifier sub-tokens; whole-word docstring prose)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    prose = _prose_pattern(denylist)
    out: list[Violation] = []
    for lineno, name in _iter_identifiers(tree):
        for token in sorted(_identifier_tokens(name) & denylist):
            out.append(Violation(path, lineno, token, "identifier", name))
    for start_line, text in _iter_docstrings(tree):
        lines = text.splitlines()
        for match in prose.finditer(text):
            offset = text[: match.start()].count("\n")
            snippet = lines[offset].strip() if offset < len(lines) else text.strip()
            word = match.group(1).lower()
            out.append(Violation(path, start_line + offset, word, "docstring", snippet[:70]))
    return out


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(f for f in p.rglob("*.py") if "__pycache__" not in f.parts)
        elif p.suffix == ".py":
            yield p


def scan_paths(paths: Iterable[Path], denylist: frozenset[str] = DENYLIST) -> list[Violation]:
    out: list[Violation] = []
    for f in iter_python_files(paths):
        out.extend(scan_file(f, denylist))
    return out


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent.parent
    targets = [Path(a) for a in args] or [root / "src" / "causalrl"]
    violations = scan_paths(targets)
    if violations:
        print(f"generality lint: {len(violations)} violation(s) (I7 -- domain leakage):\n")
        for v in violations:
            print("  " + v.render(root))
        print(
            "\nRelocate domain-specific names/prose to examples/ or a downstream package, "
            "or extend DENYLIST if a flagged word is a genuine technical term."
        )
        return 1
    print("generality lint: clean -- no domain-noun leakage in the public surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
