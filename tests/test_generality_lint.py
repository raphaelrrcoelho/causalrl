"""The generality lint (plan §12.4 / invariant I7) keeps the public core domain-agnostic.

The load-by-path shim exists because ``tools/`` is a scripts directory, not an importable package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_TOOL = _ROOT / "tools" / "generality_lint.py"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("generality_lint", _TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass annotation resolution (PEP 563 strings) looks the module up
    # in sys.modules by __module__, which would otherwise be absent for a load-by-path shim.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gl: Any = _load()


def test_shipped_public_surface_is_clean() -> None:
    """The gate: zero application-domain nouns in ``src/causalrl`` identifiers/docstrings."""
    violations = gl.scan_paths([_ROOT / "src" / "causalrl"])
    assert violations == [], "\n".join(v.render(_ROOT) for v in violations)


def test_identifier_subtokens_are_split(tmp_path: Path) -> None:
    leak = tmp_path / "leak.py"
    leak.write_text("def market_price(portfolio):\n    return portfolio\n")
    assert {v.word for v in gl.scan_file(leak)} == {"market", "price", "portfolio"}


def test_camelcase_identifier_is_split(tmp_path: Path) -> None:
    leak = tmp_path / "leak.py"
    leak.write_text("class PatientState:\n    pass\n")
    hits = gl.scan_file(leak)
    assert [v.word for v in hits] == ["patient"]
    assert hits[0].context == "identifier"


def test_docstring_prose_is_whole_word(tmp_path: Path) -> None:
    leak = tmp_path / "leak.py"
    leak.write_text('"""A patient outcome under the firm\'s policy."""\n')
    assert {v.word for v in gl.scan_file(leak)} == {"patient", "firm"}
    assert all(v.context == "docstring" for v in gl.scan_file(leak))


def test_no_false_positive_on_substrings(tmp_path: Path) -> None:
    # 'bankroll' contains 'bank' and 'priced' contains 'price', but neither is a whole token/word.
    clean = tmp_path / "clean.py"
    clean.write_text('"""The bankroll is priced fairly."""\ndef bankroll_priced():\n    return 1\n')
    assert gl.scan_file(clean) == []


def test_private_names_are_not_public_surface(tmp_path: Path) -> None:
    priv = tmp_path / "priv.py"
    priv.write_text("def _market_helper():\n    return 1\n")
    assert gl.scan_file(priv) == []


def test_reports_precise_line(tmp_path: Path) -> None:
    leak = tmp_path / "leak.py"
    leak.write_text('"""Line one.\n\nA patient is on line three.\n"""\n')
    (hit,) = gl.scan_file(leak)
    assert hit.word == "patient"
    assert hit.lineno == 3
