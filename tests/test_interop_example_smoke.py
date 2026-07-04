"""Smoke test: the interop example scripts run to completion.

They self-guard on the optional dependency, so with DoWhy / EconML absent they print a skip line and
exit 0; with the deps present they run the full estimate -> certificate flow. Either way, exit 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.mark.parametrize("script", ["interop_dowhy_certify.py", "interop_econml_certify.py"])
def test_interop_example_runs(script):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
