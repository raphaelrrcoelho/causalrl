"""Smoke test: the d3rlpy scale example runs to completion.

It self-guards on d3rlpy, so with the optional dependency absent it prints a skip line and exits 0;
with it present it runs the full train -> certify flow. Either way, exit 0.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "scale_d3rlpy_certify.py"


def test_scale_example_runs():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
