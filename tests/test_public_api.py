"""The curated public API (`causalrl.__all__`) stays consistent and importable."""

import inspect
import subprocess
import sys
import tomllib
from pathlib import Path

import causalrl


def test_every_exported_name_is_importable():
    for name in causalrl.__all__:
        assert hasattr(causalrl, name), f"{name!r} is in __all__ but not importable from causalrl"


def test_all_has_no_duplicates():
    assert len(causalrl.__all__) == len(set(causalrl.__all__))


def test_no_undeclared_public_symbols():
    # Force the intentionally lazy stable exports to materialize before checking for leaks.
    for name in causalrl.__all__:
        assert hasattr(causalrl, name)
    public = {
        name
        for name in vars(causalrl)
        if not name.startswith("_") and not inspect.ismodule(getattr(causalrl, name))
    }
    assert public == set(causalrl.__all__) - {"__version__"}


def test_key_symbols_are_exported():
    # Spot-check one representative name per layer so a missing re-export is caught loudly.
    for name in (
        "StructuralCausalModel",
        "DOVI",
        "UCDTR",
        "DTREnv",
        "generate_logs",
        "causal_q_bounds",
        "NotIdentifiableError",
        "UnverifiedAssumptionError",
        "report_to_dict",
    ):
        assert name in causalrl.__all__


def test_front_door_symbols_exported():
    # The v1.1 decision front door + the v1.2 scale path.
    for name in ("certify_estimate", "PolicyValueContrast", "certify_policy"):
        assert name in causalrl.__all__
        assert hasattr(causalrl, name)


def test_version_is_stamped():
    pyproject = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    assert causalrl.__version__ == pyproject["project"]["version"]


def test_graph_and_tabular_exports_import_without_torch():
    source = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_torch
from causalrl import CausalGraph, DOVI, DTREnv, pomis

graph = CausalGraph(directed_edges=[("X", "Y")])
assert pomis(graph, "Y") == [frozenset({"X"})]
assert DOVI is not None
assert DTREnv is not None
"""
    result = subprocess.run(
        [sys.executable, "-c", source], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_m0_symbols_exported():
    for name in ("ConfoundedContextualBandit", "CertifiedPolicyAgent", "run_m0_kill_gate"):
        assert name in causalrl.__all__
        assert getattr(causalrl, name) is not None
