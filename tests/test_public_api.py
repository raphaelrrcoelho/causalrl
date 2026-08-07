"""The curated public API (`causalrl.__all__`) stays consistent and importable."""

import inspect
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

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
    """The version is derived rather than hardcoded, and every file quoting it agrees.

    Deliberately *not* ``causalrl.__version__ == pyproject version``. ``__version__`` reads the
    installed distribution's metadata, so that comparison tests how recently the environment was
    installed, not anything about this source tree: it fails on any editable checkout where
    ``pyproject.toml`` was bumped after the last install, and it passes automatically in CI, where
    the install is always fresh. What is worth pinning is that the version cannot drift from what
    was packaged -- which derivation guarantees -- and that the files quoting it to users agree.
    """
    root = Path(__file__).parent.parent
    declared = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]

    init = (root / "src" / "causalrl" / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r"^__version__ = (?![\"'])", init, re.M), (
        "__version__ must be derived from installed metadata, not written as a string literal"
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", causalrl.__version__), (
        f"__version__ did not resolve to a release version: {causalrl.__version__!r}"
    )

    newest = re.search(
        r"^## \[(\d+\.\d+\.\d+)\]", (root / "CHANGELOG.md").read_text(encoding="utf-8"), re.M
    )
    assert newest is not None, "CHANGELOG.md has no released section"
    assert newest.group(1) == declared, (
        f"pyproject declares {declared}, but the newest CHANGELOG release is {newest.group(1)}"
    )

    cited = re.search(
        r'^version: "([^"]+)"', (root / "CITATION.cff").read_text(encoding="utf-8"), re.M
    )
    assert cited is not None, "CITATION.cff has no version field"
    assert cited.group(1) == declared, (
        f"pyproject declares {declared}, but CITATION.cff cites {cited.group(1)}"
    )


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


def test_m2_symbols_exported():
    for name in (
        "TransportableConfoundedBandit",
        "TransportBackdoorAgent",
        "run_m2_phase_diagram",
    ):
        assert name in causalrl.__all__
        assert getattr(causalrl, name) is not None


def test_m3_symbols_exported():
    for name in (
        "ContinuousConfoundedBandit",
        "FunctionApproxBackdoorAgent",
        "run_m3_function_approx_gate",
    ):
        assert name in causalrl.__all__
        assert getattr(causalrl, name) is not None


def test_agent_frontdoor_symbols_exported():
    for name in ("CausalMBRLAgent", "GFormulaBackdoorAgent"):
        assert name in causalrl.__all__
        assert getattr(causalrl, name) is not None


def test_online_causal_mbrl_is_exported_top_level_and_from_the_agents_package():
    """Both re-export sites, and the same object at each.

    A class that is only reachable by its full module path
    (``causalrl.agents.online_causal_mbrl.OnlineCausalMBRL``) is not part of the public API, and a
    package re-export that resolves to a *different* object is worse than none: callers would then
    fail ``isinstance`` checks across the two import spellings.
    """
    import causalrl.agents

    assert "OnlineCausalMBRL" in causalrl.__all__
    assert "OnlineCausalMBRL" in causalrl.agents.__all__
    assert "OnlineCausalMBRL" in dir(causalrl.agents)
    from causalrl.agents.online_causal_mbrl import OnlineCausalMBRL

    assert causalrl.OnlineCausalMBRL is OnlineCausalMBRL
    assert causalrl.agents.OnlineCausalMBRL is OnlineCausalMBRL


def test_agents_package_rejects_an_unknown_name():
    """A lazy loader must still raise ``AttributeError`` for a name it does not export.

    Module ``__getattr__`` is the fallback Python calls when normal lookup fails, so anything it
    raises is what a typo'd import surfaces as, and ``hasattr`` only reports ``False`` for
    ``AttributeError``. Letting a ``KeyError`` or a ``None`` escape here would make a misspelt
    ``from causalrl.agents import ...`` fail with the wrong exception and break introspection.
    """
    import causalrl.agents

    with pytest.raises(AttributeError, match="OnlineCausalMBLR"):
        _ = causalrl.agents.OnlineCausalMBLR  # type: ignore[attr-defined]
    assert not hasattr(causalrl.agents, "OnlineCausalMBLR")


def test_agents_package_imports_without_torch():
    """``causalrl.agents`` must stay importable with PyTorch absent.

    The top-level lazy loader reaches ``DOVI`` by importing ``causalrl.agents.dovi``, which
    executes ``causalrl/agents/__init__.py`` first. Binding a torch-backed agent there eagerly
    would therefore make the torch-free graph surface require torch, so the package re-exports
    lazily and this pins it: the module imports, and its ``__all__`` is readable, before anything
    that needs torch is touched.
    """
    source = """
import builtins

original_import = builtins.__import__

def import_without_torch(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "torch" or name.startswith("torch."):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    return original_import(name, globals, locals, fromlist, level)

builtins.__import__ = import_without_torch
import causalrl.agents

assert "OnlineCausalMBRL" in causalrl.agents.__all__
"""
    result = subprocess.run(
        [sys.executable, "-c", source], check=False, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_learned_scm_surface_is_exported_top_level():
    import causalrl

    for name in (
        "orient",
        "fit_scm",
        "fit_scm_mec",
        "counterfactual_interval",
        "CounterfactualBound",
        "FitReport",
        "NodeFit",
        "TabularCPT",
        "LinearGaussianFit",
        "ANMFit",
        "NeuralFit",
        "PoissonGLMFit",
        # Lazily exported optional [numpyro] backend: the name must resolve without numpyro
        # installed (only `fit()` needs it), exactly as `abduct_nuts` does.
        "BayesianLinearFit",
    ):
        assert name in causalrl.__all__, f"{name} missing from __all__"
        assert getattr(causalrl, name) is not None


# --- Surface curation gates -------------------------------------------------------------------
#
# `__all__` is flat and large. These two tests are what keep it navigable: every export must be
# placed in exactly one tier, and every export must actually appear in the API reference. Both
# failures are silent otherwise -- mkdocstrings validates the references that exist, never the ones
# that are missing, so `mkdocs --strict` passed for a long time with half the surface undocumented.

_SURFACE_META = {"__version__", "API_TIERS"}
"""Names that describe the surface rather than belonging to it, so they carry no tier."""


def test_api_tiers_partition_the_public_surface():
    tiered = [name for names in causalrl.API_TIERS.values() for name in names]
    assert len(tiered) == len(set(tiered)), "a name appears in more than one tier"
    assert set(tiered) == set(causalrl.__all__) - _SURFACE_META


def test_core_tier_stays_small_enough_to_read():
    # The tier exists to answer "where do I start"; a core of 40 names would not answer it.
    assert len(causalrl.API_TIERS["core"]) <= 20


def test_every_export_appears_in_the_api_reference():
    import re

    root = Path(__file__).resolve().parent.parent
    reference = (root / "docs" / "api.md").read_text(encoding="utf-8")
    documented = {m.rsplit(".", 1)[-1] for m in re.findall(r"^::: ([\w\.]+)", reference, re.M)}
    # The lazy export map may rename on the way out (attr `canonical` -> export
    # `canonical_intervention`), so resolve each export to the attribute it actually points at.
    source = (root / "src" / "causalrl" / "__init__.py").read_text(encoding="utf-8")
    block = source[source.index("_EXPORTS") : source.index("API_TIERS")]
    targets = {
        name: attr
        for name, _module, attr in re.findall(
            r'"([\w]+)":\s*\(\s*\n?\s*"([\w\.]+)",\s*\n?\s*"([\w]+)"', block
        )
    }
    missing = sorted(
        name
        for name in set(causalrl.__all__) - _SURFACE_META
        if targets.get(name, name) not in documented
    )
    assert not missing, f"{len(missing)} exported name(s) absent from docs/api.md: {missing}"
