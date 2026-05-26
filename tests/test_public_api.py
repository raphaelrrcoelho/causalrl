"""The curated public API (`causalrl.__all__`) stays consistent and importable."""

import inspect

import causalrl


def test_every_exported_name_is_importable():
    for name in causalrl.__all__:
        assert hasattr(causalrl, name), f"{name!r} is in __all__ but not importable from causalrl"


def test_all_has_no_duplicates():
    assert len(causalrl.__all__) == len(set(causalrl.__all__))


def test_no_undeclared_public_symbols():
    # Every non-underscore, non-module attribute must be declared in __all__ (no accidental leaks).
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
    ):
        assert name in causalrl.__all__


def test_version_is_stamped():
    assert causalrl.__version__ == "0.3.0"
